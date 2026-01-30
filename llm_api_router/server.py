"""FastAPI server for LLM API Router."""

import json
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .config import RouterConfig, load_default_config
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    LLMError,
    NoAvailableProviderError,
    ProviderError,
    RateLimitError,
)
from .logging import get_logger
from .models import ProviderType
from .providers import BaseProvider
from .router import LLMRouter
from .stats import StatsCollector


def _format_timestamp(timestamp: float | None) -> str | None:
    """Convert a Unix timestamp to formatted datetime string with timezone.

    Args:
        timestamp: Unix timestamp (seconds since epoch)

    Returns:
        Formatted datetime string in ISO format with timezone, or None if timestamp is None
    """
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _format_providers(providers: list[Any]) -> list[dict[str, Any]]:
    """Format provider configurations for status endpoint."""
    return [
        {
            "name": provider.display_name,
            "priority": provider.priority,
            "base_url": provider.base_url,
            "model_mapping": provider.model_mapping,
        }
        for provider in providers
    ]


def _format_stats(stats: Any) -> dict[str, Any]:
    """Format router statistics for status endpoint."""
    return {
        "uptime_seconds": stats.uptime_seconds,
        "total_requests": stats.total_requests,
        "total_input_tokens": stats.total_input_tokens,
        "total_output_tokens": stats.total_output_tokens,
        "total_cached_tokens": stats.total_cached_tokens,
        "most_used_provider": stats.most_used_provider,
        "fastest_provider": stats.fastest_provider,
        "providers": {
            name: {
                "total_requests": stat.total_requests,
                "in_progress_requests": stat.in_progress_requests,
                "successful_requests": stat.successful_requests,
                "failed_requests": stat.failed_requests,
                "success_rate": round(stat.success_rate, 2),
                "failure_rate": round(stat.failure_rate, 2),
                "total_input_tokens": stat.total_input_tokens,
                "total_output_tokens": stat.total_output_tokens,
                "total_tokens": stat.total_tokens,
                "total_cached_tokens": stat.total_cached_tokens,
                "average_latency_ms": round(stat.average_latency_ms, 2),
                "tokens_per_second": round(stat.tokens_per_second, 2),
                "last_request_time": _format_timestamp(stat.last_request_time),
                "last_success_time": _format_timestamp(stat.last_success_time),
                "last_error": stat.last_error,
            }
            for name, stat in stats.providers.items()
        },
    }


def _create_stats_tracking_generator(
    original_generator: Any,
    provider: BaseProvider,
    stats_collector: StatsCollector,
    request_start_time: float,
    request_id: str,
    endpoint: str,
) -> Any:
    """Create a generator that tracks statistics from streaming chunks."""

    async def stats_tracking_generator() -> Any:
        """Generator that tracks statistics while yielding chunks."""
        accumulated_chunks = []

        try:
            async for chunk in original_generator:
                accumulated_chunks.append(chunk)
                # Yield original chunk
                yield chunk
        finally:
            # Calculate duration
            duration_ms = (time.time() - request_start_time) * 1000

            # Log the streaming response
            # Try to reconstruct a response dict from accumulated chunks for logging
            response_dict: dict[str, Any] = {}

            # Merge all chunks to reconstruct the full response
            full_resp = ""
            try:
                for chunk in accumulated_chunks:
                    assert isinstance(chunk, bytes)
                    chunk_text = chunk.decode("utf-8", errors="ignore")
                    full_resp += chunk_text
            except Exception:
                pass

            # parse "data: {...}" SSE format
            for line in full_resp.splitlines():
                try:
                    if line.startswith("data:"):
                        # Split by "data:" and process each chunk
                        data_str = line.removeprefix("data:").strip()
                        if not data_str or data_str == "[DONE]":
                            continue

                        try:
                            data = json.loads(data_str)
                            response_dict = provider.merge_streaming_chunk(
                                response_dict, data
                            )
                        except json.JSONDecodeError:
                            pass

                except Exception:
                    # Don't fail if we can't parse for logging
                    pass

            # Postprocess response (e.g., convert partial_json for Anthropic)
            response_dict = provider.postprocess_response(response_dict)

            # Extract token counts from response
            (
                total_input_tokens,
                total_output_tokens,
                cached_tokens,
            ) = provider.extract_tokens_from_response(response_dict)

            # Record statistics after streaming completes
            stats_collector.record_request_success(
                provider.provider_name,
                request_start_time,
                total_input_tokens,
                total_output_tokens,
                cached_tokens,
            )

            # Log the response
            logger = get_logger()
            logger.log_response(
                request_id=request_id,
                endpoint=endpoint,
                response=response_dict,
                provider_name=provider.provider_name,
                duration_ms=duration_ms,
            )

    return stats_tracking_generator()


class LLMAPIServer:
    """LLM API Server with separate endpoints for each provider type."""

    def __init__(self, config: RouterConfig | None = None):
        self.app = FastAPI(
            title="LLM API Router",
            description="Router for OpenAI and Anthropic APIs with provider fallback",
            version="0.1.0",
        )

        # Load configuration
        loaded_config = config or load_default_config()
        if loaded_config is None:
            raise ConfigurationError(
                "No configuration provided and no default config file found. "
                "Create a config file or pass a RouterConfig instance."
            )
        # Type narrowing - we know loaded_config is not None here
        self.config: RouterConfig = loaded_config

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ConfigurationError(f"Configuration errors: {', '.join(errors)}")

        # Initialize routers for each provider type
        self.openai_router: LLMRouter | None = None
        self.anthropic_router: LLMRouter | None = None

        self._setup_routes()
        self._setup_middleware()

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            return {
                "name": "LLM API Router",
                "version": "0.1.0",
                "endpoints": {
                    "openai": "/openai/chat/completions",
                    "anthropic": "/anthropic/v1/messages",
                    "anthropic_count_tokens": "/anthropic/v1/messages/count_tokens",
                    "health": "/health",
                    "status": "/status",
                },
            }

        @self.app.get("/health")
        async def health() -> dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "version": "0.1.0",
                "providers": {
                    "openai": len(self.config.openai_providers) > 0,
                    "anthropic": len(self.config.anthropic_providers) > 0,
                },
            }

        @self.app.get("/status")
        async def status() -> dict[str, Any]:
            """Detailed status endpoint showing provider configurations and statistics."""
            # Format providers
            openai_providers = _format_providers(self.config.openai_providers)
            anthropic_providers = _format_providers(self.config.anthropic_providers)

            # Get statistics from routers
            stats_data = {}

            for router_name, router in [
                ("openai", self.openai_router),
                ("anthropic", self.anthropic_router),
            ]:
                try:
                    if router:
                        stats_data[router_name] = _format_stats(router.get_stats())
                except Exception:
                    pass

            return {
                "status": "healthy",
                "config": {
                    "openai": {
                        "enabled": len(openai_providers) > 0,
                        "provider_count": len(openai_providers),
                        "providers": openai_providers,
                    },
                    "anthropic": {
                        "enabled": len(anthropic_providers) > 0,
                        "provider_count": len(anthropic_providers),
                        "providers": anthropic_providers,
                    },
                },
                "statistics": stats_data if stats_data else None,
            }

        @self.app.post("/openai/chat/completions", response_model=None)
        async def openai_chat_completion(
            request: dict[str, Any],
            router: LLMRouter = Depends(self._get_openai_router),
        ) -> dict[str, Any] | StreamingResponse:
            """OpenAI-compatible chat completion endpoint."""
            return await self._handle_chat_completion(
                request, router, ProviderType.OPENAI
            )

        @self.app.post("/anthropic/v1/messages", response_model=None)
        async def anthropic_chat_completion(
            request: dict[str, Any],
            router: LLMRouter = Depends(self._get_anthropic_router),
        ) -> dict[str, Any] | StreamingResponse:
            """Anthropic-compatible chat completion endpoint."""
            return await self._handle_chat_completion(
                request, router, ProviderType.ANTHROPIC
            )

        @self.app.post(
            "/anthropic/v1/messages/count_tokens", response_model=dict[str, Any]
        )
        async def anthropic_count_tokens(
            request: dict[str, Any],
            router: LLMRouter = Depends(self._get_anthropic_router),
        ) -> dict[str, Any]:
            """Anthropic-compatible count_tokens endpoint."""
            try:
                return await router.count_tokens(request)
            except Exception as e:
                # Re-raise HTTP exceptions
                if isinstance(e, HTTPException):
                    raise e
                # Convert other exceptions to appropriate HTTP errors
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error: {str(e)}",
                ) from e

    def _setup_middleware(self) -> None:
        """Setup middleware for error handling."""

        @self.app.exception_handler(NoAvailableProviderError)
        async def no_available_provider_handler(
            request: Any, exc: NoAvailableProviderError
        ) -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "no_available_provider",
                    }
                },
            )

        @self.app.exception_handler(RateLimitError)
        async def rate_limit_handler(request: Any, exc: RateLimitError) -> JSONResponse:
            headers = {}
            if exc.retry_after:
                headers["Retry-After"] = str(exc.retry_after)

            return JSONResponse(
                status_code=429,
                headers=headers,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "rate_limit_exceeded",
                    }
                },
            )

        @self.app.exception_handler(AuthenticationError)
        async def authentication_handler(
            request: Any, exc: AuthenticationError
        ) -> JSONResponse:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "authentication_error",
                    }
                },
            )

        @self.app.exception_handler(ProviderError)
        async def provider_error_handler(
            request: Any, exc: ProviderError
        ) -> JSONResponse:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "provider_error",
                        "provider": exc.provider,
                    }
                },
            )

        @self.app.exception_handler(LLMError)
        async def llm_error_handler(request: Any, exc: LLMError) -> JSONResponse:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "llm_error",
                        "provider": exc.provider,
                    }
                },
            )

    async def _get_openai_router(self) -> LLMRouter:
        """Get or create OpenAI router."""
        if self.openai_router is None:
            if not self.config.openai_providers:
                raise HTTPException(
                    status_code=500,
                    detail="No OpenAI providers configured",
                )
            self.openai_router = LLMRouter(
                self.config.openai_providers, endpoint="openai"
            )
        return self.openai_router

    async def _get_anthropic_router(self) -> LLMRouter:
        """Get or create Anthropic router."""
        if self.anthropic_router is None:
            if not self.config.anthropic_providers:
                raise HTTPException(
                    status_code=500,
                    detail="No Anthropic providers configured",
                )
            self.anthropic_router = LLMRouter(
                self.config.anthropic_providers, endpoint="anthropic"
            )
        return self.anthropic_router

    async def _handle_chat_completion(
        self,
        request: dict[str, Any],
        router: LLMRouter,
        expected_provider: ProviderType,
    ) -> dict[str, Any] | StreamingResponse:
        """Handle chat completion request with error handling."""
        try:
            response = await router.chat_completion(request)

            # Check if this is a streaming response
            if isinstance(response, dict) and response.get("_streaming"):
                # This is a streaming response
                generator = response.get("_generator")
                request_start_time = response.get("_request_start_time", time.time())
                request_id = response.get("_request_id", "")
                endpoint = response.get("_endpoint", "")
                provider = response.get("_provider")

                if not generator:
                    raise HTTPException(
                        status_code=500, detail="Streaming response missing generator"
                    )

                if not provider:
                    raise HTTPException(
                        status_code=500, detail="Streaming response missing provider"
                    )

                # Wrap generator with statistics tracking and logging
                stats_generator = _create_stats_tracking_generator(
                    generator,
                    provider,
                    router.stats,
                    request_start_time,
                    request_id,
                    endpoint,
                )

                return StreamingResponse(
                    stats_generator,
                    media_type="text/event-stream",
                    headers={
                        "X-Provider": provider.provider_name,
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            else:
                # Non-streaming response
                return response

        except Exception as e:
            # Re-raise HTTP exceptions
            if isinstance(e, HTTPException):
                raise e
            # Convert other exceptions to appropriate HTTP errors
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}",
            ) from e


def create_app(config: RouterConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    server = LLMAPIServer(config)
    return server.app
