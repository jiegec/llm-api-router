"""FastAPI server for LLM API Router."""

import json
import time
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
from .models import (
    ProviderType,
)
from .router import LLMRouter


def _create_stats_tracking_generator(
    original_generator: Any,
    provider_name: str,
    stats_collector: Any,
    request_start_time: float,
) -> Any:
    """Create a generator that tracks statistics from streaming chunks."""

    async def stats_tracking_generator() -> Any:
        """Generator that tracks statistics while yielding chunks."""
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            async for chunk in original_generator:
                # Try to extract usage from chunk
                try:
                    # Parse chunk as text
                    if isinstance(chunk, bytes):
                        chunk_text = chunk.decode("utf-8", errors="ignore")
                    else:
                        chunk_text = str(chunk)

                    # OpenAI format: "data: {...}" SSE format
                    if chunk_text.startswith("data: "):
                        data_str = chunk_text[6:].strip()
                        if data_str == "[DONE]":
                            continue

                        try:
                            data = json.loads(data_str)
                            # Check for usage in chunk (typically in last chunk)
                            if "usage" in data:
                                usage = data["usage"]
                                total_input_tokens = usage.get("prompt_tokens", 0)
                                total_output_tokens = usage.get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            pass

                    # Anthropic format: JSON with usage field
                    elif "usage" in chunk_text:
                        try:
                            data = json.loads(chunk_text)
                            if "usage" in data:
                                usage = data["usage"]
                                total_input_tokens = usage.get("input_tokens", 0)
                                total_output_tokens = usage.get("output_tokens", 0)
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    # Don't fail streaming if we can't parse stats
                    pass

                # Yield original chunk
                yield chunk
        finally:
            # Record statistics after streaming completes
            stats_collector.record_request_success(
                provider_name,
                request_start_time,
                total_input_tokens,
                total_output_tokens,
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
            openai_providers = [
                {
                    "name": provider.display_name,
                    "priority": provider.priority,
                    "base_url": provider.base_url,
                    "model_mapping": provider.model_mapping,
                }
                for provider in self.config.openai_providers
            ]

            anthropic_providers = [
                {
                    "name": provider.display_name,
                    "priority": provider.priority,
                    "base_url": provider.base_url,
                    "model_mapping": provider.model_mapping,
                }
                for provider in self.config.anthropic_providers
            ]

            # Get statistics from routers
            openai_stats = None
            anthropic_stats = None

            try:
                if hasattr(self, "openai_router") and self.openai_router:
                    openai_stats = self.openai_router.get_stats()
            except Exception:
                pass

            try:
                if hasattr(self, "anthropic_router") and self.anthropic_router:
                    anthropic_stats = self.anthropic_router.get_stats()
            except Exception:
                pass

            # Prepare statistics data
            stats_data = {}
            if openai_stats:
                stats_data["openai"] = {
                    "uptime_seconds": openai_stats.uptime_seconds,
                    "total_requests": openai_stats.total_requests,
                    "total_input_tokens": openai_stats.total_input_tokens,
                    "total_output_tokens": openai_stats.total_output_tokens,
                    "most_used_provider": openai_stats.most_used_provider,
                    "fastest_provider": openai_stats.fastest_provider,
                    "providers": {
                        name: {
                            "total_requests": stats.total_requests,
                            "successful_requests": stats.successful_requests,
                            "failed_requests": stats.failed_requests,
                            "success_rate": round(stats.success_rate, 2),
                            "failure_rate": round(stats.failure_rate, 2),
                            "total_input_tokens": stats.total_input_tokens,
                            "total_output_tokens": stats.total_output_tokens,
                            "total_tokens": stats.total_tokens,
                            "average_latency_ms": round(stats.average_latency_ms, 2),
                            "tokens_per_second": round(stats.tokens_per_second, 2),
                            "last_request_time": stats.last_request_time,
                            "last_success_time": stats.last_success_time,
                            "last_error": stats.last_error,
                        }
                        for name, stats in openai_stats.providers.items()
                    },
                }

            if anthropic_stats:
                stats_data["anthropic"] = {
                    "uptime_seconds": anthropic_stats.uptime_seconds,
                    "total_requests": anthropic_stats.total_requests,
                    "total_input_tokens": anthropic_stats.total_input_tokens,
                    "total_output_tokens": anthropic_stats.total_output_tokens,
                    "most_used_provider": anthropic_stats.most_used_provider,
                    "fastest_provider": anthropic_stats.fastest_provider,
                    "providers": {
                        name: {
                            "total_requests": stats.total_requests,
                            "successful_requests": stats.successful_requests,
                            "failed_requests": stats.failed_requests,
                            "success_rate": round(stats.success_rate, 2),
                            "failure_rate": round(stats.failure_rate, 2),
                            "total_input_tokens": stats.total_input_tokens,
                            "total_output_tokens": stats.total_output_tokens,
                            "total_tokens": stats.total_tokens,
                            "average_latency_ms": round(stats.average_latency_ms, 2),
                            "tokens_per_second": round(stats.tokens_per_second, 2),
                            "last_request_time": stats.last_request_time,
                            "last_success_time": stats.last_success_time,
                            "last_error": stats.last_error,
                        }
                        for name, stats in anthropic_stats.providers.items()
                    },
                }

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
                provider_name = response.get("_provider", "unknown")
                generator = response.get("_generator")
                request_start_time = response.get("_request_start_time", time.time())

                if not generator:
                    raise HTTPException(
                        status_code=500, detail="Streaming response missing generator"
                    )

                # Wrap generator with statistics tracking
                stats_generator = _create_stats_tracking_generator(
                    generator,
                    provider_name,
                    router.stats,
                    request_start_time,
                )

                return StreamingResponse(
                    stats_generator,
                    media_type="text/event-stream",
                    headers={
                        "X-Provider": provider_name,
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
