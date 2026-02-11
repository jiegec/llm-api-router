"""FastAPI server for LLM API Router."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .analytics import AnalyticsQuery
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
        "total_tokens": stats.total_tokens,
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
                "last_request_time": _format_timestamp(stat.last_request_time),
                "last_success_time": _format_timestamp(stat.last_success_time),
                "last_error": stat.last_error,
                # Rate limit stats
                "rate_limits": {
                    "total_rate_limits": stat.total_rate_limits,
                    "recent_rate_limits": stat.recent_rate_limits,
                    "cooldown_count": stat.cooldown_count,
                    "total_cooldown_time_seconds": round(
                        stat.total_cooldown_time_seconds, 2
                    ),
                    "in_cooldown": stat.in_cooldown,
                    "cooldown_remaining_seconds": round(
                        stat.cooldown_remaining_seconds, 2
                    ),
                    "last_cooldown_start_time": _format_timestamp(
                        stat.last_cooldown_start_time
                    ),
                },
                # Non-streaming metrics
                "non_streaming": {
                    "requests": stat.non_streaming_requests,
                    "average_latency_ms": round(
                        stat.non_streaming_average_latency_ms, 2
                    ),
                    "tokens_per_second": round(stat.non_streaming_tokens_per_second, 2),
                },
                # Streaming metrics
                "streaming": {
                    "requests": stat.streaming_requests,
                    "average_time_to_first_token_ms": round(
                        stat.streaming_average_time_to_first_token_ms, 2
                    ),
                    "average_total_latency_ms": round(
                        stat.streaming_average_latency_ms, 2
                    ),
                    "tokens_per_second_with_first_token": round(
                        stat.streaming_tokens_per_second_with_first_token, 2
                    ),
                    "tokens_per_second_without_first_token": round(
                        stat.streaming_tokens_per_second_without_first_token, 2
                    ),
                },
                # Legacy metrics (kept for backward compatibility)
                "average_latency_ms": round(stat.average_latency_ms, 2),
                "tokens_per_second": round(stat.tokens_per_second, 2),
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
        first_chunk_time: float | None = None

        try:
            async for chunk in original_generator:
                # Track time when first chunk is received (time to first token)
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                accumulated_chunks.append(chunk)
                # Yield original chunk
                yield chunk
        finally:
            # Calculate duration
            end_time = time.time()
            duration_ms = (end_time - request_start_time) * 1000

            # Calculate time to first token
            time_to_first_token_ms: float | None = None
            if first_chunk_time is not None:
                time_to_first_token_ms = (first_chunk_time - request_start_time) * 1000

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
                is_streaming=True,
                time_to_first_token_ms=time_to_first_token_ms,
            )

            # Log the response
            logger = get_logger()
            logger.log_response(
                request_id=request_id,
                endpoint=endpoint,
                response=response_dict,
                provider_name=provider.provider_name,
                duration_ms=duration_ms,
                provider=provider,
                time_to_first_token_ms=time_to_first_token_ms,
                is_streaming=True,
            )

    return stats_tracking_generator()


class LLMAPIServer:
    """LLM API Server with separate endpoints for each provider type."""

    def __init__(self, config: RouterConfig | None = None):
        self.app = FastAPI(
            title="LLM API Router",
            description="Router for OpenAI and Anthropic APIs with provider fallback",
            version=__version__,
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
                "version": __version__,
                "endpoints": {
                    "openai": "/openai/chat/completions",
                    "anthropic": "/anthropic/v1/messages",
                    "anthropic_count_tokens": "/anthropic/v1/messages/count_tokens",
                    "health": "/health",
                    "status": "/status",
                    "metrics": "/metrics",
                    "web": "/web",
                    "analytics_requests": "/analytics/requests",
                    "analytics_tokens": "/analytics/tokens",
                    "analytics_latency": "/analytics/latency",
                    "analytics_summary": "/analytics/summary",
                    "analytics_timerange": "/analytics/timerange",
                },
            }

        @self.app.get("/health")
        async def health() -> dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "version": __version__,
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

        @self.app.get("/metrics", response_class=PlainTextResponse)
        async def metrics() -> str:
            """Prometheus-compatible metrics endpoint."""
            lines: list[str] = []

            # Collect stats from both routers
            all_stats: dict[str, Any] = {}
            for router_name, router in [
                ("openai", self.openai_router),
                ("anthropic", self.anthropic_router),
            ]:
                try:
                    if router:
                        all_stats[router_name] = router.get_stats()
                except Exception:
                    pass

            if not all_stats:
                return "# No metrics available\n"

            # Helper to format metric lines
            declared_metrics: set[str] = set()

            def metric(
                name: str,
                value: float | int,
                labels: dict[str, str] | None = None,
                help_text: str | None = None,
                metric_type: str | None = None,
            ) -> None:
                # Only write HELP and TYPE once per unique metric name
                if help_text and name not in declared_metrics:
                    lines.append(f"# HELP {name} {help_text}")
                if metric_type and name not in declared_metrics:
                    lines.append(f"# TYPE {name} {metric_type}")
                label_str = ""
                if labels:
                    label_parts = [f'{k}="{v}"' for k, v in labels.items()]
                    label_str = "{" + ",".join(label_parts) + "}"
                lines.append(f"{name}{label_str} {value}")
                declared_metrics.add(name)

            # Router info
            metric(
                "llm_router_info",
                1,
                {"version": __version__},
                "LLM API Router information",
                "gauge",
            )

            # Per-provider-type metrics
            for provider_type, stats in all_stats.items():
                # Uptime
                metric(
                    "llm_router_uptime_seconds",
                    stats.uptime_seconds,
                    {"provider_type": provider_type},
                    "Router uptime in seconds",
                    "gauge",
                )

                # Overall totals
                metric(
                    "llm_router_requests_total",
                    stats.total_requests,
                    {"provider_type": provider_type},
                    "Total number of requests",
                    "counter",
                )
                metric(
                    "llm_router_input_tokens_total",
                    stats.total_input_tokens,
                    {"provider_type": provider_type},
                    "Total number of input tokens",
                    "counter",
                )
                metric(
                    "llm_router_output_tokens_total",
                    stats.total_output_tokens,
                    {"provider_type": provider_type},
                    "Total number of output tokens",
                    "counter",
                )
                metric(
                    "llm_router_cached_tokens_total",
                    stats.total_cached_tokens,
                    {"provider_type": provider_type},
                    "Total number of cached tokens",
                    "counter",
                )
                metric(
                    "llm_router_total_tokens",
                    stats.total_input_tokens + stats.total_output_tokens,
                    {"provider_type": provider_type},
                    "Total number of tokens (input + output)",
                    "counter",
                )

                # Per-provider metrics
                if stats.providers:
                    for provider_name, pstats in stats.providers.items():
                        labels = {
                            "provider_type": provider_type,
                            "provider": provider_name,
                        }

                        # Request counts
                        metric(
                            "llm_router_provider_requests_total",
                            pstats.total_requests,
                            labels,
                            "Total requests per provider",
                            "counter",
                        )
                        metric(
                            "llm_router_provider_requests_in_progress",
                            pstats.in_progress_requests,
                            labels,
                            "In-progress requests per provider",
                            "gauge",
                        )
                        metric(
                            "llm_router_provider_requests_successful_total",
                            pstats.successful_requests,
                            labels,
                            "Successful requests per provider",
                            "counter",
                        )
                        metric(
                            "llm_router_provider_requests_failed_total",
                            pstats.failed_requests,
                            labels,
                            "Failed requests per provider",
                            "counter",
                        )

                        # Token counts
                        metric(
                            "llm_router_provider_input_tokens_total",
                            pstats.total_input_tokens,
                            labels,
                            "Input tokens per provider",
                            "counter",
                        )
                        metric(
                            "llm_router_provider_output_tokens_total",
                            pstats.total_output_tokens,
                            labels,
                            "Output tokens per provider",
                            "counter",
                        )
                        metric(
                            "llm_router_provider_cached_tokens_total",
                            pstats.total_cached_tokens,
                            labels,
                            "Cached tokens per provider",
                            "counter",
                        )

                        # Streaming metrics
                        metric(
                            "llm_router_provider_streaming_requests_total",
                            pstats.streaming_requests,
                            labels,
                            "Streaming requests per provider",
                            "counter",
                        )
                        metric(
                            "llm_router_provider_non_streaming_requests_total",
                            pstats.non_streaming_requests,
                            labels,
                            "Non-streaming requests per provider",
                            "counter",
                        )

                        # Latency metrics (in seconds for Prometheus convention)
                        if pstats.average_latency_ms > 0:
                            metric(
                                "llm_router_provider_latency_seconds",
                                pstats.average_latency_ms / 1000,
                                labels,
                                "Average latency per provider",
                                "gauge",
                            )
                        if pstats.streaming_average_latency_ms > 0:
                            metric(
                                "llm_router_provider_streaming_latency_seconds",
                                pstats.streaming_average_latency_ms / 1000,
                                labels,
                                "Average streaming latency per provider",
                                "gauge",
                            )
                        if pstats.non_streaming_average_latency_ms > 0:
                            metric(
                                "llm_router_provider_non_streaming_latency_seconds",
                                pstats.non_streaming_average_latency_ms / 1000,
                                labels,
                                "Average non-streaming latency per provider",
                                "gauge",
                            )

                        # Time to first token
                        if pstats.streaming_average_time_to_first_token_ms > 0:
                            metric(
                                "llm_router_provider_time_to_first_token_seconds",
                                pstats.streaming_average_time_to_first_token_ms / 1000,
                                labels,
                                "Average time to first token",
                                "gauge",
                            )

                        # Tokens per second
                        if pstats.tokens_per_second > 0:
                            metric(
                                "llm_router_provider_tokens_per_second",
                                pstats.tokens_per_second,
                                labels,
                                "Average tokens per second",
                                "gauge",
                            )
                        if pstats.streaming_tokens_per_second_with_first_token > 0:
                            metric(
                                "llm_router_provider_streaming_tokens_per_second",
                                pstats.streaming_tokens_per_second_with_first_token,
                                labels,
                                "Streaming tokens per second (with TTFT)",
                                "gauge",
                            )

                        # Success rate
                        metric(
                            "llm_router_provider_success_rate",
                            pstats.success_rate / 100,
                            labels,
                            "Success rate (0-1)",
                            "gauge",
                        )

                        # Rate limit metrics
                        metric(
                            "llm_router_provider_rate_limits_total",
                            pstats.total_rate_limits,
                            labels,
                            "Total rate limit errors per provider",
                            "counter",
                        )
                        metric(
                            "llm_router_provider_recent_rate_limits",
                            pstats.recent_rate_limits,
                            labels,
                            "Recent rate limits in current window",
                            "gauge",
                        )
                        metric(
                            "llm_router_provider_cooldowns_total",
                            pstats.cooldown_count,
                            labels,
                            "Total number of cooldowns entered",
                            "counter",
                        )
                        metric(
                            "llm_router_provider_cooldown_time_seconds_total",
                            pstats.total_cooldown_time_seconds,
                            labels,
                            "Total time spent in cooldown",
                            "counter",
                        )
                        metric(
                            "llm_router_provider_in_cooldown",
                            1 if pstats.in_cooldown else 0,
                            labels,
                            "Whether provider is currently in cooldown",
                            "gauge",
                        )
                        if pstats.cooldown_remaining_seconds > 0:
                            metric(
                                "llm_router_provider_cooldown_remaining_seconds",
                                pstats.cooldown_remaining_seconds,
                                labels,
                                "Remaining seconds in cooldown",
                                "gauge",
                            )

            return "\n".join(lines) + "\n"

        @self.app.get("/analytics/requests")
        async def analytics_requests(
            interval: str = "hour", hours: int = 24, provider_type: str | None = None
        ) -> list[dict[str, Any]]:
            """Get request count analytics over time.

            Args:
                interval: Time bucket size - 'minute', 'hour', or 'day'
                hours: Number of hours to look back
                provider_type: Optional filter by provider type ('openai', 'anthropic')
            """
            analytics = AnalyticsQuery()
            return analytics.get_requests_over_time(interval, hours, provider_type)

        @self.app.get("/analytics/tokens")
        async def analytics_tokens(
            interval: str = "hour", hours: int = 24, provider_type: str | None = None
        ) -> list[dict[str, Any]]:
            """Get token count analytics over time.

            Args:
                interval: Time bucket size - 'minute', 'hour', or 'day'
                hours: Number of hours to look back
                provider_type: Optional filter by provider type ('openai', 'anthropic')
            """
            analytics = AnalyticsQuery()
            return analytics.get_tokens_over_time(interval, hours, provider_type)

        @self.app.get("/analytics/latency")
        async def analytics_latency(
            interval: str = "hour", hours: int = 24, provider_type: str | None = None
        ) -> list[dict[str, Any]]:
            """Get latency analytics over time.

            Args:
                interval: Time bucket size - 'minute', 'hour', or 'day'
                hours: Number of hours to look back
                provider_type: Optional filter by provider type ('openai', 'anthropic')
            """
            analytics = AnalyticsQuery()
            return analytics.get_latency_over_time(interval, hours, provider_type)

        @self.app.get("/analytics/summary")
        async def analytics_summary(hours: int = 24) -> list[dict[str, Any]]:
            """Get summary statistics by provider.

            Args:
                hours: Number of hours to look back
            """
            analytics = AnalyticsQuery()
            return analytics.get_provider_summary(hours)

        @self.app.get("/analytics/timerange")
        async def analytics_timerange() -> dict[str, Any] | None:
            """Get the available time range of analytics data."""
            analytics = AnalyticsQuery()
            return analytics.get_available_time_range()

        @self.app.post("/openai/chat/completions", response_model=None)
        async def openai_chat_completion(
            request: dict[str, Any],
            http_request: Request,
            router: LLMRouter = Depends(self._get_openai_router),
        ) -> dict[str, Any] | StreamingResponse:
            """OpenAI-compatible chat completion endpoint."""
            user_agent = http_request.headers.get("user-agent")
            return await self._handle_chat_completion(request, router, user_agent)

        @self.app.post("/anthropic/v1/messages", response_model=None)
        async def anthropic_chat_completion(
            request: dict[str, Any],
            http_request: Request,
            router: LLMRouter = Depends(self._get_anthropic_router),
        ) -> dict[str, Any] | StreamingResponse:
            """Anthropic-compatible chat completion endpoint."""
            user_agent = http_request.headers.get("user-agent")
            return await self._handle_chat_completion(request, router, user_agent)

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

        # Mount static files for status dashboard
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount(
                "/web", StaticFiles(directory=str(static_dir), html=True), name="static"
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
        user_agent: str | None = None,
    ) -> dict[str, Any] | StreamingResponse:
        """Handle chat completion request with error handling."""
        try:
            response = await router.chat_completion(request, user_agent=user_agent)

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
