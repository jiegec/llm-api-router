"""FastAPI server for LLM API Router."""

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
                    "anthropic": "/anthropic/chat/completions",
                    "health": "/health",
                },
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "version": "0.1.0",
                "providers": {
                    "openai": self.openai_router is not None,
                    "anthropic": self.anthropic_router is not None,
                },
            }

        @self.app.get("/status")
        async def status():
            """Detailed status endpoint showing provider configurations."""
            openai_providers = []
            anthropic_providers = []

            if self.openai_router:
                openai_providers = [
                    {
                        "name": (
                            provider.name.value
                            if hasattr(provider.name, "value")
                            else str(provider.name)
                        ),
                        "priority": provider.priority,
                        "base_url": provider.base_url,
                        "model_mapping": provider.model_mapping,
                    }
                    for provider in self.config.openai_providers
                ]

            if self.anthropic_router:
                anthropic_providers = [
                    {
                        "name": (
                            provider.name.value
                            if hasattr(provider.name, "value")
                            else str(provider.name)
                        ),
                        "priority": provider.priority,
                        "base_url": provider.base_url,
                        "model_mapping": provider.model_mapping,
                    }
                    for provider in self.config.anthropic_providers
                ]

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
            }

        @self.app.get("/metrics")
        async def metrics():
            """Simple metrics endpoint (placeholder for more advanced metrics)."""
            # In a real implementation, this would track:
            # - Total requests
            # - Success/failure rates
            # - Provider usage statistics
            # - Token usage
            # - Response times

            return {
                "metrics": {
                    "note": "Metrics collection is a placeholder. Implement proper metrics tracking as needed.",
                    "suggested_metrics": [
                        "total_requests",
                        "success_rate",
                        "provider_usage",
                        "average_response_time",
                        "token_usage_by_provider",
                        "fallback_count",
                    ],
                }
            }

        @self.app.post("/openai/chat/completions")
        async def openai_chat_completion(
            request: dict[str, Any],
            router: LLMRouter = Depends(self._get_openai_router),
        ):
            """OpenAI-compatible chat completion endpoint."""
            return await self._handle_chat_completion(
                request, router, ProviderType.OPENAI
            )

        @self.app.post("/anthropic/chat/completions")
        async def anthropic_chat_completion(
            request: dict[str, Any],
            router: LLMRouter = Depends(self._get_anthropic_router),
        ):
            """Anthropic-compatible chat completion endpoint."""
            return await self._handle_chat_completion(
                request, router, ProviderType.ANTHROPIC
            )

    def _setup_middleware(self) -> None:
        """Setup middleware for error handling."""

        @self.app.exception_handler(NoAvailableProviderError)
        async def no_available_provider_handler(request, exc):
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
        async def rate_limit_handler(request, exc):
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
        async def authentication_handler(request, exc):
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
        async def provider_error_handler(request, exc):
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
        async def llm_error_handler(request, exc):
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
                httpx_response = response.get("_response")

                if not httpx_response:
                    raise HTTPException(
                        status_code=500,
                        detail="Streaming response missing HTTP response"
                    )

                # Create a streaming response that forwards the chunks
                async def stream_generator():
                    # Stream chunks as they come in
                    async for chunk in httpx_response.aiter_bytes():
                        yield chunk

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                    headers={
                        "X-Provider": provider_name,
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
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
