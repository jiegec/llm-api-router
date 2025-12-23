"""FastAPI server for LLM API Router."""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse

from .models import (
    ProviderConfig,
    ProviderType,
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from .router import LLMRouter
from .config import RouterConfig, load_default_config
from .exceptions import (
    NoAvailableProviderError,
    RateLimitError,
    AuthenticationError,
    ProviderError,
    LLMError,
    ConfigurationError,
)


class LLMAPIServer:
    """LLM API Server with separate endpoints for each provider type."""

    def __init__(self, config: Optional[RouterConfig] = None):
        self.app = FastAPI(
            title="LLM API Router",
            description="Router for OpenAI and Anthropic APIs with provider fallback",
            version="0.1.0",
        )

        # Load configuration
        self.config = config or load_default_config()
        if self.config is None:
            raise ConfigurationError(
                "No configuration provided and no default config file found. "
                "Create a config file or pass a RouterConfig instance."
            )

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ConfigurationError(f"Configuration errors: {', '.join(errors)}")

        # Initialize routers for each provider type
        self.openai_router: Optional[LLMRouter] = None
        self.anthropic_router: Optional[LLMRouter] = None

        self._setup_routes()
        self._setup_middleware()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "name": "LLM API Router",
                "version": "0.1.0",
                "endpoints": {
                    "openai": "/openai/chat/completions",
                    "anthropic": "/anthropic/chat/completions",
                    "health": "/health",
                }
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @self.app.post("/openai/chat/completions")
        async def openai_chat_completion(
            request: ChatCompletionRequest,
            router: LLMRouter = Depends(self._get_openai_router),
        ):
            """OpenAI-compatible chat completion endpoint."""
            return await self._handle_chat_completion(request, router, ProviderType.OPENAI)
        
        @self.app.post("/anthropic/chat/completions")
        async def anthropic_chat_completion(
            request: ChatCompletionRequest,
            router: LLMRouter = Depends(self._get_anthropic_router),
        ):
            """Anthropic-compatible chat completion endpoint."""
            return await self._handle_chat_completion(request, router, ProviderType.ANTHROPIC)
    
    def _setup_middleware(self):
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
            self.openai_router = LLMRouter(self.config.openai_providers)
        return self.openai_router

    async def _get_anthropic_router(self) -> LLMRouter:
        """Get or create Anthropic router."""
        if self.anthropic_router is None:
            if not self.config.anthropic_providers:
                raise HTTPException(
                    status_code=500,
                    detail="No Anthropic providers configured",
                )
            self.anthropic_router = LLMRouter(self.config.anthropic_providers)
        return self.anthropic_router
    
    async def _handle_chat_completion(
        self,
        request: ChatCompletionRequest,
        router: LLMRouter,
        expected_provider: ProviderType,
    ) -> ChatCompletionResponse:
        """Handle chat completion request with error handling."""
        try:
            async with router:
                response = await router.chat_completion(request)
                return response
        except Exception as e:
            # Re-raise HTTP exceptions
            if isinstance(e, HTTPException):
                raise e
            # Convert other exceptions to appropriate HTTP errors
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}",
            )
    
def create_app(config: Optional[RouterConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    server = LLMAPIServer(config)
    return server.app


# For backward compatibility
app = create_app()