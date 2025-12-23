"""LLM API Router - A router for OpenAI and Anthropic APIs with automatic fallback."""

from .models import Message, ChatCompletionRequest, ChatCompletionResponse, ProviderConfig
from .router import LLMRouter
from .exceptions import LLMError, RateLimitError, AuthenticationError, ProviderError
from .config import load_providers_from_env, create_router_from_env

__version__ = "0.1.0"
__all__ = [
    "LLMRouter",
    "Message",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ProviderConfig",
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "ProviderError",
    "load_providers_from_env",
    "create_router_from_env",
]