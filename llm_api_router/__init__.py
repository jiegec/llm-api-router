"""LLM API Router - A router for OpenAI and Anthropic APIs with automatic fallback."""

from .models import Message, ChatCompletionRequest, ChatCompletionResponse, ProviderConfig
from .router import LLMRouter
from .exceptions import LLMError, RateLimitError, AuthenticationError, ProviderError

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
]