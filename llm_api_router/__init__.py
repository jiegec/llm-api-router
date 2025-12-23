"""LLM API Router - A router for OpenAI and Anthropic APIs with automatic fallback."""

from .models import (
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ProviderConfig,
    ProviderType,
    Role,
    Choice,
    Usage,
)
from .router import LLMRouter
from .exceptions import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    ProviderError,
    NoAvailableProviderError,
    ConfigurationError,
)
from .config import RouterConfig, load_default_config, create_example_config

__version__ = "0.1.0"
__all__ = [
    "LLMRouter",
    "Message",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ProviderConfig",
    "ProviderType",
    "Role",
    "Choice",
    "Usage",
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "ProviderError",
    "NoAvailableProviderError",
    "ConfigurationError",
    "RouterConfig",
    "load_default_config",
    "create_example_config",
]