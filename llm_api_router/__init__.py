"""LLM API Router - A router for OpenAI and Anthropic APIs with automatic fallback."""

from .config import RouterConfig, create_example_config, load_default_config
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    LLMError,
    NoAvailableProviderError,
    ProviderError,
    RateLimitError,
)
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    ProviderConfig,
    ProviderType,
    Role,
    Usage,
)
from .router import LLMRouter

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
