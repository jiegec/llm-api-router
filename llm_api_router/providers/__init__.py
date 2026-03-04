"""Provider implementations."""

from ..models import ProviderConfig, ProviderType
from .anthropic import AnthropicProvider
from .base import BaseProvider
from .openai import OpenAIProvider


def create_provider(config: ProviderConfig) -> BaseProvider:
    """Create a provider instance based on configuration."""
    if config.provider_type == ProviderType.OPENAI:
        return OpenAIProvider(config)
    elif config.provider_type == ProviderType.ANTHROPIC:
        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unsupported provider type: {config.provider_type}")


__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_provider",
]
