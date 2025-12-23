"""Provider implementations."""

from .base import BaseProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from ..models import ProviderType, ProviderConfig


def create_provider(config: ProviderConfig) -> BaseProvider:
    """Create a provider instance based on configuration."""
    if config.name == ProviderType.OPENAI:
        return OpenAIProvider(config)
    elif config.name == ProviderType.ANTHROPIC:
        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unsupported provider type: {config.name}")


__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_provider",
]