"""Main LLM router with priority-based fallback."""

import asyncio
from contextlib import AsyncExitStack

from .exceptions import (
    AuthenticationError,
    LLMError,
    NoAvailableProviderError,
    ProviderError,
    RateLimitError,
)
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ProviderConfig,
    ProviderType,
)
from .providers import BaseProvider, create_provider


class LLMRouter:
    """LLM API router with priority-based fallback."""

    def __init__(self, providers: list[ProviderConfig]):
        """
        Initialize the router with provider configurations.

        Args:
            providers: List of provider configurations, sorted by priority
                      (lower priority number = higher priority)
        """
        if not providers:
            raise ValueError("At least one provider must be configured")

        # Sort providers by priority (lower number = higher priority)
        self.providers = sorted(providers, key=lambda p: p.priority)
        self._provider_instances: dict[str, BaseProvider] = {}
        self._exit_stack = AsyncExitStack()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _get_provider(self, config: ProviderConfig) -> BaseProvider:
        """Get or create a provider instance."""
        provider_name = config.name.value if isinstance(config.name, ProviderType) else config.name
        provider_key = f"{provider_name}-{config.priority}"
        if provider_key not in self._provider_instances:
            provider = create_provider(config)
            self._provider_instances[provider_key] = provider
            await self._exit_stack.enter_async_context(provider)
        return self._provider_instances[provider_key]

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Send a chat completion request using priority-based routing.

        Args:
            request: Chat completion request

        Returns:
            Chat completion response from the first successful provider

        Raises:
            NoAvailableProviderError: If all providers fail
        """
        errors = []

        for provider_config in self.providers:
            try:
                provider = await self._get_provider(provider_config)
                response = await provider.chat_completion(request)
                return response

            except AuthenticationError as e:
                # Authentication errors are fatal for this provider
                provider_name = provider_config.name.value if isinstance(provider_config.name, ProviderType) else provider_config.name
                errors.append((provider_name, str(e)))
                continue

            except RateLimitError as e:
                # Rate limit errors - try next provider
                provider_name = provider_config.name.value if isinstance(provider_config.name, ProviderType) else provider_config.name
                errors.append((provider_name, f"Rate limited: {str(e)}"))
                if e.retry_after:
                    # Schedule retry after delay
                    asyncio.create_task(self._schedule_provider_retry(provider_config, e.retry_after))
                continue

            except (ProviderError, LLMError) as e:
                # Other provider errors - try next provider
                provider_name = provider_config.name.value if isinstance(provider_config.name, ProviderType) else provider_config.name
                errors.append((provider_name, str(e)))
                continue

            except Exception as e:
                # Unexpected errors - try next provider
                provider_name = provider_config.name.value if isinstance(provider_config.name, ProviderType) else provider_config.name
                errors.append((provider_name, f"Unexpected error: {str(e)}"))
                continue

        # All providers failed
        error_details = "\n".join([f"- {provider}: {error}" for provider, error in errors])
        raise NoAvailableProviderError(
            f"All providers failed:\n{error_details}"
        )

    async def _schedule_provider_retry(self, config: ProviderConfig, retry_after: int):
        """Schedule a provider to be retried after a delay."""
        await asyncio.sleep(retry_after)
        # The provider will be retried on the next request

    async def close(self):
        """Close all provider connections."""
        await self._exit_stack.aclose()

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return [
            p.name.value if isinstance(p.name, ProviderType) else p.name
            for p in self.providers
        ]

    def get_provider_priority(self, provider_name: str) -> int | None:
        """Get priority of a specific provider."""
        for provider in self.providers:
            provider_name_value = provider.name.value if isinstance(provider.name, ProviderType) else provider.name
            if provider_name_value == provider_name:
                return provider.priority
        return None
