"""Main LLM router with priority-based fallback."""

import asyncio
import time
import types
import uuid
from contextlib import AsyncExitStack
from typing import Any

from .exceptions import (
    AuthenticationError,
    LLMError,
    NoAvailableProviderError,
    ProviderError,
    RateLimitError,
)
from .logging import get_logger
from .models import (
    ProviderConfig,
    ProviderType,
)
from .providers import BaseProvider, create_provider


class LLMRouter:
    """LLM API router with priority-based fallback."""

    def __init__(self, providers: list[ProviderConfig], endpoint: str = "unknown"):
        """
        Initialize the router with provider configurations.

        Args:
            providers: List of provider configurations, sorted by priority
                      (lower priority number = higher priority)
            endpoint: The API endpoint this router serves (e.g., "openai", "anthropic")
        """
        if not providers:
            raise ValueError("At least one provider must be configured")

        # Sort providers by priority (lower number = higher priority)
        self.providers = sorted(providers, key=lambda p: p.priority)
        self.endpoint = endpoint
        self._provider_instances: dict[str, BaseProvider] = {}
        self._exit_stack = AsyncExitStack()
        self.logger = get_logger()

        # Log configuration
        provider_details = []
        for provider in self.providers:
            provider_name = (
                provider.name.value
                if isinstance(provider.name, ProviderType)
                else provider.name
            )
            provider_details.append(
                {
                    "name": provider_name,
                    "priority": provider.priority,
                    "model_mapping_count": len(provider.model_mapping),
                }
            )

        self.logger.log_configuration(
            "router_start",
            {
                "endpoint": endpoint,
                "provider_count": len(providers),
                "providers": provider_details,
            },
        )

    async def __aenter__(self) -> "LLMRouter":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def _get_provider(self, config: ProviderConfig) -> BaseProvider:
        """Get or create a provider instance."""
        provider_name = (
            config.name.value if isinstance(config.name, ProviderType) else config.name
        )
        provider_key = f"{provider_name}-{config.priority}"
        if provider_key not in self._provider_instances:
            provider = create_provider(config)
            self._provider_instances[provider_key] = provider
            await self._exit_stack.enter_async_context(provider)
        return self._provider_instances[provider_key]

    async def chat_completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Send a chat completion request using priority-based routing.

        Args:
            request: Chat completion request

        Returns:
            Chat completion response from the first successful provider

        Raises:
            NoAvailableProviderError: If all providers fail
        """
        # Generate a unique request ID for tracking
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Log the incoming request
        available_providers = []
        for provider_config in self.providers:
            provider_name = (
                provider_config.name.value
                if isinstance(provider_config.name, ProviderType)
                else provider_config.name
            )
            available_providers.append((provider_name, provider_config.priority))

        self.logger.log_provider_selection(
            request_id=request_id,
            endpoint=self.endpoint,
            selected_provider="",  # Will be set when we select one
            selected_priority=0,
            available_providers=available_providers,
        )

        errors = []
        attempt = 0

        for provider_config in self.providers:
            attempt += 1
            provider_name = provider_config.display_name

            # Log request to this specific provider
            self.logger.log_request(
                request_id=request_id,
                endpoint=self.endpoint,
                request=request,
                provider_name=provider_name,
                provider_priority=provider_config.priority,
            )

            try:
                provider = await self._get_provider(provider_config)
                provider_start_time = time.time()
                result = await provider.chat_completion(request)
                provider_duration = (time.time() - provider_start_time) * 1000

                # Response is now just the raw response
                response = result

                # Log successful response - use raw response for full logging
                if response:
                    self.logger.log_response(
                        request_id=request_id,
                        endpoint=self.endpoint,
                        response=response,  # Use raw response instead of parsed
                        provider_name=provider_name,
                        duration_ms=provider_duration,
                    )

                # Log total request duration
                total_duration = (time.time() - start_time) * 1000
                self.logger.logger.info(
                    f"Request {request_id[:8]} completed in {total_duration:.0f}ms "
                    f"using {provider_name} (attempt {attempt})"
                )

                return response

            except AuthenticationError as e:
                # Authentication errors are fatal for this provider
                error_msg = str(e)
                errors.append((provider_name, error_msg))

                self.logger.log_error(
                    request_id=request_id,
                    endpoint=self.endpoint,
                    provider_name=provider_name,
                    error_type="authentication_error",
                    error_message=error_msg,
                    status_code=401,
                )

                self.logger.log_retry(
                    request_id=request_id,
                    endpoint=self.endpoint,
                    provider_name=provider_name,
                    provider_priority=provider_config.priority,
                    attempt=attempt,
                    max_attempts=len(self.providers),
                    error_type="authentication_error",
                    error_message=error_msg,
                )

                continue

            except RateLimitError as e:
                # Rate limit errors - try next provider
                error_msg = str(e)
                errors.append((provider_name, f"Rate limited: {error_msg}"))

                self.logger.log_error(
                    request_id=request_id,
                    endpoint=self.endpoint,
                    provider_name=provider_name,
                    error_type="rate_limit_error",
                    error_message=error_msg,
                    status_code=429,
                )

                self.logger.log_retry(
                    request_id=request_id,
                    endpoint=self.endpoint,
                    provider_name=provider_name,
                    provider_priority=provider_config.priority,
                    attempt=attempt,
                    max_attempts=len(self.providers),
                    error_type="rate_limit_error",
                    error_message=error_msg,
                    retry_after=e.retry_after,
                )

                if e.retry_after:
                    # Schedule retry after delay
                    asyncio.create_task(
                        self._schedule_provider_retry(provider_config, e.retry_after)
                    )

                # Log fallback if there are more providers
                if attempt < len(self.providers):
                    next_provider = self.providers[attempt]
                    next_provider_name = (
                        next_provider.name.value
                        if isinstance(next_provider.name, ProviderType)
                        else next_provider.name
                    )
                    self.logger.log_fallback(
                        request_id=request_id,
                        endpoint=self.endpoint,
                        from_provider=provider_name,
                        from_priority=provider_config.priority,
                        to_provider=next_provider_name,
                        to_priority=next_provider.priority,
                        reason="rate_limit",
                    )

                continue

            except (ProviderError, LLMError) as e:
                # Other provider errors - try next provider
                error_msg = str(e)
                errors.append((provider_name, error_msg))

                self.logger.log_error(
                    request_id=request_id,
                    endpoint=self.endpoint,
                    provider_name=provider_name,
                    error_type="provider_error",
                    error_message=error_msg,
                )

                self.logger.log_retry(
                    request_id=request_id,
                    endpoint=self.endpoint,
                    provider_name=provider_name,
                    provider_priority=provider_config.priority,
                    attempt=attempt,
                    max_attempts=len(self.providers),
                    error_type="provider_error",
                    error_message=error_msg,
                )

                # Log fallback if there are more providers
                if attempt < len(self.providers):
                    next_provider = self.providers[attempt]
                    next_provider_name = (
                        next_provider.name.value
                        if isinstance(next_provider.name, ProviderType)
                        else next_provider.name
                    )
                    self.logger.log_fallback(
                        request_id=request_id,
                        endpoint=self.endpoint,
                        from_provider=provider_name,
                        from_priority=provider_config.priority,
                        to_provider=next_provider_name,
                        to_priority=next_provider.priority,
                        reason="provider_error",
                    )

                continue

            except Exception as e:
                # Unexpected errors - try next provider
                error_msg = str(e)
                errors.append((provider_name, f"Unexpected error: {error_msg}"))

                self.logger.log_error(
                    request_id=request_id,
                    endpoint=self.endpoint,
                    provider_name=provider_name,
                    error_type="unexpected_error",
                    error_message=error_msg,
                )

                self.logger.log_retry(
                    request_id=request_id,
                    endpoint=self.endpoint,
                    provider_name=provider_name,
                    provider_priority=provider_config.priority,
                    attempt=attempt,
                    max_attempts=len(self.providers),
                    error_type="unexpected_error",
                    error_message=error_msg,
                )

                # Log fallback if there are more providers
                if attempt < len(self.providers):
                    next_provider = self.providers[attempt]
                    next_provider_name = (
                        next_provider.name.value
                        if isinstance(next_provider.name, ProviderType)
                        else next_provider.name
                    )
                    self.logger.log_fallback(
                        request_id=request_id,
                        endpoint=self.endpoint,
                        from_provider=provider_name,
                        from_priority=provider_config.priority,
                        to_provider=next_provider_name,
                        to_priority=next_provider.priority,
                        reason="unexpected_error",
                    )

                continue

        # All providers failed
        error_details = "\n".join(
            [f"- {provider}: {error}" for provider, error in errors]
        )

        self.logger.log_error(
            request_id=request_id,
            endpoint=self.endpoint,
            provider_name="all",
            error_type="no_available_provider",
            error_message=f"All {len(self.providers)} providers failed",
        )

        total_duration = (time.time() - start_time) * 1000
        self.logger.logger.error(
            f"Request {request_id[:8]} failed after {total_duration:.0f}ms: "
            f"all {len(self.providers)} providers failed"
        )

        raise NoAvailableProviderError(f"All providers failed:\n{error_details}")

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
            provider_name_value = (
                provider.name.value
                if isinstance(provider.name, ProviderType)
                else provider.name
            )
            if provider_name_value == provider_name:
                return provider.priority
        return None
