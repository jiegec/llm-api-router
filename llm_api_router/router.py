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
from .rate_limiter import RateLimiter
from .stats import RouterStats, StatsCollector


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

        # Initialize statistics collection
        self.stats = StatsCollector()

        # Initialize rate limiter for tracking provider cooldowns
        self.rate_limiter = RateLimiter(stats_collector=self.stats)

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

            # Update rate limit status in stats (before checking cooldown)
            self.rate_limiter.update_rate_limit_stats(provider_name)

            # Check if provider is in cooldown
            if self.rate_limiter.is_in_cooldown(provider_name):
                remaining = self.rate_limiter.get_cooldown_remaining_seconds(
                    provider_name
                )
                self.logger.logger.info(
                    f"Skipping {provider_name} - in cooldown for {remaining:.0f}s"
                )
                errors.append((provider_name, f"In cooldown for {remaining:.0f}s"))
                continue

            # Log request to this specific provider
            self.logger.log_request(
                request_id=request_id,
                endpoint=self.endpoint,
                request=request,
                provider_name=provider_name,
                provider_priority=provider_config.priority,
            )

            # Record request start for statistics
            request_start_time = self.stats.record_request_start(provider_name)

            try:
                provider = await self._get_provider(provider_config)
                provider_start_time = time.time()
                result = await provider.chat_completion(request)
                provider_duration = (time.time() - provider_start_time) * 1000

                # Check if this is a streaming response
                if isinstance(result, dict) and result.get("_streaming"):
                    # For streaming responses, include request start time and context for statistics and logging
                    # Statistics and logging will be recorded after streaming completes
                    result["_request_start_time"] = request_start_time
                    result["_request_id"] = request_id
                    result["_endpoint"] = self.endpoint
                    result["_original_request"] = request
                    return result
                else:
                    # Non-streaming response
                    response = result

                    # Log successful response - use raw response for full logging
                    if response:
                        self.logger.log_response(
                            request_id=request_id,
                            endpoint=self.endpoint,
                            response=response,  # Use raw response instead of parsed
                            provider_name=provider_name,
                            duration_ms=provider_duration,
                            provider=provider,
                        )

                        # Record statistics for successful request
                        (
                            input_tokens,
                            output_tokens,
                            cached_tokens,
                        ) = provider.extract_tokens_from_response(response)
                        self.stats.record_request_success(
                            provider_name,
                            request_start_time,
                            input_tokens,
                            output_tokens,
                            cached_tokens,
                            is_streaming=False,
                        )

                        # Reset rate limit counter on success
                        self.rate_limiter.record_success(provider_name)

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

                # Record statistics for authentication failure
                self.stats.record_request_failure(
                    provider_name, f"Authentication error: {error_msg}"
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

                # Record statistics for rate limit failure
                self.stats.record_request_failure(
                    provider_name, f"Rate limit error: {error_msg}"
                )

                # Record rate limit for cooldown tracking (also updates stats)
                self.rate_limiter.record_rate_limit(provider_name)

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

                # Record statistics for provider error
                self.stats.record_request_failure(
                    provider_name, f"Provider error: {error_msg}"
                )

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

                # Record statistics for unexpected error
                self.stats.record_request_failure(
                    provider_name, f"Unexpected error: {error_msg}"
                )

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

    async def count_tokens(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Send a count_tokens request using priority-based routing.

        Args:
            request: Count tokens request

        Returns:
            Count tokens response from the first successful provider

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
            selected_provider="",
            selected_priority=0,
            available_providers=available_providers,
        )

        errors = []
        attempt = 0

        for provider_config in self.providers:
            attempt += 1
            provider_name = provider_config.display_name

            # Update rate limit status in stats (before checking cooldown)
            self.rate_limiter.update_rate_limit_stats(provider_name)

            # Check if provider is in cooldown
            if self.rate_limiter.is_in_cooldown(provider_name):
                remaining = self.rate_limiter.get_cooldown_remaining_seconds(
                    provider_name
                )
                self.logger.logger.info(
                    f"Skipping {provider_name} - in cooldown for {remaining:.0f}s"
                )
                errors.append((provider_name, f"In cooldown for {remaining:.0f}s"))
                continue

            # Log request to this specific provider
            self.logger.log_request(
                request_id=request_id,
                endpoint=f"{self.endpoint}/count_tokens",
                request=request,
                provider_name=provider_name,
                provider_priority=provider_config.priority,
            )

            # Record request start for statistics
            request_start_time = self.stats.record_request_start(provider_name)

            try:
                provider = await self._get_provider(provider_config)
                provider_start_time = time.time()
                result = await provider.count_tokens(request)
                provider_duration = (time.time() - provider_start_time) * 1000

                # Log successful response
                if result:
                    self.logger.log_response(
                        request_id=request_id,
                        endpoint=f"{self.endpoint}/count_tokens",
                        response=result,
                        provider_name=provider_name,
                        duration_ms=provider_duration,
                        provider=provider,
                    )

                    # Record statistics for successful request
                    input_tokens = result.get("input_tokens", 0)
                    self.stats.record_request_success(
                        provider_name,
                        request_start_time,
                        input_tokens,
                        0,  # No output tokens for count_tokens
                        0,  # No cached tokens for count_tokens
                        is_streaming=False,
                    )

                # Log total request duration
                total_duration = (time.time() - start_time) * 1000
                self.logger.logger.info(
                    f"Request {request_id[:8]} completed in {total_duration:.0f}ms "
                    f"using {provider_name} (attempt {attempt})"
                )

                return result

            except AuthenticationError as e:
                # Authentication errors are fatal for this provider
                error_msg = str(e)
                errors.append((provider_name, error_msg))

                self.logger.log_error(
                    request_id=request_id,
                    endpoint=f"{self.endpoint}/count_tokens",
                    provider_name=provider_name,
                    error_type="authentication_error",
                    error_message=error_msg,
                    status_code=401,
                )

                # Record statistics for authentication failure
                self.stats.record_request_failure(
                    provider_name, f"Authentication error: {error_msg}"
                )

                self.logger.log_retry(
                    request_id=request_id,
                    endpoint=f"{self.endpoint}/count_tokens",
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

                # Record statistics for rate limit failure
                self.stats.record_request_failure(
                    provider_name, f"Rate limit error: {error_msg}"
                )

                # Record rate limit for cooldown tracking (also updates stats)
                self.rate_limiter.record_rate_limit(provider_name)

                self.logger.log_error(
                    request_id=request_id,
                    endpoint=f"{self.endpoint}/count_tokens",
                    provider_name=provider_name,
                    error_type="rate_limit_error",
                    error_message=error_msg,
                    status_code=429,
                )

                self.logger.log_retry(
                    request_id=request_id,
                    endpoint=f"{self.endpoint}/count_tokens",
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
                        endpoint=f"{self.endpoint}/count_tokens",
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

                # Record statistics for provider error
                self.stats.record_request_failure(
                    provider_name, f"Provider error: {error_msg}"
                )

                self.logger.log_error(
                    request_id=request_id,
                    endpoint=f"{self.endpoint}/count_tokens",
                    provider_name=provider_name,
                    error_type="provider_error",
                    error_message=error_msg,
                )

                self.logger.log_retry(
                    request_id=request_id,
                    endpoint=f"{self.endpoint}/count_tokens",
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
                        endpoint=f"{self.endpoint}/count_tokens",
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

                # Record statistics for unexpected error
                self.stats.record_request_failure(
                    provider_name, f"Unexpected error: {error_msg}"
                )

                self.logger.log_error(
                    request_id=request_id,
                    endpoint=f"{self.endpoint}/count_tokens",
                    provider_name=provider_name,
                    error_type="unexpected_error",
                    error_message=error_msg,
                )

                self.logger.log_retry(
                    request_id=request_id,
                    endpoint=f"{self.endpoint}/count_tokens",
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
                        endpoint=f"{self.endpoint}/count_tokens",
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
            endpoint=f"{self.endpoint}/count_tokens",
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

    def get_stats(self) -> RouterStats:
        """Get current router statistics."""
        return self.stats.get_stats()

    async def _schedule_provider_retry(
        self, config: ProviderConfig, retry_after: int
    ) -> None:
        """Schedule a provider to be retried after a delay."""
        await asyncio.sleep(retry_after)
        # The provider will be retried on the next request

    async def close(self) -> None:
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

    def get_rate_limiter_status(self) -> dict[str, dict[str, float | int | None]]:
        """
        Get the current status of the rate limiter for all providers.

        Returns:
            Dict mapping provider names to their rate limiter status:
            - in_cooldown: Whether the provider is in cooldown
            - cooldown_remaining: Seconds remaining in cooldown (None if not in cooldown)
            - recent_rate_limits: Number of rate limits in the current window
        """
        status = {}
        for provider in self.providers:
            provider_name = provider.display_name
            status[provider_name] = {
                "in_cooldown": self.rate_limiter.is_in_cooldown(provider_name),
                "cooldown_remaining": self.rate_limiter.get_cooldown_remaining_seconds(
                    provider_name
                ),
                "recent_rate_limits": self.rate_limiter.get_recent_rate_limit_count(
                    provider_name
                ),
            }
        return status
