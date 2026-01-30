"""Statistics collection for LLM API Router."""

import time
from collections import defaultdict

from pydantic import BaseModel, Field


class ProviderStats(BaseModel):
    """Statistics for a single provider."""

    # Request/Response counts
    total_requests: int = Field(default=0, description="Total number of requests")
    in_progress_requests: int = Field(default=0, description="In-progress requests")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")

    # Request type counts
    streaming_requests: int = Field(default=0, description="Total streaming requests")
    non_streaming_requests: int = Field(
        default=0, description="Total non-streaming requests"
    )

    # Token usage
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    total_tokens: int = Field(default=0, description="Total tokens (input + output)")
    total_cached_tokens: int = Field(default=0, description="Total cached tokens")

    # Non-streaming performance metrics
    non_streaming_total_latency_ms: float = Field(
        default=0.0, description="Total latency for non-streaming requests in ms"
    )
    non_streaming_output_tokens: int = Field(
        default=0, description="Total output tokens for non-streaming requests"
    )

    # Streaming performance metrics
    streaming_total_latency_ms: float = Field(
        default=0.0, description="Total latency for streaming requests in ms"
    )
    streaming_output_tokens: int = Field(
        default=0, description="Total output tokens for streaming requests"
    )
    streaming_time_to_first_token_ms: float = Field(
        default=0.0, description="Total time to first token for streaming in ms"
    )
    streaming_time_to_first_token_count: int = Field(
        default=0, description="Count of requests with TTFT recorded"
    )
    # Time spent on token generation excluding TTFT
    streaming_generation_time_ms: float = Field(
        default=0.0, description="Total generation time excluding TTFT in ms"
    )

    # Legacy performance metrics (kept for backward compatibility)
    total_latency_ms: float = Field(
        default=0.0, description="Total latency in milliseconds"
    )

    # Last request info
    last_request_time: float | None = Field(
        default=None, description="Timestamp of last request"
    )
    last_success_time: float | None = Field(
        default=None, description="Timestamp of last successful request"
    )
    last_error: str | None = Field(default=None, description="Last error message")

    # Rate limit stats
    total_rate_limits: int = Field(
        default=0, description="Total number of rate limit errors"
    )
    recent_rate_limits: int = Field(
        default=0, description="Number of rate limits in current window"
    )
    cooldown_count: int = Field(
        default=0, description="Number of times provider entered cooldown"
    )
    total_cooldown_time_seconds: float = Field(
        default=0.0, description="Total time spent in cooldown"
    )
    in_cooldown: bool = Field(
        default=False, description="Whether provider is currently in cooldown"
    )
    cooldown_remaining_seconds: float = Field(
        default=0.0, description="Remaining seconds in cooldown (0 if not in cooldown)"
    )
    last_cooldown_start_time: float | None = Field(
        default=None, description="Timestamp when last cooldown started"
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate based on completed requests."""
        completed_requests = self.successful_requests + self.failed_requests
        if completed_requests == 0:
            return 0.0
        return (self.successful_requests / completed_requests) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate based on completed requests."""
        completed_requests = self.successful_requests + self.failed_requests
        if completed_requests == 0:
            return 0.0
        return (self.failed_requests / completed_requests) * 100

    # Non-streaming computed properties
    @property
    def non_streaming_average_latency_ms(self) -> float:
        """Calculate average latency for non-streaming requests."""
        if self.non_streaming_requests == 0:
            return 0.0
        return self.non_streaming_total_latency_ms / self.non_streaming_requests

    @property
    def non_streaming_tokens_per_second(self) -> float:
        """Calculate tokens/s for non-streaming requests."""
        if self.non_streaming_total_latency_ms <= 0:
            return 0.0
        duration_seconds = self.non_streaming_total_latency_ms / 1000
        return self.non_streaming_output_tokens / duration_seconds

    # Streaming computed properties
    @property
    def streaming_average_latency_ms(self) -> float:
        """Calculate average latency for streaming requests."""
        if self.streaming_requests == 0:
            return 0.0
        return self.streaming_total_latency_ms / self.streaming_requests

    @property
    def streaming_average_time_to_first_token_ms(self) -> float:
        """Calculate average time to first token."""
        if self.streaming_time_to_first_token_count == 0:
            return 0.0
        return (
            self.streaming_time_to_first_token_ms
            / self.streaming_time_to_first_token_count
        )

    @property
    def streaming_tokens_per_second_with_first_token(self) -> float:
        """Calculate tokens/s including time to first token."""
        if self.streaming_total_latency_ms <= 0:
            return 0.0
        duration_seconds = self.streaming_total_latency_ms / 1000
        return self.streaming_output_tokens / duration_seconds

    @property
    def streaming_tokens_per_second_without_first_token(self) -> float:
        """Calculate tokens/s excluding time to first token."""
        if self.streaming_generation_time_ms <= 0:
            return 0.0
        duration_seconds = self.streaming_generation_time_ms / 1000
        return self.streaming_output_tokens / duration_seconds

    # Legacy computed properties
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency across all requests."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def tokens_per_second(self) -> float:
        """Calculate average output tokens/s across all requests."""
        if self.total_latency_ms <= 0:
            return 0.0
        duration_seconds = self.total_latency_ms / 1000
        return self.total_output_tokens / duration_seconds


class RouterStats(BaseModel):
    """Overall router statistics."""

    start_time: float = Field(description="Router start timestamp")
    uptime_seconds: float = Field(default=0.0, description="Router uptime in seconds")

    # Provider statistics
    providers: dict[str, ProviderStats] = Field(
        default_factory=dict, description="Per-provider statistics"
    )

    # Overall statistics
    total_requests: int = Field(
        default=0, description="Total requests across all providers"
    )
    total_input_tokens: int = Field(
        default=0, description="Total input tokens across all providers"
    )
    total_output_tokens: int = Field(
        default=0, description="Total output tokens across all providers"
    )
    total_cached_tokens: int = Field(
        default=0, description="Total cached tokens across all providers"
    )
    total_tokens: int = Field(
        default=0, description="Total tokens (input + output) across all providers"
    )

    @property
    def most_used_provider(self) -> str | None:
        """Get the most used provider by request count."""
        if not self.providers:
            return None
        return max(self.providers.items(), key=lambda x: x[1].total_requests)[0]

    @property
    def fastest_provider(self) -> str | None:
        """Get the fastest provider by average latency."""
        providers_with_latency = {
            name: stats
            for name, stats in self.providers.items()
            if stats.average_latency_ms > 0
        }
        if not providers_with_latency:
            return None
        return min(
            providers_with_latency.items(), key=lambda x: x[1].average_latency_ms
        )[0]


class StatsCollector:
    """Thread-safe statistics collector for the router."""

    def __init__(self) -> None:
        self._stats = RouterStats(start_time=time.time())
        self._provider_stats: dict[str, ProviderStats] = defaultdict(
            lambda: ProviderStats()
        )

    def get_stats(self) -> RouterStats:
        """Get current statistics snapshot."""
        # Update uptime
        self._stats.uptime_seconds = time.time() - self._stats.start_time

        # Update overall totals
        self._stats.total_requests = sum(
            stats.total_requests for stats in self._provider_stats.values()
        )
        self._stats.total_input_tokens = sum(
            stats.total_input_tokens for stats in self._provider_stats.values()
        )
        self._stats.total_output_tokens = sum(
            stats.total_output_tokens for stats in self._provider_stats.values()
        )
        self._stats.total_cached_tokens = sum(
            stats.total_cached_tokens for stats in self._provider_stats.values()
        )
        self._stats.total_tokens = (
            self._stats.total_input_tokens + self._stats.total_output_tokens
        )

        # Copy provider stats
        self._stats.providers = dict(self._provider_stats)

        return self._stats

    def record_request_start(self, provider_name: str) -> float:
        """Record the start of a request."""
        stats = self._provider_stats[provider_name]
        stats.total_requests += 1
        stats.in_progress_requests += 1
        stats.last_request_time = time.time()
        return time.time()

    def record_request_success(
        self,
        provider_name: str,
        start_time: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        is_streaming: bool = False,
        time_to_first_token_ms: float | None = None,
    ) -> None:
        """Record a successful request.

        Args:
            provider_name: Name of the provider
            start_time: Request start timestamp (from time.time())
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens
            is_streaming: Whether this was a streaming request
            time_to_first_token_ms: Time to first output token in milliseconds
                                     (only for streaming requests)
        """
        stats = self._provider_stats[provider_name]
        stats.in_progress_requests -= 1
        stats.successful_requests += 1
        stats.total_input_tokens += input_tokens
        stats.total_output_tokens += output_tokens
        stats.total_tokens += input_tokens + output_tokens
        stats.total_cached_tokens += cached_tokens
        stats.last_success_time = time.time()
        stats.last_error = None

        # Note: total_requests is incremented in record_request_start

        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000

        # Update legacy/generic metrics (for backward compatibility)
        stats.total_latency_ms += total_latency_ms

        if is_streaming:
            # Update streaming-specific metrics
            stats.streaming_requests += 1
            stats.streaming_total_latency_ms += total_latency_ms
            stats.streaming_output_tokens += output_tokens

            # Time to first token metrics
            if time_to_first_token_ms is not None and time_to_first_token_ms > 0:
                stats.streaming_time_to_first_token_ms += time_to_first_token_ms
                stats.streaming_time_to_first_token_count += 1

                # Track generation time excluding TTFT for speed calculation
                generation_time_ms = total_latency_ms - time_to_first_token_ms
                if generation_time_ms > 0:
                    stats.streaming_generation_time_ms += generation_time_ms
        else:
            # Update non-streaming specific metrics
            stats.non_streaming_requests += 1
            stats.non_streaming_total_latency_ms += total_latency_ms
            stats.non_streaming_output_tokens += output_tokens

    def record_request_failure(self, provider_name: str, error_message: str) -> None:
        """Record a failed request."""
        stats = self._provider_stats[provider_name]
        stats.in_progress_requests -= 1
        stats.failed_requests += 1
        stats.last_error = error_message

    def record_rate_limit(self, provider_name: str) -> None:
        """Record a rate limit error for a provider."""
        stats = self._provider_stats[provider_name]
        stats.total_rate_limits += 1

    def record_cooldown_start(self, provider_name: str) -> None:
        """Record when a provider enters cooldown."""
        stats = self._provider_stats[provider_name]
        stats.cooldown_count += 1
        stats.in_cooldown = True
        stats.last_cooldown_start_time = time.time()

    def record_cooldown_end(self, provider_name: str) -> None:
        """Record when a provider exits cooldown."""
        stats = self._provider_stats[provider_name]
        if stats.last_cooldown_start_time:
            cooldown_duration = time.time() - stats.last_cooldown_start_time
            stats.total_cooldown_time_seconds += cooldown_duration
        stats.in_cooldown = False
        stats.last_cooldown_start_time = None

    def update_rate_limit_status(
        self,
        provider_name: str,
        recent_count: int,
        cooldown_remaining: float | None,
    ) -> None:
        """Update rate limit status for a provider.

        Args:
            provider_name: Name of the provider
            recent_count: Number of rate limits in current window
            cooldown_remaining: Remaining seconds in cooldown (None if not in cooldown)
        """
        stats = self._provider_stats[provider_name]
        stats.recent_rate_limits = recent_count
        if cooldown_remaining is None:
            cooldown_remaining = 0.0
            stats.in_cooldown = False
        else:
            stats.in_cooldown = True
        stats.cooldown_remaining_seconds = cooldown_remaining
