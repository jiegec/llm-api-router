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

    # Token usage
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    total_tokens: int = Field(default=0, description="Total tokens (input + output)")
    total_cached_tokens: int = Field(default=0, description="Total cached tokens")

    # Performance metrics
    total_latency_ms: float = Field(
        default=0.0, description="Total latency in milliseconds"
    )
    average_latency_ms: float = Field(
        default=0.0, description="Average latency in milliseconds"
    )
    tokens_per_second: float = Field(
        default=0.0, description="Average output tokens per second"
    )

    # Last request info
    last_request_time: float | None = Field(
        default=None, description="Timestamp of last request"
    )
    last_success_time: float | None = Field(
        default=None, description="Timestamp of last successful request"
    )
    last_error: str | None = Field(default=None, description="Last error message")

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
    ) -> None:
        """Record a successful request."""
        stats = self._provider_stats[provider_name]
        stats.in_progress_requests -= 1
        stats.successful_requests += 1
        stats.total_input_tokens += input_tokens
        stats.total_output_tokens += output_tokens
        stats.total_tokens += input_tokens + output_tokens
        stats.total_cached_tokens += cached_tokens
        stats.last_success_time = time.time()
        stats.last_error = None

        # Update performance metrics
        latency_ms = (time.time() - start_time) * 1000
        stats.total_latency_ms += latency_ms
        stats.average_latency_ms = stats.total_latency_ms / stats.successful_requests

        # Calculate average tokens per second (output tokens only)
        # Average of (output_tokens / request_duration) across all successful requests
        if latency_ms > 0 and output_tokens > 0:
            duration_seconds = latency_ms / 1000
            current_tokens_per_second = output_tokens / duration_seconds
            # Recalculate average across all successful requests
            # New average = (old_average * (n-1) + new_value) / n
            stats.tokens_per_second = (
                stats.tokens_per_second * (stats.successful_requests - 1)
                + current_tokens_per_second
            ) / stats.successful_requests

    def record_request_failure(self, provider_name: str, error_message: str) -> None:
        """Record a failed request."""
        stats = self._provider_stats[provider_name]
        stats.in_progress_requests -= 1
        stats.failed_requests += 1
        stats.last_error = error_message
