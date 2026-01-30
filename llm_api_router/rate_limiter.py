"""Rate limit tracking with cooldown for LLM providers."""

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateLimitTracker:
    """Track rate limit events for a single provider."""

    # Timestamps of recent rate limit errors (within the 1-minute window)
    recent_rate_limits: deque[float] = field(default_factory=deque)

    # Timestamp when the cooldown period ends (None if not in cooldown)
    cooldown_until: float | None = None


class RateLimiter:
    """
    Track provider rate limits and enforce cooldown periods.

    Rules:
    - If a provider gets 3 consecutive rate limits within a 1-minute window,
      it enters a 10-minute cooldown.
    - During cooldown, the provider is skipped entirely.
    - A successful request resets the rate limit counter.
    """

    def __init__(
        self,
        consecutive_threshold: int = 3,
        window_seconds: int = 60,
        cooldown_seconds: int = 600,
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            consecutive_threshold: Number of consecutive rate limits to trigger cooldown
            window_seconds: Time window in seconds to check for consecutive rate limits
            cooldown_seconds: How long to skip the provider after threshold is reached
        """
        self.consecutive_threshold = consecutive_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self._trackers: dict[str, RateLimitTracker] = {}

    def _get_tracker(self, provider_name: str) -> RateLimitTracker:
        """Get or create a tracker for a provider."""
        if provider_name not in self._trackers:
            self._trackers[provider_name] = RateLimitTracker()
        return self._trackers[provider_name]

    def is_in_cooldown(self, provider_name: str) -> bool:
        """
        Check if a provider is currently in cooldown.

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if the provider is in cooldown and should be skipped
        """
        tracker = self._get_tracker(provider_name)
        if tracker.cooldown_until is None:
            return False

        # Check if cooldown has expired
        if time.time() >= tracker.cooldown_until:
            tracker.cooldown_until = None
            return False

        return True

    def record_rate_limit(self, provider_name: str) -> None:
        """
        Record a rate limit error for a provider.

        Args:
            provider_name: Name of the provider that hit rate limit
        """
        tracker = self._get_tracker(provider_name)
        current_time = time.time()

        # Clean up old rate limits outside the window
        while (
            tracker.recent_rate_limits
            and current_time - tracker.recent_rate_limits[0] > self.window_seconds
        ):
            tracker.recent_rate_limits.popleft()

        # Add the new rate limit
        tracker.recent_rate_limits.append(current_time)

        # Check if we've hit the threshold
        if len(tracker.recent_rate_limits) >= self.consecutive_threshold:
            # Start cooldown
            tracker.cooldown_until = current_time + self.cooldown_seconds
            # Clear the rate limits since we're now in cooldown
            tracker.recent_rate_limits.clear()

    def record_success(self, provider_name: str) -> None:
        """
        Record a successful request for a provider.

        This resets the rate limit counter for the provider.

        Args:
            provider_name: Name of the provider that had a successful request
        """
        tracker = self._get_tracker(provider_name)
        # Clear rate limits on success - provider is healthy again
        tracker.recent_rate_limits.clear()

    def get_cooldown_end_time(self, provider_name: str) -> float | None:
        """
        Get the timestamp when a provider's cooldown ends.

        Args:
            provider_name: Name of the provider to check

        Returns:
            Unix timestamp of cooldown end, or None if not in cooldown
        """
        tracker = self._get_tracker(provider_name)
        return tracker.cooldown_until

    def get_cooldown_remaining_seconds(self, provider_name: str) -> float | None:
        """
        Get the remaining cooldown time in seconds.

        Args:
            provider_name: Name of the provider to check

        Returns:
            Remaining seconds in cooldown, or None if not in cooldown
        """
        end_time = self.get_cooldown_end_time(provider_name)
        if end_time is None:
            return None
        remaining = end_time - time.time()
        return max(0.0, remaining)

    def get_recent_rate_limit_count(self, provider_name: str) -> int:
        """
        Get the number of rate limits in the current window for a provider.

        Args:
            provider_name: Name of the provider to check

        Returns:
            Number of rate limits in the current window
        """
        tracker = self._get_tracker(provider_name)
        current_time = time.time()

        # Clean up old rate limits first
        while (
            tracker.recent_rate_limits
            and current_time - tracker.recent_rate_limits[0] > self.window_seconds
        ):
            tracker.recent_rate_limits.popleft()

        return len(tracker.recent_rate_limits)
