"""Tests for the rate limiter module."""

import time

from llm_api_router.rate_limiter import RateLimiter, RateLimitTracker


class TestRateLimitTracker:
    """Tests for RateLimitTracker class."""

    def test_initial_state(self):
        """Test that a new tracker has empty state."""
        tracker = RateLimitTracker()
        assert len(tracker.recent_rate_limits) == 0
        assert tracker.cooldown_until is None


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_initialization(self):
        """Test that rate limiter initializes with correct defaults."""
        rl = RateLimiter()
        assert rl.consecutive_threshold == 3
        assert rl.window_seconds == 60
        assert rl.cooldown_seconds == 600

    def test_custom_thresholds(self):
        """Test that custom thresholds work."""
        rl = RateLimiter(
            consecutive_threshold=2, window_seconds=30, cooldown_seconds=300
        )
        assert rl.consecutive_threshold == 2
        assert rl.window_seconds == 30
        assert rl.cooldown_seconds == 300

    def test_is_in_cooldown_no_cooldown(self):
        """Test that providers start not in cooldown."""
        rl = RateLimiter()
        assert rl.is_in_cooldown("provider1") is False

    def test_record_single_rate_limit(self):
        """Test recording a single rate limit."""
        rl = RateLimiter()
        rl.record_rate_limit("provider1")
        assert rl.is_in_cooldown("provider1") is False
        assert rl.get_recent_rate_limit_count("provider1") == 1

    def test_record_multiple_rate_limits_triggers_cooldown(self):
        """Test that 3 rate limits trigger cooldown."""
        rl = RateLimiter(consecutive_threshold=3)
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        assert rl.is_in_cooldown("provider1") is True
        assert rl.get_recent_rate_limit_count("provider1") == 0  # Cleared on cooldown

    def test_cooldown_expires(self):
        """Test that cooldown expires after the specified time."""
        rl = RateLimiter(consecutive_threshold=3, cooldown_seconds=0.1)
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        assert rl.is_in_cooldown("provider1") is True
        time.sleep(0.15)  # Wait for cooldown to expire
        assert rl.is_in_cooldown("provider1") is False

    def test_success_resets_rate_limits(self):
        """Test that a successful request resets the rate limit counter."""
        rl = RateLimiter(consecutive_threshold=3)
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        assert rl.get_recent_rate_limit_count("provider1") == 2
        rl.record_success("provider1")
        assert rl.get_recent_rate_limit_count("provider1") == 0

    def test_success_clears_cooldown(self):
        """Test that success clears cooldown state but not the cooldown_until timestamp."""
        rl = RateLimiter(consecutive_threshold=2, cooldown_seconds=60)
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        assert rl.is_in_cooldown("provider1") is True

        # Success clears rate limits but cooldown is still active
        rl.record_success("provider1")
        assert rl.get_recent_rate_limit_count("provider1") == 0
        # Still in cooldown because cooldown_until hasn't expired
        assert rl.is_in_cooldown("provider1") is True

    def test_separate_providers(self):
        """Test that rate limits are tracked separately for each provider."""
        rl = RateLimiter(consecutive_threshold=3)
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider2")
        assert rl.get_recent_rate_limit_count("provider1") == 2
        assert rl.get_recent_rate_limit_count("provider2") == 1
        assert rl.is_in_cooldown("provider1") is False
        assert rl.is_in_cooldown("provider2") is False

    def test_window_expiration(self):
        """Test that rate limits outside the window are removed."""
        rl = RateLimiter(window_seconds=0.1)
        rl.record_rate_limit("provider1")
        time.sleep(0.15)  # Wait for window to expire
        # Recording a new rate limit should clean up the old one
        rl.record_rate_limit("provider1")
        assert rl.get_recent_rate_limit_count("provider1") == 1

    def test_get_cooldown_remaining_seconds(self):
        """Test getting remaining cooldown time."""
        rl = RateLimiter(consecutive_threshold=3, cooldown_seconds=60)
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")

        remaining = rl.get_cooldown_remaining_seconds("provider1")
        assert remaining is not None
        assert 0 < remaining <= 60

        # Not in cooldown provider should return None
        assert rl.get_cooldown_remaining_seconds("provider2") is None

    def test_get_cooldown_end_time(self):
        """Test getting cooldown end timestamp."""
        rl = RateLimiter(consecutive_threshold=3, cooldown_seconds=60)
        before = time.time()
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        rl.record_rate_limit("provider1")
        after = time.time()

        end_time = rl.get_cooldown_end_time("provider1")
        assert end_time is not None
        assert before + 60 <= end_time <= after + 60

    def test_cooldown_remaining_when_not_in_cooldown(self):
        """Test that cooldown remaining returns None when not in cooldown."""
        rl = RateLimiter()
        assert rl.get_cooldown_remaining_seconds("provider1") is None

    def test_multiple_providers_different_states(self):
        """Test multiple providers with different rate limit states."""
        rl = RateLimiter(consecutive_threshold=3)

        # Provider 1: No rate limits
        # Provider 2: Some rate limits
        rl.record_rate_limit("provider2")
        rl.record_rate_limit("provider2")

        # Provider 3: In cooldown
        rl.record_rate_limit("provider3")
        rl.record_rate_limit("provider3")
        rl.record_rate_limit("provider3")

        assert rl.is_in_cooldown("provider1") is False
        assert rl.is_in_cooldown("provider2") is False
        assert rl.is_in_cooldown("provider3") is True

        assert rl.get_recent_rate_limit_count("provider1") == 0
        assert rl.get_recent_rate_limit_count("provider2") == 2
        assert rl.get_recent_rate_limit_count("provider3") == 0  # Cleared on cooldown


class TestRateLimiterWithRouter:
    """Tests for rate limiter integration with router."""

    def test_router_has_rate_limiter(self):
        """Test that router initializes with rate limiter."""
        from llm_api_router import LLMRouter, ProviderConfig, ProviderType

        config = ProviderConfig(
            name=ProviderType.OPENAI, api_key="test-key", priority=1
        )
        router = LLMRouter([config], endpoint="openai")
        assert router.rate_limiter is not None
        assert isinstance(router.rate_limiter, RateLimiter)

    def test_router_get_rate_limiter_status(self):
        """Test that router can get rate limiter status."""
        from llm_api_router import LLMRouter, ProviderConfig, ProviderType

        config1 = ProviderConfig(
            name=ProviderType.OPENAI, api_key="test-key", priority=1
        )
        config2 = ProviderConfig(
            name=ProviderType.ANTHROPIC,
            api_key="test-key",
            priority=2,
            provider_name="anthropic-backup",
        )
        router = LLMRouter([config1, config2], endpoint="openai")
        status = router.get_rate_limiter_status()

        assert "openai-priority-1" in status
        assert "anthropic-backup" in status
        assert status["openai-priority-1"]["in_cooldown"] is False
        assert status["openai-priority-1"]["cooldown_remaining"] is None
        assert status["openai-priority-1"]["recent_rate_limits"] == 0
