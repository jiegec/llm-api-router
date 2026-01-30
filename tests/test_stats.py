"""Tests for statistics collection functionality."""

import time
from unittest.mock import AsyncMock, Mock

import pytest

from llm_api_router.models import ProviderConfig, ProviderType
from llm_api_router.router import LLMRouter
from llm_api_router.stats import ProviderStats, RouterStats, StatsCollector


def test_stats_collector_initialization():
    """Test StatsCollector initialization."""
    collector = StatsCollector()
    stats = collector.get_stats()

    assert isinstance(stats, RouterStats)
    assert stats.start_time > 0
    assert stats.uptime_seconds >= 0
    assert stats.total_requests == 0
    assert stats.total_input_tokens == 0
    assert stats.total_output_tokens == 0
    assert stats.providers == {}


def test_stats_collector_record_request_start():
    """Test recording request start."""
    collector = StatsCollector()
    start_time = collector.record_request_start("test_provider")

    assert isinstance(start_time, float)

    stats = collector.get_stats()
    assert "test_provider" in stats.providers
    provider_stats = stats.providers["test_provider"]
    assert provider_stats.total_requests == 1
    assert provider_stats.in_progress_requests == 1
    assert provider_stats.last_request_time is not None


def test_stats_collector_record_request_success():
    """Test recording successful request."""
    collector = StatsCollector()
    start_time = collector.record_request_start("test_provider")

    # Verify in-progress counter was incremented
    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]
    assert provider_stats.total_requests == 1
    assert provider_stats.in_progress_requests == 1

    # Simulate some processing time
    time.sleep(0.01)

    collector.record_request_success(
        "test_provider", start_time, input_tokens=100, output_tokens=50
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]

    assert provider_stats.total_requests == 1
    assert provider_stats.in_progress_requests == 0
    assert provider_stats.successful_requests == 1
    assert provider_stats.failed_requests == 0
    assert provider_stats.total_input_tokens == 100
    assert provider_stats.total_output_tokens == 50
    assert provider_stats.total_tokens == 150
    assert provider_stats.average_latency_ms > 0
    assert provider_stats.tokens_per_second > 0
    assert provider_stats.last_success_time is not None
    assert provider_stats.last_error is None


def test_stats_collector_record_request_failure():
    """Test recording failed request."""
    collector = StatsCollector()
    collector.record_request_start("test_provider")
    collector.record_request_failure("test_provider", "Test error")

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]

    assert provider_stats.total_requests == 1
    assert provider_stats.in_progress_requests == 0
    assert provider_stats.successful_requests == 0
    assert provider_stats.failed_requests == 1
    assert provider_stats.last_error == "Test error"


def test_provider_stats_success_rate():
    """Test provider stats success rate calculation based on completed requests."""
    stats = ProviderStats()

    # No requests
    assert stats.success_rate == 0.0
    assert stats.failure_rate == 0.0

    # Add some completed requests (success + failed)
    stats.successful_requests = 7
    stats.failed_requests = 3

    assert stats.success_rate == 70.0
    assert stats.failure_rate == 30.0


def test_provider_stats_in_progress_requests():
    """Test that in-progress requests don't affect success rate calculation."""
    stats = ProviderStats()

    # Start some requests (they become in-progress)
    stats.total_requests = 10
    stats.in_progress_requests = 3

    # Only completed requests count towards success rate
    stats.successful_requests = 5
    stats.failed_requests = 2

    # Success rate should be 5 / (5 + 2) = ~71.43%, not 5 / 10 = 50%
    assert stats.success_rate == pytest.approx(71.42857142857143)
    assert stats.failure_rate == pytest.approx(28.57142857142857)

    # Verify the relationship: total = success + failed + in-progress
    assert stats.total_requests == 10
    assert (
        stats.successful_requests + stats.failed_requests + stats.in_progress_requests
        == 10
    )


def test_router_stats_most_used_provider():
    """Test router stats most used provider calculation."""
    stats = RouterStats(start_time=time.time())

    # No providers
    assert stats.most_used_provider is None

    # Add providers with different request counts
    stats.providers["provider1"] = ProviderStats(total_requests=5)
    stats.providers["provider2"] = ProviderStats(total_requests=10)
    stats.providers["provider3"] = ProviderStats(total_requests=3)

    assert stats.most_used_provider == "provider2"


def test_router_stats_fastest_provider():
    """Test router stats fastest provider calculation."""
    stats = RouterStats(start_time=time.time())

    # No providers
    assert stats.fastest_provider is None

    # Add providers with different latencies
    # average_latency_ms is now a computed property, so we need to set the underlying fields
    provider1 = ProviderStats(successful_requests=1, total_latency_ms=100)
    provider2 = ProviderStats(successful_requests=1, total_latency_ms=50)
    provider3 = ProviderStats(successful_requests=1, total_latency_ms=200)

    stats.providers["provider1"] = provider1
    stats.providers["provider2"] = provider2
    stats.providers["provider3"] = provider3

    assert stats.fastest_provider == "provider2"


def test_stats_collector_multiple_providers():
    """Test stats collection with multiple providers."""
    collector = StatsCollector()

    # Record requests for multiple providers
    start1 = collector.record_request_start("provider1")
    collector.record_request_start("provider2")

    collector.record_request_success("provider1", start1, 100, 50)
    collector.record_request_failure("provider2", "Error")

    stats = collector.get_stats()

    assert len(stats.providers) == 2
    assert stats.total_requests == 2
    assert stats.total_input_tokens == 100
    assert stats.total_output_tokens == 50
    assert stats.most_used_provider == "provider1"  # Both have 1 request, first wins


@pytest.mark.asyncio
async def test_router_statistics_integration():
    """Test router statistics integration."""
    # Create mock provider
    mock_provider = AsyncMock()
    mock_provider.chat_completion.return_value = {
        "id": "test-id",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    # Mock extract_tokens_from_response method (synchronous)
    mock_provider.extract_tokens_from_response = Mock(return_value=(10, 5, 0))

    # Create router with mock provider
    provider_config = ProviderConfig(
        name=ProviderType.OPENAI,
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        priority=1,
        model_mapping={"gpt-3.5-turbo": "gpt-3.5-turbo"},
    )

    router = LLMRouter([provider_config])

    # Mock the provider creation to use our mock
    router._provider_instances["openai"] = mock_provider

    # Mock _get_provider method to return our mock
    async def mock_get_provider(config):
        return mock_provider

    router._get_provider = mock_get_provider

    # Make a request
    request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    await router.chat_completion(request)

    # Check statistics
    stats = router.get_stats()
    assert "openai-priority-1" in stats.providers

    provider_stats = stats.providers["openai-priority-1"]
    assert provider_stats.total_requests == 1
    assert provider_stats.successful_requests == 1
    assert provider_stats.failed_requests == 0
    assert provider_stats.total_input_tokens == 10
    assert provider_stats.total_output_tokens == 5
    assert provider_stats.total_tokens == 15
    assert provider_stats.average_latency_ms > 0
    assert provider_stats.tokens_per_second > 0


def test_record_tokens_from_response_openai():
    """Test token extraction from OpenAI response."""
    from llm_api_router.models import ProviderConfig, ProviderType
    from llm_api_router.providers.openai import OpenAIProvider

    # Create a mock provider
    provider = OpenAIProvider(
        ProviderConfig(
            name=ProviderType.OPENAI,
            api_key="test-key",
            priority=1,
        )
    )

    response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    input_tokens, output_tokens, cached_tokens = provider.extract_tokens_from_response(
        response
    )

    assert input_tokens == 100
    assert output_tokens == 50
    assert cached_tokens == 0


def test_record_tokens_from_response_anthropic():
    """Test token extraction from Anthropic response."""
    from llm_api_router.models import ProviderConfig, ProviderType
    from llm_api_router.providers.anthropic import AnthropicProvider

    # Create a mock provider
    provider = AnthropicProvider(
        ProviderConfig(
            name=ProviderType.ANTHROPIC,
            api_key="test-key",
            priority=1,
        )
    )

    response = {
        "content": [{"type": "text", "text": "Test"}],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }

    input_tokens, output_tokens, cached_tokens = provider.extract_tokens_from_response(
        response
    )

    assert input_tokens == 100
    assert output_tokens == 50
    assert cached_tokens == 0


def test_record_tokens_from_response_invalid():
    """Test token extraction from invalid response."""
    from llm_api_router.models import ProviderConfig, ProviderType
    from llm_api_router.providers.openai import OpenAIProvider

    # Create a mock provider
    provider = OpenAIProvider(
        ProviderConfig(
            name=ProviderType.OPENAI,
            api_key="test-key",
            priority=1,
        )
    )

    response = {"invalid": "response"}

    input_tokens, output_tokens, cached_tokens = provider.extract_tokens_from_response(
        response
    )

    assert input_tokens == 0
    assert output_tokens == 0
    assert cached_tokens == 0


def test_tokens_per_second_average_across_requests():
    """Test that tokens_per_second is averaged across all successful requests using only output tokens."""
    collector = StatsCollector()

    # First request: 100 output tokens in 0.1 seconds
    start_time1 = time.time()
    time.sleep(0.1)  # Simulate 0.1 second processing
    collector.record_request_success(
        "test_provider", start_time1, input_tokens=50, output_tokens=100
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]
    # Should be approximately 100 / 0.1 = 1000 tokens/second
    assert provider_stats.tokens_per_second == pytest.approx(1000, rel=0.2)

    # Second request: 50 output tokens in 0.05 seconds
    start_time2 = time.time()
    time.sleep(0.05)  # Simulate 0.05 second processing
    collector.record_request_success(
        "test_provider", start_time2, input_tokens=50, output_tokens=50
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]

    # After two requests, should be average of both:
    # First: ~1000 tokens/s (100 tokens / 0.1s)
    # Second: ~1000 tokens/s (50 tokens / 0.05s)
    # Average: (1000 + 1000) / 2 = 1000 tokens/s
    # Note: We use relative tolerance because sleep is not precise
    assert provider_stats.tokens_per_second == pytest.approx(1000, rel=0.3)

    # Verify that only output tokens are used
    # The input tokens should not affect the calculation
    assert provider_stats.total_input_tokens == 100  # 50 + 50
    assert provider_stats.total_output_tokens == 150  # 100 + 50


def test_tokens_per_second_uses_only_output_tokens():
    """Test that tokens_per_second calculation only uses output tokens, not input tokens."""
    collector = StatsCollector()

    # Request with many input tokens but few output tokens
    start_time = time.time()
    time.sleep(0.01)
    collector.record_request_success(
        "test_provider", start_time, input_tokens=1000, output_tokens=10
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]

    # Should calculate based on 10 output tokens, not 1010 total tokens
    # 10 tokens / ~0.01s = ~1000 tokens/s
    assert provider_stats.tokens_per_second == pytest.approx(1000, rel=0.3)
    assert provider_stats.total_input_tokens == 1000
    assert provider_stats.total_output_tokens == 10


def test_non_streaming_metrics():
    """Test non-streaming specific metrics."""
    collector = StatsCollector()

    # Record non-streaming request
    start_time = time.time()
    time.sleep(0.01)
    collector.record_request_success(
        "test_provider",
        start_time,
        input_tokens=50,
        output_tokens=100,
        is_streaming=False,
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]

    assert provider_stats.non_streaming_requests == 1
    assert provider_stats.streaming_requests == 0
    assert provider_stats.non_streaming_average_latency_ms > 0
    assert provider_stats.non_streaming_tokens_per_second > 0

    # Legacy metrics should also be set
    assert provider_stats.average_latency_ms > 0
    assert provider_stats.tokens_per_second > 0


def test_streaming_metrics():
    """Test streaming specific metrics including time to first token."""
    collector = StatsCollector()

    # Simulate streaming request
    start_time = time.time()
    time.sleep(0.01)  # Simulate time to first token
    time.sleep(0.05)  # Simulate rest of streaming

    # Record with time_to_first_token (e.g., 10ms)
    collector.record_request_success(
        "test_provider",
        start_time,
        input_tokens=50,
        output_tokens=100,
        is_streaming=True,
        time_to_first_token_ms=10.0,
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]

    assert provider_stats.streaming_requests == 1
    assert provider_stats.non_streaming_requests == 0
    assert provider_stats.streaming_average_time_to_first_token_ms == 10.0
    assert provider_stats.streaming_average_latency_ms > 0

    # Both tokens/s metrics should be calculated
    assert provider_stats.streaming_tokens_per_second_with_first_token > 0
    assert provider_stats.streaming_tokens_per_second_without_first_token > 0

    # Without first token should be faster (same tokens, less time)
    assert (
        provider_stats.streaming_tokens_per_second_without_first_token
        > provider_stats.streaming_tokens_per_second_with_first_token
    )


def test_streaming_metrics_without_ttft():
    """Test streaming metrics when time_to_first_token is not provided."""
    collector = StatsCollector()

    start_time = time.time()
    time.sleep(0.01)
    collector.record_request_success(
        "test_provider",
        start_time,
        input_tokens=50,
        output_tokens=100,
        is_streaming=True,
        time_to_first_token_ms=None,
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]

    assert provider_stats.streaming_requests == 1
    assert provider_stats.streaming_average_time_to_first_token_ms == 0.0
    # tokens/s without first token should remain 0 since we don't have TTFT
    assert provider_stats.streaming_tokens_per_second_without_first_token == 0.0
    # tokens/s with first token should still be calculated
    assert provider_stats.streaming_tokens_per_second_with_first_token > 0


def test_mixed_streaming_and_non_streaming():
    """Test that streaming and non-streaming metrics are tracked separately."""
    collector = StatsCollector()

    # Record non-streaming request
    start_time1 = collector.record_request_start("test_provider")
    time.sleep(0.01)
    collector.record_request_success(
        "test_provider",
        start_time1,
        input_tokens=50,
        output_tokens=100,
        is_streaming=False,
    )

    # Record streaming request
    start_time2 = collector.record_request_start("test_provider")
    time.sleep(0.01)
    collector.record_request_success(
        "test_provider",
        start_time2,
        input_tokens=50,
        output_tokens=100,
        is_streaming=True,
        time_to_first_token_ms=5.0,
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]

    assert provider_stats.total_requests == 2
    assert provider_stats.non_streaming_requests == 1
    assert provider_stats.streaming_requests == 1
    assert provider_stats.successful_requests == 2
    assert provider_stats.in_progress_requests == 0

    # Each should have its own metrics
    assert provider_stats.non_streaming_average_latency_ms > 0
    assert provider_stats.streaming_average_latency_ms > 0
    assert provider_stats.streaming_average_time_to_first_token_ms == 5.0


def test_record_rate_limit():
    """Test recording rate limit errors."""
    collector = StatsCollector()

    # Record multiple rate limits
    collector.record_rate_limit("provider1")
    collector.record_rate_limit("provider1")
    collector.record_rate_limit("provider2")

    stats = collector.get_stats()

    assert stats.providers["provider1"].total_rate_limits == 2
    assert stats.providers["provider2"].total_rate_limits == 1


def test_record_cooldown_start():
    """Test recording cooldown start."""
    collector = StatsCollector()

    collector.record_cooldown_start("provider1")

    stats = collector.get_stats()
    provider_stats = stats.providers["provider1"]

    assert provider_stats.cooldown_count == 1
    assert provider_stats.in_cooldown is True
    assert provider_stats.last_cooldown_start_time is not None
    assert isinstance(provider_stats.last_cooldown_start_time, float)


def test_record_cooldown_end():
    """Test recording cooldown end."""
    collector = StatsCollector()

    # Start cooldown
    collector.record_cooldown_start("provider1")
    time.sleep(0.1)  # Simulate some cooldown time

    # End cooldown
    collector.record_cooldown_end("provider1")

    stats = collector.get_stats()
    provider_stats = stats.providers["provider1"]

    assert provider_stats.in_cooldown is False
    assert provider_stats.last_cooldown_start_time is None
    assert provider_stats.total_cooldown_time_seconds >= 0.1


def test_update_rate_limit_status():
    """Test updating rate limit status."""
    collector = StatsCollector()

    # Update with some rate limits and cooldown
    collector.update_rate_limit_status(
        "provider1", recent_count=3, cooldown_remaining=60.0
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["provider1"]

    assert provider_stats.recent_rate_limits == 3
    assert provider_stats.in_cooldown is True
    assert provider_stats.cooldown_remaining_seconds == 60.0

    # Update again with no cooldown
    collector.update_rate_limit_status(
        "provider1", recent_count=0, cooldown_remaining=None
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["provider1"]

    assert provider_stats.recent_rate_limits == 0
    assert provider_stats.in_cooldown is False
    assert provider_stats.cooldown_remaining_seconds == 0.0


def test_provider_stats_rate_limit_fields():
    """Test ProviderStats rate limit field defaults."""
    stats = ProviderStats()

    assert stats.total_rate_limits == 0
    assert stats.recent_rate_limits == 0
    assert stats.cooldown_count == 0
    assert stats.total_cooldown_time_seconds == 0.0
    assert stats.in_cooldown is False
    assert stats.cooldown_remaining_seconds == 0.0
    assert stats.last_cooldown_start_time is None


def test_multiple_cooldowns():
    """Test tracking multiple cooldown cycles."""
    collector = StatsCollector()

    # First cooldown cycle
    collector.record_cooldown_start("provider1")
    time.sleep(0.05)
    collector.record_cooldown_end("provider1")

    # Second cooldown cycle
    collector.record_cooldown_start("provider1")
    time.sleep(0.05)
    collector.record_cooldown_end("provider1")

    stats = collector.get_stats()
    provider_stats = stats.providers["provider1"]

    assert provider_stats.cooldown_count == 2
    assert provider_stats.total_cooldown_time_seconds >= 0.1
    assert provider_stats.in_cooldown is False


def test_rate_limit_status_with_no_cooldown():
    """Test update_rate_limit_status with None for cooldown_remaining."""
    collector = StatsCollector()

    # None means no cooldown
    collector.update_rate_limit_status(
        "provider1", recent_count=1, cooldown_remaining=None
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["provider1"]

    assert provider_stats.recent_rate_limits == 1
    assert provider_stats.in_cooldown is False
    assert provider_stats.cooldown_remaining_seconds == 0.0
