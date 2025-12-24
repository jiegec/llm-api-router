"""Tests for statistics collection functionality."""

import time
from unittest.mock import AsyncMock

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
    assert stats.providers["test_provider"].total_requests == 1
    assert stats.providers["test_provider"].last_request_time is not None


def test_stats_collector_record_request_success():
    """Test recording successful request."""
    collector = StatsCollector()
    start_time = collector.record_request_start("test_provider")

    # Simulate some processing time
    time.sleep(0.01)

    collector.record_request_success(
        "test_provider", start_time, input_tokens=100, output_tokens=50
    )

    stats = collector.get_stats()
    provider_stats = stats.providers["test_provider"]

    assert provider_stats.total_requests == 1
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
    assert provider_stats.successful_requests == 0
    assert provider_stats.failed_requests == 1
    assert provider_stats.last_error == "Test error"


def test_provider_stats_success_rate():
    """Test provider stats success rate calculation."""
    stats = ProviderStats()

    # No requests
    assert stats.success_rate == 0.0
    assert stats.failure_rate == 0.0

    # Add some requests
    stats.total_requests = 10
    stats.successful_requests = 7
    stats.failed_requests = 3

    assert stats.success_rate == 70.0
    assert stats.failure_rate == 30.0


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
    stats.providers["provider1"] = ProviderStats(average_latency_ms=100)
    stats.providers["provider2"] = ProviderStats(average_latency_ms=50)
    stats.providers["provider3"] = ProviderStats(average_latency_ms=200)

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
    collector = StatsCollector()

    response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    input_tokens, output_tokens = collector.record_tokens_from_response(
        "test_provider", response
    )

    assert input_tokens == 100
    assert output_tokens == 50


def test_record_tokens_from_response_anthropic():
    """Test token extraction from Anthropic response."""
    collector = StatsCollector()

    response = {
        "content": [{"type": "text", "text": "Test"}],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }

    input_tokens, output_tokens = collector.record_tokens_from_response(
        "test_provider", response
    )

    assert input_tokens == 100
    assert output_tokens == 50


def test_record_tokens_from_response_invalid():
    """Test token extraction from invalid response."""
    collector = StatsCollector()

    response = {"invalid": "response"}

    input_tokens, output_tokens = collector.record_tokens_from_response(
        "test_provider", response
    )

    assert input_tokens == 0
    assert output_tokens == 0
