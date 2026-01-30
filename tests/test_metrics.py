"""Tests for the Prometheus metrics endpoint."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from llm_api_router.config import RouterConfig
from llm_api_router.server import LLMAPIServer
from llm_api_router.stats import RouterStats, StatsCollector


def test_metrics_endpoint_basic():
    """Test that the metrics endpoint returns Prometheus format."""
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-test-key",
                    "priority": 1,
                }
            ]
        }
    )

    server = LLMAPIServer(config)
    app = server.app

    # Mock the router and its stats
    mock_stats = RouterStats(start_time=1000)
    mock_stats.uptime_seconds = 123.45
    mock_stats.total_requests = 10
    mock_stats.total_input_tokens = 100
    mock_stats.total_output_tokens = 50
    mock_stats.total_cached_tokens = 25

    mock_router = Mock()
    mock_router.get_stats.return_value = mock_stats
    server.openai_router = mock_router

    with TestClient(app) as test_client:
        response = test_client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        content = response.text

        # Check for basic Prometheus format
        assert "# HELP" in content
        assert "# TYPE" in content

        # Check for expected metrics
        assert "llm_router_info" in content
        assert "llm_router_uptime_seconds" in content
        assert "llm_router_requests_total" in content
        assert 'provider_type="openai"' in content


def test_metrics_endpoint_with_anthropic():
    """Test metrics endpoint includes both provider types."""
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-test-key",
                    "priority": 1,
                }
            ],
            "anthropic": [
                {
                    "api_key": "sk-ant-test",
                    "priority": 1,
                }
            ],
        }
    )

    server = LLMAPIServer(config)
    app = server.app

    # Mock both routers
    mock_openai_stats = RouterStats(start_time=1000)
    mock_openai_stats.uptime_seconds = 100.0
    mock_openai_stats.total_requests = 50

    mock_anthropic_stats = RouterStats(start_time=1000)
    mock_anthropic_stats.uptime_seconds = 200.0
    mock_anthropic_stats.total_requests = 30

    mock_openai_router = Mock()
    mock_openai_router.get_stats.return_value = mock_openai_stats
    server.openai_router = mock_openai_router

    mock_anthropic_router = Mock()
    mock_anthropic_router.get_stats.return_value = mock_anthropic_stats
    server.anthropic_router = mock_anthropic_router

    with TestClient(app) as test_client:
        response = test_client.get("/metrics")

        assert response.status_code == 200
        content = response.text

        # Both provider types should be present
        assert 'provider_type="openai"' in content
        assert 'provider_type="anthropic"' in content

        # Check for token metrics
        assert "llm_router_input_tokens_total" in content
        assert "llm_router_output_tokens_total" in content
        assert "llm_router_cached_tokens_total" in content


def test_metrics_endpoint_with_provider_stats():
    """Test metrics endpoint includes per-provider stats."""
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-test-key",
                    "priority": 1,
                    "provider_name": "openai-primary",
                }
            ]
        }
    )

    server = LLMAPIServer(config)
    app = server.app

    # Create stats with provider details
    mock_stats = RouterStats(start_time=1000)
    mock_stats.uptime_seconds = 60.0
    mock_stats.total_requests = 100
    mock_stats.total_input_tokens = 1000
    mock_stats.total_output_tokens = 500
    mock_stats.total_cached_tokens = 200

    # Add provider-specific stats
    from llm_api_router.stats import ProviderStats

    provider_stats = ProviderStats()
    provider_stats.total_requests = 100
    provider_stats.successful_requests = 95
    provider_stats.failed_requests = 5
    provider_stats.total_input_tokens = 1000
    provider_stats.total_output_tokens = 500
    provider_stats.total_cached_tokens = 200
    provider_stats.streaming_requests = 60
    provider_stats.non_streaming_requests = 40

    mock_stats.providers = {"openai-primary": provider_stats}

    mock_router = Mock()
    mock_router.get_stats.return_value = mock_stats
    server.openai_router = mock_router

    with TestClient(app) as test_client:
        response = test_client.get("/metrics")

        assert response.status_code == 200
        content = response.text

        # Check for provider-specific metrics
        assert 'provider="openai-primary"' in content
        assert "llm_router_provider_requests_total" in content
        assert "llm_router_provider_requests_successful" in content
        assert "llm_router_provider_requests_failed" in content
        assert "llm_router_provider_input_tokens_total" in content
        assert "llm_router_provider_streaming_requests_total" in content
        assert "llm_router_provider_non_streaming_requests_total" in content
        assert "llm_router_provider_success_rate" in content


def test_metrics_endpoint_no_providers():
    """Test metrics endpoint when no providers are configured."""
    config = RouterConfig.from_dict({})

    server = LLMAPIServer(config)
    app = server.app

    with TestClient(app) as test_client:
        response = test_client.get("/metrics")

        assert response.status_code == 200
        content = response.text

        # Should return a message indicating no metrics
        assert "# No metrics available" in content


def test_metrics_endpoint_in_root_response():
    """Test that metrics endpoint is listed in root endpoint."""
    config = RouterConfig.from_dict({})

    server = LLMAPIServer(config)
    app = server.app

    with TestClient(app) as test_client:
        response = test_client.get("/")

        assert response.status_code == 200
        result = response.json()

        assert "endpoints" in result
        assert "metrics" in result["endpoints"]
        assert result["endpoints"]["metrics"] == "/metrics"
