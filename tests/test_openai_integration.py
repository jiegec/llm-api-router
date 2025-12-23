"""Integration tests for OpenAI client compatibility."""

import json
import time
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from httpx import Response

from llm_api_router.config import RouterConfig
from llm_api_router.server import LLMAPIServer


def create_mock_openai_response() -> dict:
    """Create a mock OpenAI API response."""
    return {
        "id": "chatcmpl-integration-test-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from integration test!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


def test_openai_client_compatibility():
    """Test that our server is compatible with official OpenAI client format."""
    # Create test config
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

    # Create FastAPI app
    server = LLMAPIServer(config)
    app = server.app

    # Mock the OpenAI provider's HTTP client
    with patch("llm_api_router.providers.base.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock successful response
        mock_response = Response(
            status_code=200,
            content=json.dumps(create_mock_openai_response()),
        )
        mock_client.post.return_value = mock_response

        with TestClient(app) as test_client:
            # Test OpenAI endpoint
            response = test_client.post(
                "/openai/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 200
            result = response.json()

            # Verify all required OpenAI fields are present
            required_fields = ["id", "object", "created", "model", "choices", "usage"]
            for field in required_fields:
                assert field in result, f"Missing required OpenAI field: {field}"

            # Verify choices structure
            assert isinstance(result["choices"], list)
            assert len(result["choices"]) > 0
            choice = result["choices"][0]
            assert "index" in choice
            assert "message" in choice
            assert "finish_reason" in choice

            # Verify message structure
            message = choice["message"]
            assert "role" in message
            assert "content" in message

            # Verify usage structure
            usage = result["usage"]
            assert "prompt_tokens" in usage
            assert "completion_tokens" in usage
            assert "total_tokens" in usage

            # Verify our custom field
            assert "provider" in result
            assert result["provider"] == "openai"


def test_openai_client_headers():
    """Test that our server handles OpenAI client headers correctly."""
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

    with patch("llm_api_router.providers.base.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        mock_response = Response(
            status_code=200,
            content=json.dumps(create_mock_openai_response()),
        )
        mock_client.post.return_value = mock_response

        with TestClient(app) as test_client:
            # Test with Authorization header (like OpenAI client sends)
            response = test_client.post(
                "/openai/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={
                    "Authorization": "Bearer any-key",
                    "Content-Type": "application/json",
                },
            )

            assert response.status_code == 200
            # Server should ignore the Authorization header and use configured API key


def test_multiple_providers_fallback_integration():
    """Test fallback with multiple providers in integration scenario."""
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-primary",
                    "priority": 1,
                },
                {
                    "api_key": "sk-backup",
                    "priority": 2,
                },
            ]
        }
    )

    server = LLMAPIServer(config)
    app = server.app

    with patch("llm_api_router.providers.base.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # First call fails with rate limit
        rate_limit_response = Response(
            status_code=429,
            content=json.dumps(
                {
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                    }
                }
            ),
        )

        # Second call succeeds
        success_response = Response(
            status_code=200,
            content=json.dumps(create_mock_openai_response()),
        )

        mock_client.post.side_effect = [rate_limit_response, success_response]

        with TestClient(app) as test_client:
            response = test_client.post(
                "/openai/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Test fallback"}],
                },
            )

            assert response.status_code == 200
            result = response.json()
            assert result["provider"] == "openai"

            # Verify both providers were tried
            assert mock_client.post.call_count == 2


def test_anthropic_endpoint_compatibility():
    """Test that Anthropic endpoint also returns OpenAI-compatible format."""
    config = RouterConfig.from_dict(
        {
            "anthropic": [
                {
                    "api_key": "sk-ant-test",
                    "priority": 1,
                }
            ]
        }
    )

    server = LLMAPIServer(config)
    app = server.app

    with patch("llm_api_router.providers.base.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock Anthropic response converted to OpenAI format
        mock_response = Response(
            status_code=200,
            content=json.dumps(create_mock_openai_response()),
        )
        mock_client.post.return_value = mock_response

        with TestClient(app) as test_client:
            response = test_client.post(
                "/anthropic/chat/completions",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 200
            result = response.json()

            # Should still have all OpenAI fields
            assert "id" in result
            assert "choices" in result
            assert "usage" in result
            assert "provider" in result
            assert result["provider"] == "anthropic"
