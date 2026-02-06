"""Integration test with real Anthropic Python client using TestClient.

This test uses the real Anthropic Python client to make requests to our server
via TestClient, which simulates HTTP requests without starting a real server.
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
from anthropic import Anthropic
from fastapi.testclient import TestClient

from llm_api_router.config import RouterConfig
from llm_api_router.logging import get_logger
from llm_api_router.server import LLMAPIServer


def create_mock_anthropic_response() -> dict:
    """Create a mock Anthropic API response in Anthropic format."""
    return {
        "id": "msg-anthropic-test-123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello from real Anthropic client!"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "stop",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 15,
        },
    }


def test_real_anthropic_client_with_testclient():
    """Test that our server works with real Anthropic Python client via TestClient."""
    # Create test config
    config = RouterConfig.from_dict(
        {
            "anthropic": [
                {
                    "api_key": "sk-ant-test-key-real-client",
                    "priority": 1,
                }
            ]
        }
    )

    # force the global logger to use temporary log folder
    get_logger("/tmp/logs", force_new=True)

    # Create FastAPI app
    server = LLMAPIServer(config)
    app = server.app

    # Mock Anthropic provider's HTTP client
    with patch("llm_api_router.providers.base.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock successful response
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(create_mock_anthropic_response()),
        )
        mock_client.post.return_value = mock_response

        # Create Anthropic client pointing to our TestClient
        # We need to use a custom transport to route through TestClient
        class TestClientTransport(httpx.BaseTransport):
            def __init__(self, test_client: TestClient):
                self.test_client = test_client

            def handle_request(self, request: httpx.Request) -> httpx.Response:
                # Convert httpx request to TestClient request
                method = request.method
                url = str(request.url)
                headers = dict(request.headers)
                content = request.content

                # Extract path from URL and make request through TestClient
                # TestClient expects relative paths, not full URLs
                from urllib.parse import urlparse

                parsed = urlparse(url)
                path = parsed.path
                if parsed.query:
                    path += "?" + parsed.query

                response = self.test_client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    content=content,
                )

                # Convert TestClient response to httpx response
                return httpx.Response(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=response.content,
                )

        with TestClient(app) as test_client:
            # Create Anthropic client with custom transport
            # Note: base_url should not include /v1 since the SDK adds it
            client = Anthropic(
                api_key="any-key-works-since-we-mock",
                base_url="http://testserver/anthropic",
                http_client=httpx.Client(transport=TestClientTransport(test_client)),
            )

            # Make a request using the real Anthropic client
            # Note: Anthropic API has a different structure than OpenAI
            # We need to use the OpenAI-compatible endpoint format
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "Hello from real Anthropic client!"}
                ],
            )

            # Verify response structure
            # Anthropic SDK returns a Message object
            assert message.id == "msg-anthropic-test-123"
            assert message.model == "claude-3-5-sonnet-20241022"
            assert message.type == "message"
            assert message.role == "assistant"

            # Anthropic uses content blocks
            assert message.content is not None
            assert len(message.content) > 0
            # The first content block should be text
            assert message.content[0].type == "text"
            assert message.content[0].text == "Hello from real Anthropic client!"

            # Check usage
            assert message.usage.input_tokens == 20
            assert message.usage.output_tokens == 15

            assert (
                message.model_dump(exclude_none=True)
                == create_mock_anthropic_response()
            )

            stats = server.anthropic_router.get_stats()
            assert stats.total_requests == 1
            assert stats.total_input_tokens == 20
            assert stats.total_output_tokens == 15
            assert stats.total_cached_tokens == 0
            assert stats.providers["anthropic-priority-1"].total_requests == 1
            assert stats.providers["anthropic-priority-1"].in_progress_requests == 0
            assert stats.providers["anthropic-priority-1"].successful_requests == 1
            assert stats.providers["anthropic-priority-1"].total_input_tokens == 20
            assert stats.providers["anthropic-priority-1"].total_output_tokens == 15
            assert stats.providers["anthropic-priority-1"].total_tokens == 35
            assert stats.providers["anthropic-priority-1"].total_cached_tokens == 0


def test_real_anthropic_client_fallback():
    """Test fallback with real Anthropic client."""
    # Create config with multiple providers
    config = RouterConfig.from_dict(
        {
            "anthropic": [
                {
                    "api_key": "sk-ant-primary",
                    "priority": 1,
                },
                {
                    "api_key": "sk-ant-backup",
                    "priority": 2,
                },
            ]
        }
    )

    # force the global logger to use temporary log folder
    get_logger("/tmp/logs", force_new=True)

    server = LLMAPIServer(config)
    app = server.app

    # Mock HTTP client with side effects
    with patch("llm_api_router.providers.base.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # First call fails with rate limit
        rate_limit_response = httpx.Response(
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
        success_response = httpx.Response(
            status_code=200,
            content=json.dumps(create_mock_anthropic_response()),
        )

        mock_client.post.side_effect = [rate_limit_response, success_response]

        # Create Anthropic client with custom transport
        class TestClientTransport(httpx.BaseTransport):
            def __init__(self, test_client: TestClient):
                self.test_client = test_client

            def handle_request(self, request: httpx.Request) -> httpx.Response:
                method = request.method
                url = str(request.url)
                headers = dict(request.headers)
                content = request.content

                # Extract path from URL and make request through TestClient
                from urllib.parse import urlparse

                parsed = urlparse(url)
                path = parsed.path
                if parsed.query:
                    path += "?" + parsed.query

                response = self.test_client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    content=content,
                )

                return httpx.Response(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=response.content,
                )

        with TestClient(app) as test_client:
            client = Anthropic(
                api_key="any-key",
                base_url="http://testserver/anthropic",
                http_client=httpx.Client(transport=TestClientTransport(test_client)),
            )

            # Make a request - should trigger fallback
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Test fallback"}],
            )

            # Should still get successful response
            assert message.id == "msg-anthropic-test-123"
            assert message.content[0].text == "Hello from real Anthropic client!"


def test_real_anthropic_client_with_system_message():
    """Test Anthropic client with system message."""
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

    # force the global logger to use temporary log folder
    get_logger("/tmp/logs", force_new=True)

    server = LLMAPIServer(config)
    app = server.app

    with patch("llm_api_router.providers.base.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(create_mock_anthropic_response()),
        )
        mock_client.post.return_value = mock_response

        class TestClientTransport(httpx.BaseTransport):
            def __init__(self, test_client: TestClient):
                self.test_client = test_client

            def handle_request(self, request: httpx.Request) -> httpx.Response:
                method = request.method
                url = str(request.url)
                headers = dict(request.headers)
                content = request.content

                # Extract path from URL and make request through TestClient
                from urllib.parse import urlparse

                parsed = urlparse(url)
                path = parsed.path
                if parsed.query:
                    path += "?" + parsed.query

                response = self.test_client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    content=content,
                )

                return httpx.Response(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    content=response.content,
                )

        with TestClient(app) as test_client:
            client = Anthropic(
                api_key="any-key",
                base_url="http://testserver/anthropic",
                http_client=httpx.Client(transport=TestClientTransport(test_client)),
            )

            # Make a request with system message
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            # Verify response
            assert message.id == "msg-anthropic-test-123"
            assert message.content[0].text == "Hello from real Anthropic client!"


if __name__ == "__main__":
    test_real_anthropic_client_with_testclient()
    print("Test 1 passed!")
    test_real_anthropic_client_fallback()
    print("Test 2 passed!")
    test_real_anthropic_client_with_system_message()
    print("Test 3 passed!")
    print("All tests passed!")
