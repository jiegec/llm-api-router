"""Tests for user agent logging functionality."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from llm_api_router.logging import RouterLogger
from llm_api_router.server import LLMAPIServer


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary log directory."""
    return str(tmp_path / "logs")


@pytest.fixture
def logger_with_user_agent(temp_log_dir):
    """Create a RouterLogger for testing user agent logging."""
    return RouterLogger(log_dir=temp_log_dir, log_level="DEBUG")


class TestUserAgentLogging:
    """Test user agent logging in JSON logs."""

    def test_log_request_with_user_agent(self, logger_with_user_agent, temp_log_dir):
        """Test that user agent is logged in JSON log entry."""
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        logger_with_user_agent.log_request(
            request_id="req-123",
            endpoint="openai/chat/completions",
            request=request,
            provider_name="test-provider",
            provider_priority=1,
            user_agent="MyApp/1.0.0 (Python)",
        )

        # Read the main JSON log file (not the error log)
        log_files = [
            f for f in Path(temp_log_dir).glob("*.jsonl") if "errors" not in f.name
        ]
        assert len(log_files) == 1

        with open(log_files[0], encoding="utf-8") as f:
            log_entry = json.loads(f.readline())

        assert log_entry["type"] == "request"
        assert log_entry["user_agent"] == "MyApp/1.0.0 (Python)"

    def test_log_request_without_user_agent(self, logger_with_user_agent, temp_log_dir):
        """Test that log_request handles missing user agent (None)."""
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        logger_with_user_agent.log_request(
            request_id="req-456",
            endpoint="openai/chat/completions",
            request=request,
            provider_name="test-provider",
            provider_priority=1,
            user_agent=None,
        )

        # Read the main JSON log file (not the error log)
        log_files = [
            f for f in Path(temp_log_dir).glob("*.jsonl") if "errors" not in f.name
        ]
        assert len(log_files) == 1

        with open(log_files[0], encoding="utf-8") as f:
            log_entry = json.loads(f.readline())

        assert log_entry["type"] == "request"
        assert log_entry["user_agent"] is None

    def test_log_request_with_various_user_agents(
        self, logger_with_user_agent, temp_log_dir
    ):
        """Test logging various user agent formats."""
        user_agents = [
            "curl/7.68.0",
            "Python/3.9 httpx/0.24.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "OpenAI/Python 1.0.0",
            "Anthropic/TypeScript 0.1.0",
        ]

        for i, ua in enumerate(user_agents):
            logger_with_user_agent.log_request(
                request_id=f"req-{i}",
                endpoint="test/endpoint",
                request={"model": "test"},
                provider_name="test-provider",
                provider_priority=1,
                user_agent=ua,
            )

        # Read the main JSON log file (not the error log)
        log_files = [
            f for f in Path(temp_log_dir).glob("*.jsonl") if "errors" not in f.name
        ]
        assert len(log_files) == 1

        with open(log_files[0], encoding="utf-8") as f:
            for i, line in enumerate(f):
                log_entry = json.loads(line)
                assert log_entry["user_agent"] == user_agents[i]


class TestRouterUserAgentPassThrough:
    """Test router passes user agent through to logger."""

    @pytest.mark.asyncio
    async def test_router_chat_completion_passes_user_agent(
        self, openai_config, sample_request, temp_log_dir, monkeypatch
    ):
        """Test that router.chat_completion passes user_agent to logger."""
        # Mock the provider
        with patch("llm_api_router.router.create_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.extract_tokens_from_response = Mock(return_value=(10, 8, 0))
            mock_provider.chat_completion.return_value = {
                "id": "test-123",
                "model": "gpt-4",
                "choices": [{"finish_reason": "stop"}],
            }
            mock_create.return_value = mock_provider

            # Reset the global logger before creating router to use temp_log_dir

            monkeypatch.setattr("llm_api_router.logging._logger", None)

            from llm_api_router import LLMRouter

            router = LLMRouter(providers=[openai_config], log_dir=temp_log_dir)

            # Call with user_agent
            await router.chat_completion(sample_request, user_agent="TestApp/1.0")

            # Get the logger and verify log_request was called with user_agent
            log_files = [
                f for f in Path(temp_log_dir).glob("*.jsonl") if "errors" not in f.name
            ]
            assert len(log_files) == 1

            with open(log_files[0], encoding="utf-8") as f:
                log_lines = f.readlines()

            # Find the request log entry
            request_log = None
            for line in log_lines:
                entry = json.loads(line)
                if entry.get("type") == "request":
                    request_log = entry
                    break

            assert request_log is not None
            assert request_log["user_agent"] == "TestApp/1.0"

    @pytest.mark.asyncio
    async def test_router_chat_completion_without_user_agent(
        self, openai_config, sample_request, temp_log_dir, monkeypatch
    ):
        """Test that router.chat_completion handles missing user_agent."""
        with patch("llm_api_router.router.create_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.extract_tokens_from_response = Mock(return_value=(10, 8, 0))
            mock_provider.chat_completion.return_value = {
                "id": "test-456",
                "model": "gpt-4",
                "choices": [{"finish_reason": "stop"}],
            }
            mock_create.return_value = mock_provider

            # Reset the global logger before creating router to use temp_log_dir

            monkeypatch.setattr("llm_api_router.logging._logger", None)

            from llm_api_router import LLMRouter

            router = LLMRouter(providers=[openai_config], log_dir=temp_log_dir)

            # Call without user_agent
            await router.chat_completion(sample_request)

            # Get the logger and verify log_request was called with user_agent=None
            # The logger creates files inside temp_log_dir
            log_files = [
                f for f in Path(temp_log_dir).glob("*.jsonl") if "errors" not in f.name
            ]
            assert len(log_files) == 1

            with open(log_files[0], encoding="utf-8") as f:
                log_lines = f.readlines()

            # Find the request log entry
            request_log = None
            for line in log_lines:
                entry = json.loads(line)
                if entry.get("type") == "request":
                    request_log = entry
                    break

            assert request_log is not None
            assert request_log["user_agent"] is None


class TestServerUserAgentExtraction:
    """Test server extracts user agent from HTTP headers."""

    def test_openai_endpoint_accepts_user_agent_header(
        self, openai_config, monkeypatch
    ):
        """Test OpenAI endpoint accepts and processes user agent from headers."""
        # Track what user_agent is passed to the router
        captured_user_agent = []

        # Mock the provider to avoid actual API calls
        def mock_create_provider(config):
            mock_provider = Mock()
            mock_provider.chat_completion = AsyncMock(
                return_value={
                    "id": "test-123",
                    "model": "gpt-4",
                    "choices": [{"finish_reason": "stop"}],
                }
            )
            mock_provider.extract_tokens_from_response = Mock(return_value=(10, 8, 0))
            mock_provider.provider_name = "test-openai"
            mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
            mock_provider.__aexit__ = AsyncMock()
            return mock_provider

        def mock_chat_completion(self, request, user_agent=None):
            captured_user_agent.append(user_agent)
            mock_provider = Mock()
            mock_provider.chat_completion = AsyncMock(
                return_value={
                    "id": "test-123",
                    "model": "gpt-4",
                    "choices": [{"finish_reason": "stop"}],
                }
            )
            mock_provider.extract_tokens_from_response = Mock(return_value=(10, 8, 0))
            mock_provider.provider_name = "test-openai"
            mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
            mock_provider.__aexit__ = AsyncMock()

            from llm_api_router.router import LLMRouter

            with patch(
                "llm_api_router.router.create_provider",
                side_effect=mock_create_provider,
            ):
                router = LLMRouter(providers=[openai_config])
                return router.chat_completion(request, user_agent=user_agent)

        with patch(
            "llm_api_router.router.create_provider", side_effect=mock_create_provider
        ):
            # Create server with config
            config = Mock()
            config.openai_providers = [openai_config]
            config.anthropic_providers = []
            config.validate = Mock(return_value=[])

            # Mock the router's chat_completion to capture user_agent
            from llm_api_router import router as router_module

            original_method = router_module.LLMRouter.chat_completion

            async def wrapped_chat_completion(self, request, user_agent=None):
                captured_user_agent.append(user_agent)
                return await original_method(self, request, user_agent=user_agent)

            monkeypatch.setattr(
                router_module.LLMRouter, "chat_completion", wrapped_chat_completion
            )

            server = LLMAPIServer(config)
            client = TestClient(server.app)

            # Make request with user agent header
            response = client.post(
                "/openai/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={"User-Agent": "MyTestClient/1.0"},
            )

            # Verify response is successful
            assert response.status_code == 200
            # Verify user agent was passed through
            assert captured_user_agent == ["MyTestClient/1.0"]

    def test_openai_endpoint_handles_missing_user_agent(
        self, openai_config, monkeypatch
    ):
        """Test OpenAI endpoint handles missing user agent header."""
        captured_user_agent = []

        # Mock the provider
        def mock_create_provider(config):
            mock_provider = Mock()
            mock_provider.chat_completion = AsyncMock(
                return_value={
                    "id": "test-789",
                    "model": "gpt-4",
                    "choices": [{"finish_reason": "stop"}],
                }
            )
            mock_provider.extract_tokens_from_response = Mock(return_value=(10, 8, 0))
            mock_provider.provider_name = "test-openai"
            mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
            mock_provider.__aexit__ = AsyncMock()
            return mock_provider

        with patch(
            "llm_api_router.router.create_provider", side_effect=mock_create_provider
        ):
            # Create server with config
            config = Mock()
            config.openai_providers = [openai_config]
            config.anthropic_providers = []
            config.validate = Mock(return_value=[])

            from llm_api_router import router as router_module

            original_method = router_module.LLMRouter.chat_completion

            async def wrapped_chat_completion(self, request, user_agent=None):
                captured_user_agent.append(user_agent)
                return await original_method(self, request, user_agent=user_agent)

            monkeypatch.setattr(
                router_module.LLMRouter, "chat_completion", wrapped_chat_completion
            )

            server = LLMAPIServer(config)
            client = TestClient(server.app)

            # Make request with empty user agent header (TestClient defaults to "testclient")
            response = client.post(
                "/openai/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={"User-Agent": ""},  # Explicitly set empty user agent
            )

            # Verify response is successful
            assert response.status_code == 200
            # Verify empty string was passed as user agent (FastAPI TestClient uses "testclient" by default, we override with "")
            # When header is empty string, FastAPI passes empty string
            assert captured_user_agent == [""]

    def test_anthropic_endpoint_extracts_user_agent(
        self, anthropic_config, monkeypatch
    ):
        """Test Anthropic endpoint extracts user agent from headers."""
        captured_user_agent = []

        # Mock the provider
        def mock_create_provider(config):
            mock_provider = Mock()
            mock_provider.chat_completion = AsyncMock(
                return_value={
                    "id": "msg-123",
                    "model": "claude-3",
                    "stop_reason": "end_turn",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                }
            )
            mock_provider.extract_tokens_from_response = Mock(return_value=(10, 8, 0))
            mock_provider.provider_name = "test-anthropic"
            mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
            mock_provider.__aexit__ = AsyncMock()
            return mock_provider

        with patch(
            "llm_api_router.router.create_provider", side_effect=mock_create_provider
        ):
            # Create server with config
            config = Mock()
            config.openai_providers = []
            config.anthropic_providers = [anthropic_config]
            config.validate = Mock(return_value=[])

            from llm_api_router import router as router_module

            original_method = router_module.LLMRouter.chat_completion

            async def wrapped_chat_completion(self, request, user_agent=None):
                captured_user_agent.append(user_agent)
                return await original_method(self, request, user_agent=user_agent)

            monkeypatch.setattr(
                router_module.LLMRouter, "chat_completion", wrapped_chat_completion
            )

            server = LLMAPIServer(config)
            client = TestClient(server.app)

            # Make request with user agent header
            response = client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-3-opus",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100,
                },
                headers={"User-Agent": "AnthropicClient/2.0"},
            )

            # Verify response is successful
            assert response.status_code == 200
            # Verify user agent was passed through
            assert captured_user_agent == ["AnthropicClient/2.0"]
