"""Tests for CSV logging functionality."""

import csv
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_api_router.logging import RouterLogger, get_logger


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary log directory."""
    return str(tmp_path / "logs")


@pytest.fixture
def csv_logger(temp_log_dir):
    """Create a RouterLogger with CSV logging enabled."""
    return RouterLogger(log_dir=temp_log_dir, log_level="INFO")


class TestCSVLoggingSetup:
    """Test CSV logging setup and file creation."""

    def test_csv_file_created_with_headers(self, temp_log_dir):
        """Test that CSV file is created with proper headers when it doesn't exist."""
        logger = RouterLogger(log_dir=temp_log_dir, log_level="INFO")
        csv_path = Path(temp_log_dir) / "request_stats.csv"

        assert csv_path.exists()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)

        assert headers == logger.CSV_COLUMNS

    def test_csv_file_appends_to_existing(self, temp_log_dir):
        """Test that existing CSV file is appended to, not overwritten."""
        csv_path = Path(temp_log_dir) / "request_stats.csv"
        temp_log_dir_path = Path(temp_log_dir)
        temp_log_dir_path.mkdir(parents=True, exist_ok=True)

        # Create existing CSV with some data
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RouterLogger.CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "provider_type": "openai",
                    "provider_name": "test-provider",
                    "latency_ms": 100.0,
                    "is_streaming": False,
                    "time_to_first_token_ms": "",
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cached_tokens": 5,
                }
            )

        # Create logger - should append, not overwrite
        RouterLogger(log_dir=temp_log_dir, log_level="INFO")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["provider_name"] == "test-provider"

    def test_csv_columns_are_correct(self, csv_logger):
        """Test that CSV columns match expected format."""
        expected_columns = [
            "timestamp",
            "provider_type",
            "provider_name",
            "latency_ms",
            "is_streaming",
            "time_to_first_token_ms",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
        ]
        assert csv_logger.CSV_COLUMNS == expected_columns


class TestCSVLoggingFunctionality:
    """Test CSV logging for different request types."""

    def test_log_response_non_streaming(self, csv_logger, temp_log_dir):
        """Test logging a non-streaming response to CSV."""
        mock_provider = Mock()
        mock_provider.extract_tokens_from_response.return_value = (10, 20, 5)

        response = {
            "id": "resp-123",
            "model": "gpt-4",
            "choices": [{"finish_reason": "stop"}],
        }

        csv_logger.log_response(
            request_id="req-123",
            endpoint="openai/chat/completions",
            response=response,
            provider_name="test-openai",
            duration_ms=150.5,
            provider=mock_provider,
            is_streaming=False,
        )

        csv_path = Path(temp_log_dir) / "request_stats.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["provider_type"] == "openai"
        assert row["provider_name"] == "test-openai"
        assert float(row["latency_ms"]) == 150.5
        assert row["is_streaming"] == "False"
        assert row["time_to_first_token_ms"] == ""
        assert int(row["input_tokens"]) == 10
        assert int(row["output_tokens"]) == 20
        assert int(row["cached_tokens"]) == 5

    def test_log_response_streaming(self, csv_logger, temp_log_dir):
        """Test logging a streaming response with TTFT to CSV."""
        mock_provider = Mock()
        mock_provider.extract_tokens_from_response.return_value = (15, 25, 0)

        response = {
            "id": "resp-456",
            "model": "claude-3",
            "stop_reason": "end_turn",
        }

        csv_logger.log_response(
            request_id="req-456",
            endpoint="anthropic/v1/messages",
            response=response,
            provider_name="test-anthropic",
            duration_ms=500.75,
            provider=mock_provider,
            time_to_first_token_ms=50.25,
            is_streaming=True,
        )

        csv_path = Path(temp_log_dir) / "request_stats.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["provider_type"] == "anthropic"
        assert row["provider_name"] == "test-anthropic"
        assert float(row["latency_ms"]) == 500.75
        assert row["is_streaming"] == "True"
        assert float(row["time_to_first_token_ms"]) == 50.25
        assert int(row["input_tokens"]) == 15
        assert int(row["output_tokens"]) == 25
        assert int(row["cached_tokens"]) == 0

    def test_log_response_count_tokens_not_logged(self, csv_logger, temp_log_dir):
        """Test that count_tokens responses are not logged to CSV."""
        mock_provider = Mock()

        # count_tokens response format has input_tokens but no model
        response = {
            "input_tokens": 42,
        }

        csv_logger.log_response(
            request_id="req-789",
            endpoint="anthropic/v1/messages/count_tokens",
            response=response,
            provider_name="test-anthropic",
            duration_ms=25.0,
            provider=mock_provider,
        )

        csv_path = Path(temp_log_dir) / "request_stats.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have no rows since count_tokens is not logged to CSV
        assert len(rows) == 0

    def test_multiple_requests_appended(self, csv_logger, temp_log_dir):
        """Test that multiple requests are appended to the CSV file."""
        mock_provider = Mock()
        mock_provider.extract_tokens_from_response.return_value = (10, 20, 0)

        responses = [
            {
                "id": f"resp-{i}",
                "model": "gpt-4",
                "choices": [{"finish_reason": "stop"}],
            }
            for i in range(5)
        ]

        for i, response in enumerate(responses):
            csv_logger.log_response(
                request_id=f"req-{i}",
                endpoint="openai/chat/completions",
                response=response,
                provider_name=f"provider-{i % 2}",
                duration_ms=100.0 + i * 10,
                provider=mock_provider,
                is_streaming=i % 2 == 0,
            )

        csv_path = Path(temp_log_dir) / "request_stats.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert row["provider_name"] == f"provider-{i % 2}"
            assert float(row["latency_ms"]) == 100.0 + i * 10
            assert row["is_streaming"] == str(i % 2 == 0)

    def test_unknown_provider_type(self, csv_logger, temp_log_dir):
        """Test that unknown endpoints result in 'unknown' provider type."""
        mock_provider = Mock()
        mock_provider.extract_tokens_from_response.return_value = (5, 5, 0)

        response = {
            "id": "resp-999",
            "model": "unknown-model",
            "choices": [{"finish_reason": "stop"}],
        }

        csv_logger.log_response(
            request_id="req-999",
            endpoint="/some/other/endpoint",
            response=response,
            provider_name="custom-provider",
            duration_ms=75.0,
            provider=mock_provider,
        )

        csv_path = Path(temp_log_dir) / "request_stats.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["provider_type"] == "unknown"


class TestCSVLoggingErrorHandling:
    """Test error handling in CSV logging."""

    def test_csv_logging_error_does_not_fail_request(
        self, csv_logger, temp_log_dir, caplog
    ):
        """Test that CSV logging errors don't fail the request."""
        mock_provider = Mock()
        mock_provider.extract_tokens_from_response.return_value = (10, 20, 0)

        # Make the CSV file read-only to cause an error
        csv_path = Path(temp_log_dir) / "request_stats.csv"
        csv_path.chmod(0o444)

        response = {
            "id": "resp-123",
            "model": "gpt-4",
            "choices": [{"finish_reason": "stop"}],
        }

        # Should not raise an exception
        csv_logger.log_response(
            request_id="req-123",
            endpoint="openai/chat/completions",
            response=response,
            provider_name="test-openai",
            duration_ms=150.0,
            provider=mock_provider,
        )

        # Restore permissions for cleanup
        csv_path.chmod(0o644)

        # Check that a warning was logged
        assert "Failed to write to CSV log" in caplog.text


class TestCSVLoggerSingleton:
    """Test the global logger instance with CSV logging."""

    def test_get_logger_creates_csv_file(self, tmp_path, monkeypatch):
        """Test that get_logger creates the CSV file."""
        log_dir = str(tmp_path / "logs")
        monkeypatch.setattr("llm_api_router.logging._logger", None)

        get_logger(log_dir=log_dir, log_level="INFO", force_new=True)

        csv_path = Path(log_dir) / "request_stats.csv"
        assert csv_path.exists()

    def test_global_logger_csv_functionality(self, tmp_path, monkeypatch):
        """Test CSV logging works with the global logger."""
        log_dir = str(tmp_path / "logs")
        monkeypatch.setattr("llm_api_router.logging._logger", None)

        logger = get_logger(log_dir=log_dir, log_level="INFO", force_new=True)

        mock_provider = Mock()
        mock_provider.extract_tokens_from_response.return_value = (100, 200, 50)

        response = {
            "id": "resp-global",
            "model": "gpt-4",
            "choices": [{"finish_reason": "stop"}],
        }

        logger.log_response(
            request_id="req-global",
            endpoint="openai/chat/completions",
            response=response,
            provider_name="global-test",
            duration_ms=300.0,
            provider=mock_provider,
            is_streaming=False,
        )

        csv_path = Path(log_dir) / "request_stats.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["provider_name"] == "global-test"
        assert int(rows[0]["input_tokens"]) == 100
        assert int(rows[0]["output_tokens"]) == 200
        assert int(rows[0]["cached_tokens"]) == 50
