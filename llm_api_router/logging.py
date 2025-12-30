"""Logging system for LLM API Router."""

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import Usage


class RouterLogger:
    """Logger for LLM API Router requests, responses, and retries."""

    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """Initialize the logger.

        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Generate a unique session ID for this run
        self.session_id = str(uuid4())[:8]

        # Get current datetime for filename
        self.start_time = datetime.now(timezone.utc)
        self.datetime_str = self.start_time.strftime("%Y%m%d_%H%M%S")

        # Set up structured logging
        self._setup_logging(log_level)

    def _setup_logging(self, log_level: str) -> None:
        """Set up logging configuration."""
        # Create main logger for console output
        self.logger = logging.getLogger("llm_api_router")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler (detailed)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(console_handler)

        # Create separate logger for structured JSON logs
        self.json_logger = logging.getLogger(f"llm_api_router.json.{self.session_id}")
        self.json_logger.setLevel(logging.DEBUG)
        self.json_logger.handlers.clear()
        self.json_logger.propagate = False  # Don't propagate to parent logger

        # JSON formatter - just passes through the message
        json_formatter = logging.Formatter("%(message)s")

        # File handler for structured JSON logs - include datetime in filename
        log_file = self.log_dir / f"router_{self.datetime_str}_{self.session_id}.jsonl"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_formatter)
        self.json_logger.addHandler(file_handler)

        # Separate error log file - include datetime in filename
        error_file = (
            self.log_dir / f"router_{self.datetime_str}_{self.session_id}_errors.jsonl"
        )
        error_handler = logging.FileHandler(error_file, encoding="utf-8")
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(json_formatter)
        self.json_logger.addHandler(error_handler)

        self.logger.info(f"Logging initialized. Session ID: {self.session_id}")
        self.logger.info(f"Log files: {log_file}, {error_file}")

    def log_request(
        self,
        request_id: str,
        endpoint: str,
        request: dict[str, Any],
        provider_name: str,
        provider_priority: int,
    ) -> None:
        """Log an incoming request."""
        # Extract values from request dict
        model = request.get("model", "")
        message_count = len(request.get("messages", []))
        temperature = request.get("temperature")
        max_tokens = request.get("max_tokens")
        stream = request.get("stream", False)

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "type": "request",
            "session_id": self.session_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "provider": provider_name,
            "provider_priority": provider_priority,
            "model": model,
            "message_count": message_count,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "full_request": request,  # Log the complete request
        }

        # Log structured JSON to file
        self.json_logger.debug(json.dumps(log_entry))

        # Also log a summary at INFO level to console
        self.logger.info(
            f"Request {request_id[:8]} to {endpoint}: "
            f"model={model}, "
            f"provider={provider_name} (priority {provider_priority}), "
            f"messages={message_count}"
        )

    def log_response(
        self,
        request_id: str,
        endpoint: str,
        response: dict[str, Any],
        provider_name: str,
        duration_ms: float,
    ) -> None:
        """Log a successful response."""
        # Extract values from response (now always a dict)
        model = response.get("model", "")
        response_id = response.get("id", "")
        usage_data = response.get("usage")
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
        else:
            usage = None
        choices = response.get("choices", [])
        has_content = bool(choices and choices[0].get("message", {}).get("content"))
        finish_reason = choices[0].get("finish_reason") if choices else None
        full_response = response

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "type": "response",
            "session_id": self.session_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "provider": provider_name,
            "model": model,
            "response_id": response_id,
            "duration_ms": duration_ms,
            "usage": (
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage
                else None
            ),
            "has_content": has_content,
            "finish_reason": finish_reason,
            "full_response": full_response,  # Log the complete response
        }

        self.json_logger.debug(json.dumps(log_entry))

        self.logger.info(
            f"Response {request_id[:8]} from {provider_name}: "
            f"duration={duration_ms:.0f}ms, "
            f"tokens={usage.total_tokens if usage else 'N/A'}, "
            f"response_id={response_id[:8] if response_id else 'N/A'}"
        )

    def log_retry(
        self,
        request_id: str,
        endpoint: str,
        provider_name: str,
        provider_priority: int,
        attempt: int,
        max_attempts: int,
        error_type: str,
        error_message: str,
        retry_after: int | None = None,
    ) -> None:
        """Log a retry attempt."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "type": "retry",
            "session_id": self.session_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "provider": provider_name,
            "provider_priority": provider_priority,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "error_type": error_type,
            "error_message": error_message,
            "retry_after": retry_after,
        }

        self.json_logger.warning(json.dumps(log_entry))

        retry_msg = (
            f"Retry {request_id[:8]}: "
            f"attempt {attempt}/{max_attempts} for {provider_name} "
            f"(priority {provider_priority}) - {error_type}: {error_message}"
        )
        if retry_after:
            retry_msg += f" (retry after: {retry_after}s)"

        self.logger.warning(retry_msg)

    def log_error(
        self,
        request_id: str,
        endpoint: str,
        provider_name: str,
        error_type: str,
        error_message: str,
        status_code: int | None = None,
    ) -> None:
        """Log an error."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "type": "error",
            "session_id": self.session_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "provider": provider_name,
            "error_type": error_type,
            "error_message": error_message,
            "status_code": status_code,
        }

        self.json_logger.error(json.dumps(log_entry))

        error_msg = (
            f"Error {request_id[:8]} from {provider_name}: "
            f"{error_type}: {error_message}"
        )
        if status_code:
            error_msg += f" (status: {status_code})"

        self.logger.error(error_msg)

    def log_fallback(
        self,
        request_id: str,
        endpoint: str,
        from_provider: str,
        from_priority: int,
        to_provider: str,
        to_priority: int,
        reason: str,
    ) -> None:
        """Log a fallback to another provider."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "type": "fallback",
            "session_id": self.session_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "from_provider": from_provider,
            "from_priority": from_priority,
            "to_provider": to_provider,
            "to_priority": to_priority,
            "reason": reason,
        }

        self.json_logger.warning(json.dumps(log_entry))

        self.logger.warning(
            f"Fallback {request_id[:8]}: "
            f"{from_provider} (priority {from_priority}) -> "
            f"{to_provider} (priority {to_priority}) - {reason}"
        )

    def log_provider_selection(
        self,
        request_id: str,
        endpoint: str,
        selected_provider: str,
        selected_priority: int,
        available_providers: list[tuple[str, int]],
    ) -> None:
        """Log provider selection."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "type": "provider_selection",
            "session_id": self.session_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "selected_provider": selected_provider,
            "selected_priority": selected_priority,
            "available_providers": [
                {"name": name, "priority": priority}
                for name, priority in available_providers
            ],
        }

        self.json_logger.debug(json.dumps(log_entry))

        providers_str = ", ".join(
            f"{name} (priority {priority})" for name, priority in available_providers
        )

        self.logger.debug(
            f"Provider selection {request_id[:8]}: "
            f"selected {selected_provider} (priority {selected_priority}) "
            f"from available: {providers_str}"
        )

    def log_configuration(
        self,
        config_type: str,
        details: dict[str, Any],
    ) -> None:
        """Log configuration details."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "type": "configuration",
            "session_id": self.session_id,
            "config_type": config_type,
            "details": details,
        }

        self.json_logger.info(json.dumps(log_entry))

        if config_type == "router_start":
            self.logger.info(
                f"Router started with {details.get('provider_count', 0)} providers"
            )


# Global logger instance
_logger: RouterLogger | None = None


def get_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    force_new: bool = False,
) -> RouterLogger:
    """Get or create the global logger instance."""
    global _logger

    if _logger is None or force_new:
        _logger = RouterLogger(log_dir, log_level)

    return _logger


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
) -> RouterLogger:
    """Set up logging and return the logger instance."""
    return get_logger(log_dir, log_level, force_new=True)
