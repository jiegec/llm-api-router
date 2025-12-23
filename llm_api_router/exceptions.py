"""Exceptions for LLM API Router."""

from typing import Any


class LLMError(Exception):
    """Base exception for LLM API errors."""
    def __init__(self, message: str, provider: str | None = None, status_code: int | None = None):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, provider: str, retry_after: int | None = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {provider}: {message}", provider)


class AuthenticationError(LLMError):
    """Raised when authentication fails."""
    def __init__(self, message: str, provider: str):
        super().__init__(f"Authentication failed for {provider}: {message}", provider)


class ProviderError(LLMError):
    """Raised when a provider-specific error occurs."""
    def __init__(self, message: str, provider: str, error_type: str | None = None,
                 error_data: dict[str, Any] | None = None):
        self.error_type = error_type
        self.error_data = error_data
        super().__init__(f"Provider error for {provider}: {message}", provider)


class NoAvailableProviderError(LLMError):
    """Raised when no providers are available."""
    def __init__(self, message: str = "No available providers"):
        super().__init__(message)


class ConfigurationError(LLMError):
    """Raised when there's a configuration error."""
    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")
