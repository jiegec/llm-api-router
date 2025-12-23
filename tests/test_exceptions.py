"""Test exceptions."""

import pytest
from llm_api_router.exceptions import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    ProviderError,
    NoAvailableProviderError,
    ConfigurationError,
)


def test_llm_error():
    """Test LLMError."""
    error = LLMError("Test error", "openai", 400)
    assert str(error) == "Test error"
    assert error.message == "Test error"
    assert error.provider == "openai"
    assert error.status_code == 400


def test_llm_error_defaults():
    """Test LLMError with defaults."""
    error = LLMError("Test error")
    assert error.message == "Test error"
    assert error.provider is None
    assert error.status_code is None


def test_rate_limit_error():
    """Test RateLimitError."""
    error = RateLimitError("Too many requests", "openai", 60)
    assert str(error) == "Rate limit exceeded for openai: Too many requests"
    assert error.message == "Rate limit exceeded for openai: Too many requests"
    assert error.provider == "openai"
    assert error.retry_after == 60


def test_rate_limit_error_no_retry_after():
    """Test RateLimitError without retry_after."""
    error = RateLimitError("Too many requests", "anthropic")
    assert str(error) == "Rate limit exceeded for anthropic: Too many requests"
    assert error.provider == "anthropic"
    assert error.retry_after is None


def test_authentication_error():
    """Test AuthenticationError."""
    error = AuthenticationError("Invalid API key", "openai")
    assert str(error) == "Authentication failed for openai: Invalid API key"
    assert error.message == "Authentication failed for openai: Invalid API key"
    assert error.provider == "openai"


def test_provider_error():
    """Test ProviderError."""
    error_data = {"code": "invalid_request"}
    error = ProviderError("Invalid request", "openai", "invalid_request", error_data)
    assert str(error) == "Provider error for openai: Invalid request"
    assert error.message == "Provider error for openai: Invalid request"
    assert error.provider == "openai"
    assert error.error_type == "invalid_request"
    assert error.error_data == error_data


def test_provider_error_defaults():
    """Test ProviderError with defaults."""
    error = ProviderError("Unknown error", "anthropic")
    assert str(error) == "Provider error for anthropic: Unknown error"
    assert error.error_type is None
    assert error.error_data is None


def test_no_available_provider_error():
    """Test NoAvailableProviderError."""
    error = NoAvailableProviderError("All providers failed")
    assert str(error) == "All providers failed"
    assert error.message == "All providers failed"


def test_no_available_provider_error_default():
    """Test NoAvailableProviderError with default message."""
    error = NoAvailableProviderError()
    assert str(error) == "No available providers"


def test_configuration_error():
    """Test ConfigurationError."""
    error = ConfigurationError("Invalid configuration")
    assert str(error) == "Configuration error: Invalid configuration"
    assert error.message == "Configuration error: Invalid configuration"