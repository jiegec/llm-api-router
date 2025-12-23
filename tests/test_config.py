"""Test configuration utilities."""

import os
import pytest
from unittest.mock import patch

from llm_api_router.config import load_providers_from_env, create_router_from_env
from llm_api_router.models import ProviderType


def test_load_providers_from_env_empty():
    """Test loading providers with empty environment."""
    with patch.dict(os.environ, clear=True):
        providers = load_providers_from_env()
        assert providers == []


def test_load_providers_from_env_openai():
    """Test loading OpenAI provider from environment."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_PRIORITY": "1",
        "OPENAI_BASE_URL": "https://api.test.com",
    }):
        providers = load_providers_from_env()
        
        assert len(providers) == 1
        provider = providers[0]
        assert provider.name == ProviderType.OPENAI
        assert provider.api_key == "test-openai-key"
        assert provider.priority == 1
        assert provider.base_url == "https://api.test.com"


def test_load_providers_from_env_anthropic():
    """Test loading Anthropic provider from environment."""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "ANTHROPIC_PRIORITY": "2",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.test.com",
    }):
        providers = load_providers_from_env()
        
        assert len(providers) == 1
        provider = providers[0]
        assert provider.name == ProviderType.ANTHROPIC
        assert provider.api_key == "test-anthropic-key"
        assert provider.priority == 2
        assert provider.base_url == "https://api.anthropic.test.com"


def test_load_providers_from_env_both():
    """Test loading both providers from environment."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_PRIORITY": "1",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "ANTHROPIC_PRIORITY": "2",
    }):
        providers = load_providers_from_env()
        
        assert len(providers) == 2
        
        # Should be sorted by priority
        assert providers[0].name == ProviderType.OPENAI
        assert providers[0].priority == 1
        assert providers[1].name == ProviderType.ANTHROPIC
        assert providers[1].priority == 2


def test_load_providers_from_env_default_priority():
    """Test loading providers with default priority."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
    }):
        providers = load_providers_from_env()
        
        assert len(providers) == 2
        assert providers[0].name == ProviderType.OPENAI
        assert providers[0].priority == 1  # Default
        assert providers[1].name == ProviderType.ANTHROPIC
        assert providers[1].priority == 2  # Default


def test_load_providers_from_env_custom_priority():
    """Test loading providers with custom priority order."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_PRIORITY": "3",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "ANTHROPIC_PRIORITY": "1",
    }):
        providers = load_providers_from_env()
        
        assert len(providers) == 2
        # Should be sorted by priority
        assert providers[0].name == ProviderType.ANTHROPIC  # Priority 1
        assert providers[0].priority == 1
        assert providers[1].name == ProviderType.OPENAI  # Priority 3
        assert providers[1].priority == 3


def test_create_router_from_env_empty():
    """Test creating router from empty environment."""
    with patch.dict(os.environ, clear=True):
        router = create_router_from_env()
        assert router is None


def test_create_router_from_env_with_providers():
    """Test creating router from environment with providers."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
    }):
        router = create_router_from_env()
        assert router is not None
        assert len(router.providers) == 2
        assert router.providers[0].name == ProviderType.OPENAI
        assert router.providers[1].name == ProviderType.ANTHROPIC


def test_create_router_from_env_single_provider():
    """Test creating router from environment with single provider."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
    }):
        router = create_router_from_env()
        assert router is not None
        assert len(router.providers) == 1
        assert router.providers[0].name == ProviderType.OPENAI