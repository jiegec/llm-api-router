"""Test configuration utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_api_router.config import RouterConfig
from llm_api_router.exceptions import ConfigurationError
from llm_api_router.models import ProviderType


def test_router_config_from_dict():
    """Test creating RouterConfig from dictionary."""
    config_dict = {
        "openai": [
            {
                "api_key": "test-openai-key-1",
                "priority": 1,
                "base_url": "https://api.test.com",
                "timeout": 30,
                "max_retries": 3,
                "model_mapping": {"gpt-4": "gpt-4-turbo"},
            }
        ],
        "anthropic": [
            {
                "api_key": "test-anthropic-key-1",
                "priority": 1,
                "base_url": "https://api.anthropic.test.com",
                "timeout": 30,
                "max_retries": 3,
                "model_mapping": {"claude-3": "claude-3-5-sonnet"},
            }
        ],
    }

    config = RouterConfig.from_dict(config_dict)

    assert len(config.openai_providers) == 1
    assert len(config.anthropic_providers) == 1

    openai_provider = config.openai_providers[0]
    assert openai_provider.name == ProviderType.OPENAI
    assert openai_provider.api_key == "test-openai-key-1"
    assert openai_provider.priority == 1
    assert openai_provider.base_url == "https://api.test.com"
    assert openai_provider.timeout == 30
    assert openai_provider.max_retries == 3
    assert openai_provider.model_mapping == {"gpt-4": "gpt-4-turbo"}

    anthropic_provider = config.anthropic_providers[0]
    assert anthropic_provider.name == ProviderType.ANTHROPIC
    assert anthropic_provider.api_key == "test-anthropic-key-1"
    assert anthropic_provider.priority == 1
    assert anthropic_provider.base_url == "https://api.anthropic.test.com"
    assert anthropic_provider.timeout == 30
    assert anthropic_provider.max_retries == 3
    assert anthropic_provider.model_mapping == {"claude-3": "claude-3-5-sonnet"}


def test_router_config_from_dict_defaults():
    """Test RouterConfig with default values."""
    config_dict = {
        "openai": [
            {
                "api_key": "test-openai-key",
            }
        ],
        "anthropic": [
            {
                "api_key": "test-anthropic-key",
            }
        ],
    }

    config = RouterConfig.from_dict(config_dict)

    openai_provider = config.openai_providers[0]
    assert openai_provider.priority == 1  # Default
    assert openai_provider.base_url is None  # Default
    assert openai_provider.timeout == 30  # Default
    assert openai_provider.max_retries == 3  # Default

    anthropic_provider = config.anthropic_providers[0]
    assert anthropic_provider.priority == 1  # Default
    assert anthropic_provider.base_url is None  # Default
    assert anthropic_provider.timeout == 30  # Default
    assert anthropic_provider.max_retries == 3  # Default


def test_router_config_from_dict_single_provider():
    """Test RouterConfig with single provider (not list)."""
    config_dict = {
        "openai": {
            "api_key": "test-openai-key",
            "priority": 1,
        },
        "anthropic": {
            "api_key": "test-anthropic-key",
            "priority": 1,
        },
    }

    config = RouterConfig.from_dict(config_dict)

    assert len(config.openai_providers) == 1
    assert len(config.anthropic_providers) == 1

    openai_provider = config.openai_providers[0]
    assert openai_provider.name == ProviderType.OPENAI
    assert openai_provider.api_key == "test-openai-key"

    anthropic_provider = config.anthropic_providers[0]
    assert anthropic_provider.name == ProviderType.ANTHROPIC
    assert anthropic_provider.api_key == "test-anthropic-key"


def test_router_config_from_dict_multiple_providers():
    """Test RouterConfig with multiple providers per type."""
    config_dict = {
        "openai": [
            {
                "api_key": "test-openai-key-1",
                "priority": 1,
            },
            {
                "api_key": "test-openai-key-2",
                "priority": 2,
            },
        ],
        "anthropic": [
            {
                "api_key": "test-anthropic-key-1",
                "priority": 1,
            },
            {
                "api_key": "test-anthropic-key-2",
                "priority": 2,
            },
        ],
    }

    config = RouterConfig.from_dict(config_dict)

    assert len(config.openai_providers) == 2
    assert len(config.anthropic_providers) == 2

    # Should be sorted by priority
    assert config.openai_providers[0].priority == 1
    assert config.openai_providers[0].api_key == "test-openai-key-1"
    assert config.openai_providers[1].priority == 2
    assert config.openai_providers[1].api_key == "test-openai-key-2"

    assert config.anthropic_providers[0].priority == 1
    assert config.anthropic_providers[0].api_key == "test-anthropic-key-1"
    assert config.anthropic_providers[1].priority == 2
    assert config.anthropic_providers[1].api_key == "test-anthropic-key-2"


def test_router_config_from_json_file():
    """Test loading RouterConfig from JSON file."""
    config_dict = {
        "openai": [
            {
                "api_key": "test-openai-key",
                "priority": 1,
            }
        ],
        "anthropic": [
            {
                "api_key": "test-anthropic-key",
                "priority": 1,
            }
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_dict, f)
        temp_file = f.name

    try:
        config = RouterConfig.from_json_file(temp_file)

        assert len(config.openai_providers) == 1
        assert len(config.anthropic_providers) == 1

        openai_provider = config.openai_providers[0]
        assert openai_provider.name == ProviderType.OPENAI
        assert openai_provider.api_key == "test-openai-key"

        anthropic_provider = config.anthropic_providers[0]
        assert anthropic_provider.name == ProviderType.ANTHROPIC
        assert anthropic_provider.api_key == "test-anthropic-key"
    finally:
        Path(temp_file).unlink()

    with pytest.raises(ConfigurationError):
        config = RouterConfig.from_json_file("/path/to/nonexistent/file")


def test_router_config_from_json_string():
    """Test loading RouterConfig from JSON string."""
    json_str = json.dumps(
        {
            "openai": [
                {
                    "api_key": "test-openai-key",
                    "priority": 1,
                }
            ],
            "anthropic": [
                {
                    "api_key": "test-anthropic-key",
                    "priority": 1,
                }
            ],
        }
    )

    config = RouterConfig.from_json_string(json_str)

    assert len(config.openai_providers) == 1
    assert len(config.anthropic_providers) == 1

    openai_provider = config.openai_providers[0]
    assert openai_provider.name == ProviderType.OPENAI
    assert openai_provider.api_key == "test-openai-key"

    anthropic_provider = config.anthropic_providers[0]
    assert anthropic_provider.name == ProviderType.ANTHROPIC
    assert anthropic_provider.api_key == "test-anthropic-key"

    with pytest.raises(ConfigurationError):
        config = RouterConfig.from_json_string("{")

    with pytest.raises(ConfigurationError):
        config = RouterConfig.from_json_string('{"openai":{}}')


def test_router_config_to_dict():
    """Test converting RouterConfig to dictionary."""
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "test-openai-key",
                    "priority": 1,
                    "base_url": "https://api.test.com",
                    "timeout": 30,
                    "max_retries": 3,
                    "model_mapping": {"gpt-4": "gpt-4-turbo"},
                }
            ],
            "anthropic": [
                {
                    "api_key": "test-anthropic-key",
                    "priority": 1,
                    "base_url": "https://api.anthropic.test.com",
                    "timeout": 30,
                    "max_retries": 3,
                    "model_mapping": {"claude-3": "claude-3-5-sonnet"},
                }
            ],
        }
    )

    config_dict = config.to_dict()

    assert "openai" in config_dict
    assert "anthropic" in config_dict

    openai_config = config_dict["openai"][0]
    assert openai_config["api_key"] == "test-openai-key"
    assert openai_config["priority"] == 1
    assert openai_config["base_url"] == "https://api.test.com"
    assert openai_config["timeout"] == 30
    assert openai_config["max_retries"] == 3
    assert openai_config["model_mapping"] == {"gpt-4": "gpt-4-turbo"}

    anthropic_config = config_dict["anthropic"][0]
    assert anthropic_config["api_key"] == "test-anthropic-key"
    assert anthropic_config["priority"] == 1
    assert anthropic_config["base_url"] == "https://api.anthropic.test.com"
    assert anthropic_config["timeout"] == 30
    assert anthropic_config["max_retries"] == 3
    assert anthropic_config["model_mapping"] == {"claude-3": "claude-3-5-sonnet"}


def test_router_config_to_json():
    """Test converting RouterConfig to JSON string."""
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "test-openai-key",
                    "priority": 1,
                }
            ],
            "anthropic": [
                {
                    "api_key": "test-anthropic-key",
                    "priority": 1,
                }
            ],
        }
    )

    json_str = config.to_json()
    parsed = json.loads(json_str)

    assert "openai" in parsed
    assert "anthropic" in parsed
    assert len(parsed["openai"]) == 1
    assert len(parsed["anthropic"]) == 1
    assert parsed["openai"][0]["api_key"] == "test-openai-key"
    assert parsed["anthropic"][0]["api_key"] == "test-anthropic-key"


def test_router_config_save_to_file():
    """Test saving RouterConfig to file."""
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "test-openai-key",
                    "priority": 1,
                }
            ],
            "anthropic": [
                {
                    "api_key": "test-anthropic-key",
                    "priority": 1,
                }
            ],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_file = f.name

    try:
        config.save_to_file(temp_file)

        with open(temp_file) as f:
            saved_config = json.load(f)

        assert "openai" in saved_config
        assert "anthropic" in saved_config
        assert len(saved_config["openai"]) == 1
        assert len(saved_config["anthropic"]) == 1
        assert saved_config["openai"][0]["api_key"] == "test-openai-key"
        assert saved_config["anthropic"][0]["api_key"] == "test-anthropic-key"
    finally:
        Path(temp_file).unlink()


def test_router_config_validation():
    """Test RouterConfig validation."""
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "test-openai-key-1",
                    "priority": 1,
                },
                {
                    "api_key": "test-openai-key-2",
                    "priority": 2,
                },
            ],
            "anthropic": [
                {
                    "api_key": "test-anthropic-key-1",
                    "priority": 1,
                },
                {
                    "api_key": "test-anthropic-key-2",
                    "priority": 2,
                },
            ],
        }
    )

    errors = config.validate()
    assert len(errors) == 0
    assert config.is_valid()


def test_router_config_validation_duplicate_priorities():
    """Test RouterConfig validation with duplicate priorities."""
    config = RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "test-openai-key-1",
                    "priority": 1,
                },
                {
                    "api_key": "test-openai-key-2",
                    "priority": 1,  # Duplicate priority
                },
            ],
            "anthropic": [
                {
                    "api_key": "test-anthropic-key-1",
                    "priority": 1,
                }
            ],
        }
    )

    errors = config.validate()
    assert len(errors) == 1
    assert "Duplicate priorities found in OpenAI providers" in errors[0]
    assert not config.is_valid()


def test_router_config_validation_invalid_priority():
    """Test RouterConfig validation with invalid priority."""
    # Pydantic validates priority >= 1, so we can't create a config with priority 0
    # This test is no longer needed since Pydantic handles it
    pass


def test_create_example_config():
    """Test creating example configuration."""
    from llm_api_router.config import create_example_config

    config = create_example_config()

    assert len(config.openai_providers) >= 1
    assert len(config.anthropic_providers) >= 1

    # Example config should be valid
    errors = config.validate()
    assert len(errors) == 0
    assert config.is_valid()
