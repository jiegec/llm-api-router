"""JSON-based configuration for LLM API Router."""

import json
import os
from typing import Any

from .exceptions import ConfigurationError
from .models import ProviderConfig, ProviderType


class RouterConfig:
    """Configuration for LLM API Router."""

    def __init__(
        self,
        openai_providers: list[ProviderConfig] | None = None,
        anthropic_providers: list[ProviderConfig] | None = None,
    ):
        self.openai_providers = openai_providers or []
        self.anthropic_providers = anthropic_providers or []

    @staticmethod
    def _parse_provider_configs(
        configs: dict | list[dict], provider_type: ProviderType
    ) -> list[ProviderConfig]:
        """Parse provider configurations from a dict or list of dicts."""
        if isinstance(configs, dict):
            # Single provider configuration
            configs = [configs]

        providers = []
        for provider_config in configs:
            providers.append(
                ProviderConfig(
                    name=provider_type,
                    api_key=provider_config["api_key"],
                    priority=provider_config.get("priority", 1),
                    base_url=provider_config.get("base_url"),
                    timeout=provider_config.get("timeout", 30),
                    max_retries=provider_config.get("max_retries", 3),
                    model_mapping=provider_config.get("model_mapping", {}),
                    provider_name=provider_config.get("provider_name"),
                )
            )
        return providers

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RouterConfig":
        """Create configuration from dictionary."""
        openai_providers = cls._parse_provider_configs(
            config_dict.get("openai", []), ProviderType.OPENAI
        )
        anthropic_providers = cls._parse_provider_configs(
            config_dict.get("anthropic", []), ProviderType.ANTHROPIC
        )
        return cls(openai_providers, anthropic_providers)

    @classmethod
    def from_json_file(cls, filepath: str) -> "RouterConfig":
        """Load configuration from JSON file."""
        try:
            with open(filepath) as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            raise ConfigurationError(
                f"Configuration file not found: {filepath}"
            ) from None
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}") from e
        except KeyError as e:
            raise ConfigurationError(
                f"Missing required field in configuration: {e}"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}") from e

    @classmethod
    def from_json_string(cls, json_str: str) -> "RouterConfig":
        """Load configuration from JSON string."""
        try:
            config_dict = json.loads(json_str)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON string: {e}") from e
        except KeyError as e:
            raise ConfigurationError(
                f"Missing required field in configuration: {e}"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}") from e

    @staticmethod
    def _provider_to_dict(provider: ProviderConfig) -> dict[str, Any]:
        """Convert a single ProviderConfig to dictionary."""
        return {
            "api_key": provider.api_key,
            "priority": provider.priority,
            "base_url": provider.base_url,
            "timeout": provider.timeout,
            "max_retries": provider.max_retries,
            "model_mapping": provider.model_mapping,
            "provider_name": provider.provider_name,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "openai": [self._provider_to_dict(p) for p in self.openai_providers],
            "anthropic": [self._provider_to_dict(p) for p in self.anthropic_providers],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_to_file(self, filepath: str, indent: int = 2) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @staticmethod
    def _check_duplicate_priorities(
        providers: list[ProviderConfig], provider_type: str
    ) -> list[str]:
        """Check for duplicate priorities in provider list."""
        priorities = [p.priority for p in providers]
        if len(priorities) != len(set(priorities)):
            return [f"Duplicate priorities found in {provider_type} providers"]
        return []

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        errors.extend(self._check_duplicate_priorities(self.openai_providers, "OpenAI"))
        errors.extend(
            self._check_duplicate_priorities(self.anthropic_providers, "Anthropic")
        )
        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


def load_default_config() -> RouterConfig | None:
    """Load configuration from default locations."""
    # Check current directory
    config_paths = [
        "llm_router_config.json",
        "config/llm_router_config.json",
        "~/.llm_router_config.json",
        "/etc/llm_router/config.json",
    ]

    for path in config_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            try:
                return RouterConfig.from_json_file(expanded_path)
            except ConfigurationError:
                continue

    return None


def create_example_config() -> RouterConfig:
    """Create example configuration."""
    return RouterConfig.from_dict(
        {
            "openai": [
                {
                    "api_key": "sk-your-openai-api-key-here",
                    "priority": 1,
                    "provider_name": "openai-primary",
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 30,
                    "max_retries": 3,
                },
                {
                    "api_key": "sk-your-backup-openai-api-key-here",
                    "priority": 2,
                    "provider_name": "openai-backup",
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 30,
                    "max_retries": 3,
                },
            ],
            "anthropic": [
                {
                    "api_key": "sk-ant-your-anthropic-api-key-here",
                    "priority": 1,
                    "provider_name": "anthropic-primary",
                    "base_url": "https://api.anthropic.com",
                    "timeout": 30,
                    "max_retries": 3,
                }
            ],
        }
    )
