"""Data models for LLM API Router."""

from enum import Enum

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    name: ProviderType
    api_key: str
    priority: int = Field(ge=1, description="Lower number = higher priority")
    base_url: str | None = None
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    model_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Map generic model names to provider-specific names",
    )
    provider_name: str | None = Field(
        default=None,
        description="User-defined name for the provider (e.g., 'openai-priority-1'). "
        "If not provided, defaults to '{provider_type}-priority-{priority}'",
    )

    @property
    def display_name(self) -> str:
        """Get the display name for the provider."""
        if self.provider_name:
            return self.provider_name
        return f"{self.name.value}-priority-{self.priority}"


class Usage(BaseModel):
    """Token usage information.

    Note: prompt_tokens includes cached_tokens (i.e., cached tokens are a subset
    of prompt_tokens, not a separate addition). This matches the OpenAI semantic.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
