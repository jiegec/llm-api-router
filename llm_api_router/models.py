"""Data models for LLM API Router."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A message in a chat conversation."""

    role: Role
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


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
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


