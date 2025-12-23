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


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    messages: list[Message]
    model: str
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1000, ge=1)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    stream: bool = False


class Choice(BaseModel):
    """A choice in the chat completion response."""
    index: int
    message: Message
    finish_reason: str | None = None


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    provider: ProviderType
