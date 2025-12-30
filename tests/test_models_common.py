"""Test models for LLM API Router."""

from enum import Enum
from typing import Any

from pydantic import BaseModel

from llm_api_router.models import Usage


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


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    messages: list[Message]
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None
    stream: bool = False


class Choice(BaseModel):
    """A choice in the chat completion response."""

    index: int
    message: Message
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
