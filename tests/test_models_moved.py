"""Test models for ChatCompletionResponse, Choice, and ChatCompletionRequest."""

from pydantic import BaseModel, Field

from llm_api_router.models import Message, ProviderType, Usage


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


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    provider: ProviderType
