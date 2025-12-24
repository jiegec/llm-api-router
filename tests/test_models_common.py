"""Test models for LLM API Router."""

from typing import Any

from llm_api_router.models import Message, ProviderType, Role, Usage


class ChatCompletionRequest:
    """Request for chat completion."""

    def __init__(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: str | list[str] | None = None,
        stream: bool = False,
    ):
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.stream = stream

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value, like a dict."""
        return getattr(self, key, default)


class Choice:
    """A choice in the chat completion response."""

    def __init__(
        self,
        index: int,
        message: Message,
        finish_reason: str | None = None,
    ):
        self.index = index
        self.message = message
        self.finish_reason = finish_reason


class ChatCompletionResponse:
    """Response from chat completion."""

    def __init__(
        self,
        id: str,
        created: int,
        model: str,
        choices: list[Choice],
        usage: Usage,
        provider: ProviderType,
    ):
        self.id = id
        self.object = "chat.completion"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.provider = provider