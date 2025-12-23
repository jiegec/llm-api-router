"""OpenAI provider implementation."""

import time
from typing import Dict, Any, List

from .base import BaseProvider
from ..models import (
    ProviderConfig,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    Choice,
    Usage,
    ProviderType,
    Role
)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""
    
    MODEL_MAPPING = {
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-instruct",
    }
    
    def _get_default_base_url(self) -> str:
        return "https://api.openai.com/v1"
    
    def _map_model(self, model: str) -> str:
        """Map generic model name to OpenAI model name."""
        return self.MODEL_MAPPING.get(model, model)
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API."""
        formatted = []
        for msg in messages:
            formatted_msg = {
                "role": msg.role.value,
                "content": msg.content
            }
            if msg.name:
                formatted_msg["name"] = msg.name
            if msg.tool_calls:
                formatted_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            formatted.append(formatted_msg)
        return formatted
    
    def _parse_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse OpenAI response to standard format."""
        choices = []
        for choice_data in response.get("choices", []):
            message_data = choice_data.get("message", {})
            message = Message(
                role=Role(message_data.get("role", "assistant")),
                content=message_data.get("content", ""),
                name=message_data.get("name"),
                tool_calls=message_data.get("tool_calls"),
                tool_call_id=message_data.get("tool_call_id"),
            )
            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
            )
            choices.append(choice)
        
        usage_data = response.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        
        return ChatCompletionResponse(
            id=response.get("id", ""),
            created=response.get("created", int(time.time())),
            model=model,
            choices=choices,
            usage=usage,
            provider=ProviderType.OPENAI,
        )
    
    def _get_endpoint(self) -> str:
        return "/chat/completions"