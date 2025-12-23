"""Anthropic provider implementation."""

import time
from typing import Dict, Any, List, Optional

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


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""
    
    MODEL_MAPPING = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-2": "claude-2.1",
        "claude-instant": "claude-instant-1.2",
    }
    
    def _get_default_base_url(self) -> str:
        return "https://api.anthropic.com"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Anthropic API."""
        headers = super()._get_headers()
        headers.update({
            "anthropic-version": "2023-06-01",
            "x-api-key": self.config.api_key,
        })
        # Remove Bearer prefix for Anthropic
        headers.pop("Authorization", None)
        return headers
    
    def _map_model(self, model: str) -> str:
        """Map generic model name to Anthropic model name."""
        return self.MODEL_MAPPING.get(model, model)
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Anthropic API."""
        formatted = []

        # Extract system message if present
        for msg in messages:
            role_value = msg.role.value if isinstance(msg.role, Role) else msg.role
            if role_value != Role.SYSTEM.value:  # Skip system messages for Anthropic format
                formatted_msg = {
                    "role": role_value,
                    "content": msg.content
                }
                formatted.append(formatted_msg)

        return formatted

    def _extract_system_message(self, messages: List[Message]) -> Optional[str]:
        """Extract system message from messages."""
        for msg in messages:
            role_value = msg.role.value if isinstance(msg.role, Role) else msg.role
            if role_value == Role.SYSTEM.value:
                return msg.content
        return None
    
    def _parse_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse Anthropic response to standard format."""
        # Anthropic returns content as a list
        content_list = response.get("content", [])
        content = ""
        for content_item in content_list:
            if content_item.get("type") == "text":
                content = content_item.get("text", "")
                break
        
        message = Message(
            role=Role.ASSISTANT,
            content=content,
        )
        
        choice = Choice(
            index=0,
            message=message,
            finish_reason=response.get("stop_reason"),
        )
        
        usage_data = response.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )
        
        return ChatCompletionResponse(
            id=response.get("id", f"anthropic-{int(time.time())}"),
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage,
            provider=ProviderType.ANTHROPIC,
        )
    
    def _get_endpoint(self) -> str:
        return "/v1/messages"
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Send a chat completion request to Anthropic."""
        provider_model = self._map_model(request.model)
        formatted_messages = self._format_messages(request.messages)
        system_message = self._extract_system_message(request.messages)

        payload = {
            "model": provider_model,
            "messages": formatted_messages,
            "max_tokens": request.max_tokens or 1000,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop_sequences": request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
            "stream": request.stream,
        }

        if system_message:
            payload["system"] = system_message

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.post(
                    self._get_endpoint(),
                    json=payload
                )

                if response.status_code == 200:
                    response_data = response.json()
                    return self._parse_response(response_data, request.model)
                else:
                    try:
                        error_data = response.json()
                    except Exception:
                        error_data = {"error": {"message": response.text}}

                    self._handle_error(response.status_code, error_data)

            except (RateLimitError, AuthenticationError) as e:
                raise e
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise ProviderError(
                        f"Request failed after {self.config.max_retries + 1} attempts: {str(e)}",
                        self.config.name.value
                    )
                import asyncio
                await asyncio.sleep(2 ** attempt)