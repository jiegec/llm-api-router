"""OpenAI provider implementation."""

import time
from typing import Any

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def _get_default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for OpenAI API."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _get_endpoint(self) -> str:
        return "/chat/completions"

    def _get_count_tokens_endpoint(self) -> str:
        raise NotImplementedError(
            "OpenAI does not have a count_tokens endpoint at this time"
        )

    def merge_streaming_chunk(
        self, response_dict: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge a streaming chunk from OpenAI into a response dict."""
        # Initialize response dict with metadata from first chunk
        if not response_dict:
            response_dict = {
                "id": data.get("id", ""),
                "object": "chat.completion",
                "created": data.get("created", int(time.time())),
                "model": data.get("model", ""),
                "choices": [],
            }

        # Merge content from delta
        if "choices" in data and data["choices"]:
            for choice in data["choices"]:
                # Update existing choice or add new one
                choice_idx = choice.get("index", 0)
                while len(response_dict["choices"]) <= choice_idx:
                    response_dict["choices"].append(
                        {
                            "index": len(response_dict["choices"]),
                            "message": {
                                "content": "",
                                "tool_calls": [],
                            },
                        }
                    )
                existing_message = response_dict["choices"][choice_idx]["message"]

                if "delta" in choice:
                    delta = choice["delta"]
                    # Set role
                    if "role" in delta:
                        existing_message["role"] = delta["role"]
                    # Accumulate content
                    if "content" in delta:
                        existing_message["content"] += delta["content"]
                    # Accumulate reasoning content (for reasoning models)
                    if "reasoning_content" in delta:
                        if "reasoning_content" not in existing_message:
                            existing_message["reasoning_content"] = ""
                        existing_message["reasoning_content"] += delta[
                            "reasoning_content"
                        ]
                    # Accumulate tool calls
                    if "tool_calls" in delta:
                        for tool_call in delta["tool_calls"]:
                            tool_call_idx = tool_call.get("index", 0)
                            # Extend tool_calls list if needed
                            while len(existing_message["tool_calls"]) <= tool_call_idx:
                                existing_message["tool_calls"].append({})
                            # Merge tool call fields
                            existing = existing_message["tool_calls"][tool_call_idx]
                            if "id" in tool_call:
                                existing["id"] = tool_call["id"]
                            if "type" in tool_call:
                                existing["type"] = tool_call["type"]
                            if "function" in tool_call:
                                if "function" not in existing:
                                    existing["function"] = {
                                        "name": "",
                                        "arguments": "",
                                    }
                                func = existing["function"]
                                if "name" in tool_call["function"]:
                                    func["name"] = tool_call["function"]["name"]
                                if "arguments" in tool_call["function"]:
                                    func["arguments"] += tool_call["function"][
                                        "arguments"
                                    ]
                # Store finish reason if present
                if "finish_reason" in choice:
                    response_dict["choices"][choice_idx]["finish_reason"] = choice[
                        "finish_reason"
                    ]
        # Store usage data if present
        if "usage" in data:
            response_dict["usage"] = data["usage"]
        return response_dict

    def postprocess_response(self, response_dict: dict[str, Any]) -> dict[str, Any]:
        """Postprocess response dict from OpenAI."""
        # No postprocessing needed for OpenAI
        return response_dict

    def extract_tokens_from_response(
        self, response: dict[str, Any]
    ) -> tuple[int, int, int]:
        """Extract token counts from OpenAI response."""
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cached_tokens = (
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if "prompt_tokens_details" in usage
            else 0
        )
        return input_tokens, output_tokens, cached_tokens
