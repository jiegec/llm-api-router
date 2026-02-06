"""Test OpenAI provider streaming chunk merging."""

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llm_api_router.models import ProviderConfig, ProviderType
from llm_api_router.providers.openai import OpenAIProvider


def _chat_completion_chunk_to_dict(chunk: ChatCompletionChunk) -> dict:
    """Convert ChatCompletionChunk to dict."""
    return chunk.model_dump(mode="json", exclude_none=True)


def _validate_chat_completion_dict(data: dict) -> ChatCompletion:
    """Validate that a dict conforms to ChatCompletion schema."""
    return ChatCompletion.model_construct(**data)


def _get_provider() -> OpenAIProvider:
    """Create an OpenAI provider instance for testing."""
    config = ProviderConfig(
        name=ProviderType.OPENAI,
        api_key="test-key",
        priority=1,
    )
    return OpenAIProvider(config)


def test_merge_openai_chunk_initializes_from_first_chunk():
    """Test that merge_streaming_chunk initializes response dict from first chunk."""
    provider = _get_provider()

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": None,
            }
        ],
    )

    data = _chat_completion_chunk_to_dict(chunk)
    result = provider.merge_streaming_chunk({}, data)

    assert result["id"] == "chatcmpl-123"
    assert result["object"] == "chat.completion"
    assert result["created"] == 1234567890
    assert result["model"] == "gpt-4"
    assert len(result["choices"]) == 1
    assert result["choices"][0]["index"] == 0
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["choices"][0]["message"]["content"] == "Hello"

    # Validate result conforms to ChatCompletion schema
    completion = _validate_chat_completion_dict(result)
    assert completion.id == "chatcmpl-123"
    assert completion.model == "gpt-4"


def test_merge_openai_chunk_accumulates_content():
    """Test that merge_streaming_chunk accumulates content across chunks."""
    provider = _get_provider()

    chunk1 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }
        ],
    )

    chunk2 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {"content": " world"},
                "finish_reason": None,
            }
        ],
    )

    chunk3 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {"content": "!"},
                "finish_reason": "stop",
            }
        ],
    )

    data1 = _chat_completion_chunk_to_dict(chunk1)
    result = provider.merge_streaming_chunk({}, data1)

    data2 = _chat_completion_chunk_to_dict(chunk2)
    result = provider.merge_streaming_chunk(result, data2)

    data3 = _chat_completion_chunk_to_dict(chunk3)
    result = provider.merge_streaming_chunk(result, data3)

    assert result["choices"][0]["message"]["content"] == "Hello world!"
    assert result["choices"][0]["finish_reason"] == "stop"

    # Validate result conforms to ChatCompletion schema
    completion = _validate_chat_completion_dict(result)
    assert completion.choices[0].message.content == "Hello world!"
    assert completion.choices[0].finish_reason == "stop"


def test_merge_openai_chunk_role():
    """Test that merge_streaming_chunk sets role correctly."""
    provider = _get_provider()

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    )

    data = _chat_completion_chunk_to_dict(chunk)
    result = provider.merge_streaming_chunk({}, data)

    assert result["choices"][0]["message"]["role"] == "assistant"

    # Validate result conforms to ChatCompletion schema
    completion = _validate_chat_completion_dict(result)
    assert completion.choices[0].message.role == "assistant"


def test_merge_openai_chunk_reasoning_content():
    """Test that merge_streaming_chunk accumulates reasoning_content."""
    provider = _get_provider()

    chunk1 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {
                    "content": "Hello",
                    "reasoning_content": "Let me think",
                },
                "finish_reason": None,
            }
        ],
    )

    chunk2 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {
                    "content": " world",
                    "reasoning_content": " about this",
                },
                "finish_reason": None,
            }
        ],
    )

    data1 = _chat_completion_chunk_to_dict(chunk1)
    result = provider.merge_streaming_chunk({}, data1)

    data2 = _chat_completion_chunk_to_dict(chunk2)
    result = provider.merge_streaming_chunk(result, data2)

    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert (
        result["choices"][0]["message"]["reasoning_content"]
        == "Let me think about this"
    )

    # Validate result conforms to ChatCompletion schema
    # Note: reasoning_content is not part of standard ChatCompletionMessage,
    # so it's stored as extra data
    completion = _validate_chat_completion_dict(result)
    assert completion.choices[0].message.content == "Hello world"


def test_merge_openai_chunk_reasoning_content_none():
    """Test that merge_streaming_chunk handles None reasoning_content correctly.

    Regression test for bug where reasoning_content field present but set to None
    would cause errors when trying to concatenate with existing reasoning_content.
    """
    provider = _get_provider()

    # Chunk with reasoning_content explicitly set to None
    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="o1-preview",  # Reasoning model
        choices=[
            {
                "index": 0,
                "delta": {
                    "content": "Hello",
                    "reasoning_content": None,
                },
                "finish_reason": None,
            }
        ],
    )

    data = _chat_completion_chunk_to_dict(chunk)
    # This should not raise an error
    result = provider.merge_streaming_chunk({}, data)

    # Verify content is accumulated
    assert result["choices"][0]["message"]["content"] == "Hello"
    # reasoning_content should not be set when None values are received
    assert "reasoning_content" not in result["choices"][0]["message"]


def test_merge_openai_chunk_reasoning_content_mixed_none():
    """Test that merge_streaming_chunk handles mixed None and actual reasoning_content."""
    provider = _get_provider()

    # First chunk with actual reasoning_content
    chunk1 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="o1-preview",
        choices=[
            {
                "index": 0,
                "delta": {
                    "content": "Hello",
                    "reasoning_content": "Let me think",
                },
                "finish_reason": None,
            }
        ],
    )

    # Second chunk with reasoning_content set to None
    chunk2 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="o1-preview",
        choices=[
            {
                "index": 0,
                "delta": {
                    "content": " world",
                    "reasoning_content": None,
                },
                "finish_reason": None,
            }
        ],
    )

    data1 = _chat_completion_chunk_to_dict(chunk1)
    result = provider.merge_streaming_chunk({}, data1)

    data2 = _chat_completion_chunk_to_dict(chunk2)
    # This should not raise an error
    result = provider.merge_streaming_chunk(result, data2)

    # Verify content is accumulated
    assert result["choices"][0]["message"]["content"] == "Hello world"
    # reasoning_content from first chunk should be preserved
    assert result["choices"][0]["message"]["reasoning_content"] == "Let me think"


def test_merge_openai_chunk_tool_calls():
    """Test that merge_streaming_chunk accumulates tool_calls correctly."""
    provider = _get_provider()

    chunk1 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    )

    chunk2 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '":"hello"}'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    )

    chunk3 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_def456",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": "2+2"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
    )

    data1 = _chat_completion_chunk_to_dict(chunk1)
    result = provider.merge_streaming_chunk({}, data1)

    data2 = _chat_completion_chunk_to_dict(chunk2)
    result = provider.merge_streaming_chunk(result, data2)

    data3 = _chat_completion_chunk_to_dict(chunk3)
    result = provider.merge_streaming_chunk(result, data3)

    assert len(result["choices"][0]["message"]["tool_calls"]) == 2

    # First tool call
    tool_call_0 = result["choices"][0]["message"]["tool_calls"][0]
    assert tool_call_0["id"] == "call_abc123"
    assert tool_call_0["type"] == "function"
    assert tool_call_0["function"]["name"] == "search"
    assert tool_call_0["function"]["arguments"] == '{"query":"hello"}'

    # Second tool call
    tool_call_1 = result["choices"][0]["message"]["tool_calls"][1]
    assert tool_call_1["id"] == "call_def456"
    assert tool_call_1["type"] == "function"
    assert tool_call_1["function"]["name"] == "calculate"
    assert tool_call_1["function"]["arguments"] == "2+2"

    # Validate result conforms to ChatCompletion schema
    completion = _validate_chat_completion_dict(result)
    assert len(completion.choices[0].message.tool_calls) == 2
    assert completion.choices[0].message.tool_calls[0].id == "call_abc123"
    assert completion.choices[0].message.tool_calls[1].id == "call_def456"


def test_merge_openai_chunk_multiple_choices():
    """Test that merge_streaming_chunk handles multiple choices correctly."""
    provider = _get_provider()

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {"index": 0, "delta": {"content": "Choice 1"}, "finish_reason": None},
            {"index": 1, "delta": {"content": "Choice 2"}, "finish_reason": None},
        ],
    )

    data = _chat_completion_chunk_to_dict(chunk)
    result = provider.merge_streaming_chunk({}, data)

    assert len(result["choices"]) == 2
    assert result["choices"][0]["index"] == 0
    assert result["choices"][0]["message"]["content"] == "Choice 1"
    assert result["choices"][1]["index"] == 1
    assert result["choices"][1]["message"]["content"] == "Choice 2"

    # Validate result conforms to ChatCompletion schema
    completion = _validate_chat_completion_dict(result)
    assert len(completion.choices) == 2
    assert completion.choices[0].message.content == "Choice 1"
    assert completion.choices[1].message.content == "Choice 2"


def test_merge_openai_chunk_usage():
    """Test that merge_streaming_chunk stores usage data."""
    provider = _get_provider()

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    )

    data = _chat_completion_chunk_to_dict(chunk)
    result = provider.merge_streaming_chunk({}, data)

    assert "usage" in result
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 2

    # Validate result conforms to ChatCompletion schema
    completion = _validate_chat_completion_dict(result)
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 5
    assert completion.usage.total_tokens == 15


def test_merge_openai_chunk_finish_reason():
    """Test that merge_streaming_chunk stores finish_reason correctly."""
    provider = _get_provider()

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": "stop",
            }
        ],
    )

    data = _chat_completion_chunk_to_dict(chunk)
    result = provider.merge_streaming_chunk({}, data)

    assert result["choices"][0]["finish_reason"] == "stop"

    # Validate result conforms to ChatCompletion schema
    completion = _validate_chat_completion_dict(result)
    assert completion.choices[0].finish_reason == "stop"


def test_merge_openai_chunk_empty_delta():
    """Test that merge_streaming_chunk handles empty delta correctly."""
    provider = _get_provider()

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {},
                "finish_reason": None,
            }
        ],
    )

    data = _chat_completion_chunk_to_dict(chunk)
    result = provider.merge_streaming_chunk({}, data)

    assert result["choices"][0]["message"].get("content") == ""

    # Validate result conforms to ChatCompletion schema
    completion = _validate_chat_completion_dict(result)
    assert completion.choices[0].message.content == ""


def test_merge_openai_chunk_out_of_order_choices():
    """Test that merge_streaming_chunk handles choices arriving out of order."""
    provider = _get_provider()

    # First chunk with choice index 1
    chunk1 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 1,
                "delta": {"content": "Choice 2"},
                "finish_reason": None,
            }
        ],
    )

    # Second chunk with choice index 0
    chunk2 = ChatCompletionChunk(
        id="chatcmpl-123",
        object="chat.completion.chunk",
        created=1234567890,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "delta": {"content": "Choice 1"},
                "finish_reason": None,
            }
        ],
    )

    data1 = _chat_completion_chunk_to_dict(chunk1)
    result = provider.merge_streaming_chunk({}, data1)

    data2 = _chat_completion_chunk_to_dict(chunk2)
    result = provider.merge_streaming_chunk(result, data2)

    assert len(result["choices"]) == 2
    assert result["choices"][0]["index"] == 0
    assert result["choices"][0]["message"]["content"] == "Choice 1"
    assert result["choices"][1]["index"] == 1
    assert result["choices"][1]["message"]["content"] == "Choice 2"

    # Validate result conforms to ChatCompletion schema
    completion = _validate_chat_completion_dict(result)
    assert len(completion.choices) == 2
    assert completion.choices[0].message.content == "Choice 1"
    assert completion.choices[1].message.content == "Choice 2"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
