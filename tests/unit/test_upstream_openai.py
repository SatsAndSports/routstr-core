import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from routstr.payment.models import Architecture, Model, Pricing
from routstr.upstream.openai import OpenAIUpstreamProvider


def create_test_model(model_id: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        created=0,
        description="test",
        context_length=8192,
        architecture=Architecture(
            modality="text",
            input_modalities=["text"],
            output_modalities=["text"],
            tokenizer="test",
            instruct_type=None,
        ),
        pricing=Pricing(
            prompt=0.001,
            completion=0.001,
            request=0.0,
            image=0.0,
            web_search=0.0,
            internal_reasoning=0.0,
        ),
    )


def decode_body(body: bytes | None) -> dict[str, object]:
    assert body is not None
    return json.loads(body.decode())


def create_openai_reasoning_function_request(stream: bool = False) -> dict[str, object]:
    return {
        "model": "openai-gpt-54",
        "max_tokens": 128,
        "reasoning_effort": "medium",
        "messages": [{"role": "user", "content": "Call the weather tool"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "stream": stream,
        "stream_options": {"include_usage": True},
    }


def test_prepare_request_body_rewrites_openai_web_search_chat_request() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "openai-gpt-54",
                "messages": [{"role": "user", "content": "Latest sports news"}],
                "stream": True,
                "tools": [{"type": "web_search"}],
            }
        ).encode(),
        model,
        "chat/completions",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-5-search-api"
    assert data["messages"] == [{"role": "user", "content": "Latest sports news"}]
    assert data["stream"] is True
    assert data["stream_options"] == {"include_usage": True}
    assert data["web_search_options"] == {}
    assert "tools" not in data


def test_translate_chat_completions_to_responses_request_maps_reasoning_and_function_tools() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    translated = provider.translate_chat_completions_to_responses_request(
        create_openai_reasoning_function_request(), model
    )

    assert translated["model"] == "gpt-5.4"
    assert translated["stream"] is False
    assert translated["max_output_tokens"] == 128
    assert translated["reasoning"] == {"effort": "medium"}
    assert translated["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert translated["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather by city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": False,
        }
    ]
    assert translated["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Call the weather tool"}],
        }
    ]


def test_translate_chat_completions_to_responses_request_maps_prior_tool_turns() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    translated = provider.translate_chat_completions_to_responses_request(
        {
            **create_openai_reasoning_function_request(),
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather_1",
                    "content": '{"temperature":"20C"}',
                },
            ],
        },
        model,
    )

    assert translated["input"] == [
        {
            "type": "function_call",
            "call_id": "call_weather_1",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_weather_1",
            "output": '{"temperature":"20C"}',
        },
    ]


def test_prepare_request_body_rewrites_max_tokens_for_openai_chat_completions() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 128,
            }
        ).encode(),
        model,
        "chat/completions",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-5.4"
    assert data["max_completion_tokens"] == 128
    assert "max_tokens" not in data


def test_convert_responses_to_chat_completion_json_maps_function_calls() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")

    chat_response = provider.convert_responses_to_chat_completion_json(
        {
            "id": "resp_123",
            "created_at": 123456,
            "model": "openai-gpt-54",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_weather_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                    "status": "completed",
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "cost_sats": 7,
                "remaining_balance_msats": 9000,
            },
            "metadata": {"routstr": {"cost": {"total_msats": 7000}}},
            "cost": {"total_msats": 7000},
        }
    )

    assert chat_response == {
        "id": "resp_123",
        "object": "chat.completion",
        "created": 123456,
        "model": "openai-gpt-54",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost_sats": 7,
            "remaining_balance_msats": 9000,
        },
        "metadata": {"routstr": {"cost": {"total_msats": 7000}}},
        "cost": {"total_msats": 7000},
    }


def test_prepare_request_body_prefers_existing_max_completion_tokens() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 256,
                "max_completion_tokens": 64,
            }
        ).encode(),
        model,
        "chat/completions",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-5.4"
    assert data["max_completion_tokens"] == 64
    assert "max_tokens" not in data


def test_prepare_request_body_maps_web_search_options_for_chat_completions() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "What is the weather?"}],
                "tools": [
                    {
                        "type": "web_search",
                        "search_context_size": "high",
                        "user_location": {
                            "type": "approximate",
                            "country": "GB",
                            "city": "London",
                            "region": "London",
                            "timezone": "Europe/London",
                        },
                    }
                ],
            }
        ).encode(),
        model,
        "v1/chat/completions",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-5-search-api"
    assert data["web_search_options"] == {
        "search_context_size": "high",
        "user_location": {
            "type": "approximate",
            "approximate": {
                "country": "GB",
                "city": "London",
                "region": "London",
                "timezone": "Europe/London",
            },
        },
    }
    assert "tools" not in data


def test_prepare_request_body_adds_stream_usage_for_openai_chat_streams() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Stream normally"}],
                "stream": True,
            }
        ).encode(),
        model,
        "chat/completions",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-5.4"
    assert data["stream_options"] == {"include_usage": True}


def test_prepare_request_body_preserves_existing_stream_usage_choice() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Stream normally"}],
                "stream": True,
                "stream_options": {"include_usage": False, "foo": "bar"},
            }
        ).encode(),
        model,
        "chat/completions",
    )

    data = decode_body(transformed)
    assert data["stream_options"] == {"include_usage": False, "foo": "bar"}


def test_prepare_request_body_rejects_mixed_web_search_and_function_tools() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    with pytest.raises(HTTPException) as exc_info:
        provider.prepare_request_body(
            json.dumps(
                {
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "Search and call a tool"}],
                    "tools": [
                        {"type": "web_search"},
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup_score",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        },
                    ],
                }
            ).encode(),
            model,
            "chat/completions",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": {
            "message": "OpenAI chat/completions compatibility only supports a single web_search tool",
            "type": "invalid_request_error",
        }
    }


def test_prepare_request_body_rejects_unsupported_web_search_fields() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")

    with pytest.raises(HTTPException) as exc_info:
        provider.prepare_request_body(
            json.dumps(
                {
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "Search with filters"}],
                    "tools": [
                        {
                            "type": "web_search",
                            "filters": {"allowed_domains": ["example.com"]},
                        }
                    ],
                }
            ).encode(),
            model,
            "chat/completions",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": {
            "message": "OpenAI chat/completions compatibility does not support web_search fields: filters",
            "type": "invalid_request_error",
        }
    }


@pytest.mark.asyncio
async def test_forward_request_returns_local_400_for_unsupported_web_search_shape() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")
    request = Mock()
    request.method = "POST"
    request.query_params = {}
    request.state = Mock(request_id="req-123")
    key = Mock()
    key.hashed_key = "deadbeefcafebabe"
    key.balance = 5000

    response = await provider.forward_request(
        request=request,
        path="v1/chat/completions",
        headers={},
        request_body=json.dumps(
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "Search and call a tool"}],
                "tools": [
                    {"type": "web_search"},
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_score",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ],
            }
        ).encode(),
        key=key,
        max_cost_for_model=1000,
        session=AsyncMock(),
        model_obj=model,
    )

    assert response.status_code == 400
    assert json.loads(response.body.decode()) == {
        "error": {
            "message": "OpenAI chat/completions compatibility only supports a single web_search tool",
            "type": "invalid_request_error",
            "code": 400,
        },
        "request_id": "req-123",
    }


@pytest.mark.asyncio
async def test_forward_request_bridges_openai_reasoning_function_tools_to_responses() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")
    request = Mock()
    request.method = "POST"
    request.query_params = {}
    request.state = Mock(request_id="req-bridge")
    key = Mock()
    key.hashed_key = "deadbeefcafebabe"
    key.balance = 5000

    bridged_response = Response(
        content=json.dumps(
            {
                "id": "resp_123",
                "model": "openai-gpt-54",
                "created_at": 123456,
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_weather_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                        "status": "completed",
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                },
                "metadata": {"routstr": {"cost": {"total_msats": 7000}}},
                "cost": {"total_msats": 7000},
            }
        ).encode(),
        media_type="application/json",
    )
    provider.forward_responses_request = AsyncMock(return_value=bridged_response)

    response = await provider.forward_request(
        request=request,
        path="v1/chat/completions",
        headers={},
        request_body=json.dumps(create_openai_reasoning_function_request()).encode(),
        key=key,
        max_cost_for_model=1000,
        session=AsyncMock(),
        model_obj=model,
    )

    provider.forward_responses_request.assert_awaited_once()
    forwarded_path = provider.forward_responses_request.await_args.args[1]
    forwarded_body = json.loads(provider.forward_responses_request.await_args.args[3].decode())
    assert forwarded_path == "v1/responses"
    assert forwarded_body["reasoning"] == {"effort": "medium"}
    assert forwarded_body["max_output_tokens"] == 128
    assert response.status_code == 200
    body = json.loads(response.body.decode())
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_forward_request_bridges_openai_reasoning_function_tools_to_synthetic_stream() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("gpt-5.4")
    request = Mock()
    request.method = "POST"
    request.query_params = {}
    request.state = Mock(request_id="req-stream-bridge")
    key = Mock()
    key.hashed_key = "deadbeefcafebabe"
    key.balance = 5000

    bridged_response = Response(
        content=json.dumps(
            {
                "id": "resp_456",
                "model": "openai-gpt-54",
                "created_at": 123457,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Bridge works"}],
                        "status": "completed",
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                },
            }
        ).encode(),
        media_type="application/json",
    )
    provider.forward_responses_request = AsyncMock(return_value=bridged_response)

    response = await provider.forward_request(
        request=request,
        path="v1/chat/completions",
        headers={},
        request_body=json.dumps(
            create_openai_reasoning_function_request(stream=True)
        ).encode(),
        key=key,
        max_cost_for_model=1000,
        session=AsyncMock(),
        model_obj=model,
    )

    assert isinstance(response, StreamingResponse)

    stream_output = b""
    async for chunk in response.body_iterator:
        stream_output += chunk

    stream_text = stream_output.decode()
    assert "chat.completion.chunk" in stream_text
    assert "Bridge works" in stream_text
    assert '"usage"' in stream_text
    assert "[DONE]" in stream_text
