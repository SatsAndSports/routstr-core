import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

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
    assert data["web_search_options"] == {}
    assert "tools" not in data


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
