import json
import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("UPSTREAM_BASE_URL", "http://test")
os.environ.setdefault("UPSTREAM_API_KEY", "test")

from routstr.payment.models import Architecture, Model, Pricing  # noqa: E402
from routstr.proxy import (  # noqa: E402
    normalize_request_body_for_upstream,
    parse_request_body_json,
)
from routstr.upstream.base import BaseUpstreamProvider  # noqa: E402


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


def test_parse_request_body_json_accepts_integer_max_completion_tokens() -> None:
    parsed = parse_request_body_json(
        json.dumps(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_completion_tokens": 64,
            }
        ).encode(),
        "v1/chat/completions",
    )

    assert parsed["max_completion_tokens"] == 64


def test_parse_request_body_json_rejects_non_integer_max_completion_tokens() -> None:
    with pytest.raises(HTTPException) as exc_info:
        parse_request_body_json(
            json.dumps(
                {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_completion_tokens": "64",
                }
            ).encode(),
            "v1/chat/completions",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": "max_completion_tokens must be an integer"
    }


def test_parse_request_body_json_rejects_conflicting_token_limits() -> None:
    with pytest.raises(HTTPException) as exc_info:
        parse_request_body_json(
            json.dumps(
                {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 128,
                    "max_completion_tokens": 64,
                }
            ).encode(),
            "v1/chat/completions",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": "max_tokens and max_completion_tokens must match when both are provided"
    }


def test_normalize_request_body_for_upstream_rewrites_openai_limits_only() -> None:
    model = create_test_model("gpt-4o")
    upstream = BaseUpstreamProvider(base_url="https://api.openai.com/v1", api_key="")

    normalized = normalize_request_body_for_upstream(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
        },
        upstream,
        model,
        "v1/chat/completions",
    )

    assert normalized["max_completion_tokens"] == 128
    assert "max_tokens" not in normalized
