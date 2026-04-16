import json
import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("UPSTREAM_BASE_URL", "http://test")
os.environ.setdefault("UPSTREAM_API_KEY", "test")

from routstr.proxy import parse_request_body_json  # noqa: E402


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
