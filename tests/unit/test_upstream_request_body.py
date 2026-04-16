import json
import os

os.environ.setdefault("UPSTREAM_BASE_URL", "http://test")
os.environ.setdefault("UPSTREAM_API_KEY", "test")

from routstr.payment.models import Architecture, Model, Pricing  # noqa: E402
from routstr.upstream.azure import AzureUpstreamProvider  # noqa: E402
from routstr.upstream.base import BaseUpstreamProvider  # noqa: E402
from routstr.upstream.generic import GenericUpstreamProvider  # noqa: E402
from routstr.upstream.openai import OpenAIUpstreamProvider  # noqa: E402


def create_test_model(model_id: str, canonical_slug: str | None = None) -> Model:
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
        canonical_slug=canonical_slug,
    )


def decode_body(body: bytes | None) -> dict[str, object]:
    assert body is not None
    return json.loads(body.decode())


def test_prepare_request_body_translates_max_tokens_for_openai_base_url() -> None:
    provider = BaseUpstreamProvider(base_url="https://api.openai.com/v1", api_key="")
    model = create_test_model("gpt-4o")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 128,
            }
        ).encode(),
        model,
        "chat/completions",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-4o"
    assert data["max_completion_tokens"] == 128
    assert "max_tokens" not in data


def test_prepare_request_body_prefers_existing_max_completion_tokens() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("openai/gpt-4o")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 256,
                "max_completion_tokens": 64,
            }
        ).encode(),
        model,
        "chat/completions",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-4o"
    assert data["max_completion_tokens"] == 64
    assert "max_tokens" not in data


def test_prepare_request_body_translates_max_tokens_for_azure_chat_completions() -> None:
    provider = AzureUpstreamProvider(
        base_url="https://example.openai.azure.com/openai/v1",
        api_key="azure-key",
        api_version="2024-02-15-preview",
    )
    model = create_test_model("azure/gpt-4o", canonical_slug="deploy-gpt4o")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "azure/gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 32,
            }
        ).encode(),
        model,
        "openai/deployments/deploy-gpt4o/chat/completions",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-4o"
    assert data["max_completion_tokens"] == 32
    assert "max_tokens" not in data


def test_prepare_request_body_leaves_max_tokens_for_non_openai_provider() -> None:
    provider = GenericUpstreamProvider(base_url="https://example.com/v1", api_key="")
    model = create_test_model("gpt-4o")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 48,
            }
        ).encode(),
        model,
        "chat/completions",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-4o"
    assert data["max_tokens"] == 48
    assert "max_completion_tokens" not in data


def test_prepare_request_body_only_rewrites_chat_completions() -> None:
    provider = OpenAIUpstreamProvider(api_key="test-key")
    model = create_test_model("openai/gpt-4o")

    transformed = provider.prepare_request_body(
        json.dumps(
            {
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 96,
            }
        ).encode(),
        model,
        "responses",
    )

    data = decode_body(transformed)
    assert data["model"] == "gpt-4o"
    assert data["max_tokens"] == 96
    assert "max_completion_tokens" not in data
