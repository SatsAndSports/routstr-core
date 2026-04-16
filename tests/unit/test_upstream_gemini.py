import json
import os
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("UPSTREAM_BASE_URL", "http://test")
os.environ.setdefault("UPSTREAM_API_KEY", "test")

from routstr.payment.models import Architecture, Model, Pricing  # noqa: E402
from routstr.upstream.gemini import GeminiUpstreamProvider  # noqa: E402


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


async def test_forward_request_uses_max_completion_tokens_for_v1_chat_completions() -> None:
    provider = GeminiUpstreamProvider(api_key="test-key")
    provider._client = Mock()
    provider._client.generate_content = AsyncMock(
        return_value={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    )

    model = create_test_model("gemini/gemini-2.0-flash")
    request = Mock()
    request.query_params = {}
    key = Mock()
    key.hashed_key = "test-key-hash"
    key.balance = 9000
    session = AsyncMock()

    with patch(
        "routstr.auth.adjust_payment_for_tokens",
        new=AsyncMock(return_value={"total_msats": 1000}),
    ):
        response = await provider.forward_request(
            request=request,
            path="v1/chat/completions",
            headers={},
            request_body=json.dumps(
                {
                    "model": "gemini/gemini-2.0-flash",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_completion_tokens": 32,
                }
            ).encode(),
            key=key,
            max_cost_for_model=1000,
            session=session,
            model_obj=model,
        )

    provider._client.generate_content.assert_awaited_once()
    assert provider._client.generate_content.await_args.kwargs["max_tokens"] == 32
    assert response.status_code == 200
