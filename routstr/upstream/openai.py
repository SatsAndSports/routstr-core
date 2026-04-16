from typing import TYPE_CHECKING

from fastapi import HTTPException

from ..core import get_logger
from ..payment.models import Model, async_fetch_openrouter_models
from .base import BaseUpstreamProvider

if TYPE_CHECKING:
    from ..core.db import UpstreamProviderRow


logger = get_logger(__name__)


class OpenAIUpstreamProvider(BaseUpstreamProvider):
    """Upstream provider specifically configured for OpenAI API."""

    provider_type = "openai"
    default_base_url = "https://api.openai.com/v1"
    platform_url = "https://platform.openai.com/api-keys"
    unsupported_web_search_fields = {
        "filters",
        "external_web_access",
        "search_content_types",
    }

    def __init__(self, api_key: str, provider_fee: float = 1.01):
        super().__init__(
            base_url=self.default_base_url, api_key=api_key, provider_fee=provider_fee
        )

    @classmethod
    def from_db_row(
        cls, provider_row: "UpstreamProviderRow"
    ) -> "OpenAIUpstreamProvider":
        return cls(
            api_key=provider_row.api_key,
            provider_fee=provider_row.provider_fee,
        )

    @classmethod
    def get_provider_metadata(cls) -> dict[str, object]:
        return {
            "id": cls.provider_type,
            "name": "OpenAI",
            "default_base_url": cls.default_base_url,
            "fixed_base_url": True,
            "platform_url": cls.platform_url,
        }

    def transform_model_name(self, model_id: str) -> str:
        """Strip 'openai/' prefix for OpenAI API compatibility."""
        return model_id.removeprefix("openai/")

    def get_chat_search_model(self, model_id: str) -> str | None:
        """Map general OpenAI chat models to web-search chat models."""
        if model_id.startswith("gpt-5"):
            return "gpt-5-search-api"
        if model_id.startswith("gpt-4o-mini"):
            return "gpt-4o-mini-search-preview"
        if model_id.startswith("gpt-4o"):
            return "gpt-4o-search-preview"
        return None

    def transform_chat_completions_request(
        self,
        data: dict[str, object],
        path: str | None,
        model_obj: Model,
    ) -> bool:
        clean_path = (path or "").lstrip("/")
        if not clean_path.endswith("chat/completions"):
            return False

        tools = data.get("tools")
        if not isinstance(tools, list):
            return False

        web_search_tools = [
            tool
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") == "web_search"
        ]
        if not web_search_tools:
            return False

        if len(tools) != 1 or len(web_search_tools) != 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI chat/completions compatibility only supports a single web_search tool"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        if any(
            key in data for key in ("tool_choice", "parallel_tool_calls", "functions", "function_call")
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI chat/completions web_search compatibility does not support additional tool control fields"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        if "web_search_options" in data:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI chat/completions compatibility does not support requests that include both tools and web_search_options"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        model = data.get("model")
        if not isinstance(model, str):
            model = self.transform_model_name(model_obj.id)

        search_model = self.get_chat_search_model(model)
        if not search_model:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            f"OpenAI chat/completions web_search compatibility is not available for model '{model}'"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        web_search_tool = web_search_tools[0]
        unsupported_fields = sorted(
            field
            for field in web_search_tool.keys()
            if field in self.unsupported_web_search_fields
        )
        if unsupported_fields:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI chat/completions compatibility does not support web_search fields: "
                            + ", ".join(unsupported_fields)
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        allowed_fields = {"type", "search_context_size", "user_location"}
        extra_fields = sorted(
            field for field in web_search_tool.keys() if field not in allowed_fields
        )
        if extra_fields:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI chat/completions compatibility does not recognize web_search fields: "
                            + ", ".join(extra_fields)
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        web_search_options: dict[str, object] = {}
        search_context_size = web_search_tool.get("search_context_size")
        if search_context_size is not None:
            web_search_options["search_context_size"] = search_context_size

        user_location = web_search_tool.get("user_location")
        if user_location is not None:
            if not isinstance(user_location, dict) or user_location.get("type") != "approximate":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "OpenAI chat/completions compatibility only supports approximate web_search user_location"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            location_fields = {"type", "country", "city", "region", "timezone"}
            extra_location_fields = sorted(
                field for field in user_location.keys() if field not in location_fields
            )
            if extra_location_fields:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "OpenAI chat/completions compatibility does not recognize web_search user_location fields: "
                                + ", ".join(extra_location_fields)
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            approximate = {
                key: value
                for key in ("country", "city", "region", "timezone")
                if (value := user_location.get(key)) is not None
            }
            web_search_options["user_location"] = {
                "type": "approximate",
                "approximate": approximate,
            }

        data["model"] = search_model
        data["web_search_options"] = web_search_options
        data.pop("tools", None)

        logger.debug(
            "Rewrote OpenAI chat web_search request",
            extra={
                "original_model": model,
                "search_model": search_model,
                "has_user_location": "user_location" in web_search_options,
                "search_context_size": web_search_options.get("search_context_size"),
            },
        )

        return True

    async def fetch_models(self) -> list[Model]:
        """Fetch OpenAI models from OpenRouter API filtered by openai source."""
        models_data = await async_fetch_openrouter_models(source_filter="openai")
        return [Model(**model) for model in models_data]  # type: ignore
