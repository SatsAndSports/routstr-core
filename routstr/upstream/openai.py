import json
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..core import get_logger
from ..payment.helpers import create_error_response
from ..payment.models import Model, async_fetch_openrouter_models
from .base import BaseUpstreamProvider

if TYPE_CHECKING:
    from ..core.db import ApiKey, AsyncSession, UpstreamProviderRow


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

        body_updated = False

        if data.get("stream") is True:
            stream_options = data.get("stream_options")
            if stream_options is None:
                data["stream_options"] = {"include_usage": True}
                body_updated = True
            elif isinstance(stream_options, dict) and "include_usage" not in stream_options:
                updated_stream_options = dict(stream_options)
                updated_stream_options["include_usage"] = True
                data["stream_options"] = updated_stream_options
                body_updated = True

        if "max_completion_tokens" in data:
            body_updated = data.pop("max_tokens", None) is not None or body_updated
        elif "max_tokens" in data:
            data["max_completion_tokens"] = data.pop("max_tokens")
            body_updated = True

        tools = data.get("tools")
        if not isinstance(tools, list):
            return body_updated

        web_search_tools = [
            tool
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") == "web_search"
        ]
        if not web_search_tools:
            return body_updated

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

    def should_bridge_chat_completions_to_responses(
        self, data: dict[str, object], path: str | None
    ) -> bool:
        clean_path = (path or "").lstrip("/")
        if not clean_path.endswith("chat/completions"):
            return False

        if "reasoning_effort" not in data:
            return False

        tools = data.get("tools")
        if not isinstance(tools, list):
            return False

        return any(isinstance(tool, dict) and tool.get("type") == "function" for tool in tools)

    def should_bridge_messages_to_responses(
        self, data: dict[str, object], path: str | None
    ) -> bool:
        clean_path = (path or "").lstrip("/")
        return clean_path.endswith("messages")

    def _translate_chat_input_content(self, content: object) -> list[dict[str, object]]:
        if content is None:
            return []

        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]

        if not isinstance(content, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "OpenAI chat-to-responses bridge only supports string or list message content",
                        "type": "invalid_request_error",
                    }
                },
            )

        translated_parts: list[dict[str, object]] = []
        for part in content:
            if not isinstance(part, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": "OpenAI chat-to-responses bridge received an invalid message content part",
                            "type": "invalid_request_error",
                        }
                    },
                )

            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                translated_parts.append(
                    {"type": "input_text", "text": str(part.get("text", ""))}
                )
                continue

            if part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    image_url_value = image_url.get("url")
                    detail = image_url.get("detail")
                else:
                    image_url_value = image_url
                    detail = None

                if not isinstance(image_url_value, str):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": "OpenAI chat-to-responses bridge received an invalid image_url part",
                                "type": "invalid_request_error",
                            }
                        },
                    )

                translated_part: dict[str, object] = {
                    "type": "input_image",
                    "image_url": image_url_value,
                }
                if detail is not None:
                    translated_part["detail"] = detail
                translated_parts.append(translated_part)
                continue

            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI chat-to-responses bridge does not support message content part type "
                            f"'{part_type}'"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        return translated_parts

    def _translate_chat_assistant_content(self, content: object) -> list[dict[str, object]]:
        if content is None:
            return []

        if isinstance(content, str):
            return [{"type": "output_text", "text": content}]

        if not isinstance(content, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "OpenAI chat-to-responses bridge only supports string or list assistant content",
                        "type": "invalid_request_error",
                    }
                },
            )

        translated_parts: list[dict[str, object]] = []
        for part in content:
            if not isinstance(part, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": "OpenAI chat-to-responses bridge received an invalid assistant content part",
                            "type": "invalid_request_error",
                        }
                    },
                )

            part_type = part.get("type")
            if part_type in {"text", "output_text", "input_text"}:
                translated_parts.append(
                    {"type": "output_text", "text": str(part.get("text", ""))}
                )
                continue

            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI chat-to-responses bridge does not support assistant content part type "
                            f"'{part_type}'"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        return translated_parts

    def _stringify_chat_tool_output(self, content: object) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}:
                    text_parts.append(str(part.get("text", "")))
                else:
                    return json.dumps(content)
            return "".join(text_parts)

        return json.dumps(content)

    def _translate_chat_messages_to_responses_input(
        self, messages: object
    ) -> list[dict[str, object]]:
        if not isinstance(messages, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "OpenAI chat-to-responses bridge requires a messages array",
                        "type": "invalid_request_error",
                    }
                },
            )

        translated_input: list[dict[str, object]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": "OpenAI chat-to-responses bridge received an invalid message entry",
                            "type": "invalid_request_error",
                        }
                    },
                )

            role = message.get("role")
            if role == "tool":
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": "OpenAI chat-to-responses bridge requires tool messages to include tool_call_id",
                                "type": "invalid_request_error",
                            }
                        },
                    )

                translated_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": self._stringify_chat_tool_output(message.get("content")),
                    }
                )
                continue

            if role == "assistant":
                assistant_parts = self._translate_chat_assistant_content(
                    message.get("content")
                )
                if assistant_parts:
                    translated_input.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": assistant_parts,
                        }
                    )

                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict) or tool_call.get("type") != "function":
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": {
                                        "message": "OpenAI chat-to-responses bridge only supports assistant function tool calls",
                                        "type": "invalid_request_error",
                                    }
                                },
                            )

                        function = tool_call.get("function")
                        call_id = tool_call.get("id")
                        if not isinstance(function, dict) or not isinstance(call_id, str):
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": {
                                        "message": "OpenAI chat-to-responses bridge received an invalid assistant tool call",
                                        "type": "invalid_request_error",
                                    }
                                },
                            )

                        name = function.get("name")
                        arguments = function.get("arguments", "{}")
                        if not isinstance(name, str) or not isinstance(arguments, str):
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": {
                                        "message": "OpenAI chat-to-responses bridge requires assistant tool calls to include function name and string arguments",
                                        "type": "invalid_request_error",
                                    }
                                },
                            )

                        translated_input.append(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": arguments,
                                "status": "completed",
                            }
                        )
                continue

            if role in {"user", "system", "developer"}:
                translated_input.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": self._translate_chat_input_content(message.get("content")),
                    }
                )
                continue

            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI chat-to-responses bridge does not support message role "
                            f"'{role}'"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        return translated_input

    def _translate_chat_tools_to_responses_tools(
        self, tools: object
    ) -> list[dict[str, object]]:
        if not isinstance(tools, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "OpenAI chat-to-responses bridge requires a tools array",
                        "type": "invalid_request_error",
                    }
                },
            )

        translated_tools: list[dict[str, object]] = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": "OpenAI chat-to-responses bridge only supports function tools",
                            "type": "invalid_request_error",
                        }
                    },
                )

            function = tool.get("function")
            if not isinstance(function, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": "OpenAI chat-to-responses bridge received an invalid function tool definition",
                            "type": "invalid_request_error",
                        }
                    },
                )

            name = function.get("name")
            if not isinstance(name, str):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": "OpenAI chat-to-responses bridge requires function tools to include a name",
                            "type": "invalid_request_error",
                        }
                    },
                )

            translated_tool: dict[str, object] = {
                "type": "function",
                "name": name,
                "parameters": function.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
                "strict": function.get("strict", False),
            }
            if description := function.get("description"):
                translated_tool["description"] = description
            translated_tools.append(translated_tool)

        return translated_tools

    def _translate_chat_tool_choice_to_responses(self, tool_choice: object) -> object:
        if isinstance(tool_choice, str):
            if tool_choice in {"auto", "required", "none"}:
                return tool_choice
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI chat-to-responses bridge does not support tool_choice "
                            f"'{tool_choice}'"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function = tool_choice.get("function")
                if isinstance(function, dict) and isinstance(function.get("name"), str):
                    return {"type": "function", "name": function["name"]}
                if isinstance(tool_choice.get("name"), str):
                    return {"type": "function", "name": tool_choice["name"]}

            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "OpenAI chat-to-responses bridge received an unsupported tool_choice object",
                        "type": "invalid_request_error",
                    }
                },
            )

        return tool_choice

    def translate_chat_completions_to_responses_request(
        self, data: dict[str, object], model_obj: Model
    ) -> dict[str, object]:
        model = self.transform_model_name(model_obj.id)

        translated: dict[str, object] = {
            "model": model,
            "input": self._translate_chat_messages_to_responses_input(
                data.get("messages")
            ),
            "stream": False,
            "tools": self._translate_chat_tools_to_responses_tools(data.get("tools")),
            "reasoning": {"effort": data["reasoning_effort"]},
        }

        if "tool_choice" in data:
            translated["tool_choice"] = self._translate_chat_tool_choice_to_responses(
                data.get("tool_choice")
            )

        if "parallel_tool_calls" in data:
            translated["parallel_tool_calls"] = data["parallel_tool_calls"]

        max_output_tokens = data.get("max_completion_tokens", data.get("max_tokens"))
        if max_output_tokens is not None:
            translated["max_output_tokens"] = max_output_tokens

        passthrough_fields = (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "metadata",
            "service_tier",
            "store",
            "user",
        )
        for field in passthrough_fields:
            if field in data:
                translated[field] = data[field]

        return translated

    def _responses_usage_to_chat_usage(self, usage: object) -> dict[str, object] | None:
        if not isinstance(usage, dict):
            return None

        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)
        total_tokens = int(
            usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        )

        chat_usage: dict[str, object] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, dict):
            chat_usage["prompt_tokens_details"] = dict(input_details)

        output_details = usage.get("output_tokens_details")
        if isinstance(output_details, dict):
            chat_usage["completion_tokens_details"] = dict(output_details)

        for key in ("cost", "cost_sats", "remaining_balance_msats"):
            if key in usage:
                chat_usage[key] = usage[key]

        return chat_usage

    def convert_responses_to_chat_completion_json(
        self, response_json: dict[str, object]
    ) -> dict[str, object]:
        output = response_json.get("output")
        assistant_text_parts: list[str] = []
        tool_calls: list[dict[str, object]] = []

        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "message" and item.get("role") == "assistant":
                    content = item.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if not isinstance(part, dict):
                                continue
                            part_type = part.get("type")
                            if part_type == "output_text":
                                assistant_text_parts.append(str(part.get("text", "")))
                            elif part_type == "refusal":
                                assistant_text_parts.append(str(part.get("refusal", "")))
                elif item_type == "function_call":
                    name = item.get("name")
                    arguments = item.get("arguments")
                    call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
                    if isinstance(name, str) and isinstance(arguments, str):
                        tool_calls.append(
                            {
                                "id": str(call_id),
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        )

        message: dict[str, object] = {
            "role": "assistant",
            "content": "".join(assistant_text_parts) if assistant_text_parts else None,
        }
        finish_reason = "stop"
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        chat_response: dict[str, object] = {
            "id": response_json.get("id") or f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(response_json.get("created_at") or time.time()),
            "model": response_json.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }

        if usage := self._responses_usage_to_chat_usage(response_json.get("usage")):
            chat_response["usage"] = usage

        if metadata := response_json.get("metadata"):
            chat_response["metadata"] = metadata

        if cost := response_json.get("cost"):
            chat_response["cost"] = cost

        return chat_response

    def _stringify_messages_tool_result_content(self, content: object) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                else:
                    return json.dumps(content)
            return "".join(text_parts)

        return json.dumps(content)

    def _parse_response_function_arguments(self, arguments: object) -> dict[str, object]:
        if isinstance(arguments, dict):
            return dict(arguments)

        if not isinstance(arguments, str):
            return {}

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}

        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    def _build_responses_message_item(
        self, role: str, content: list[dict[str, object]]
    ) -> dict[str, object]:
        item: dict[str, object] = {
            "type": "message",
            "role": role,
            "content": content,
        }
        if role == "assistant":
            item["status"] = "completed"
        return item

    def _translate_messages_content_to_responses_items(
        self, role: str, content: object
    ) -> list[dict[str, object]]:
        if content is None:
            return []

        if isinstance(content, str):
            part_type = "output_text" if role == "assistant" else "input_text"
            return [
                self._build_responses_message_item(
                    role,
                    [{"type": part_type, "text": content}],
                )
            ]

        if not isinstance(content, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI messages-to-responses bridge only supports string or list message content"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        translated_items: list[dict[str, object]] = []
        current_parts: list[dict[str, object]] = []

        def flush_parts() -> None:
            nonlocal current_parts
            if current_parts:
                translated_items.append(
                    self._build_responses_message_item(role, current_parts)
                )
                current_parts = []

        for block in content:
            if not isinstance(block, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "OpenAI messages-to-responses bridge received an invalid content block"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            block_type = block.get("type")
            if block_type == "text":
                part_type = "output_text" if role == "assistant" else "input_text"
                current_parts.append(
                    {"type": part_type, "text": str(block.get("text", ""))}
                )
                continue

            if block_type == "tool_use":
                if role != "assistant":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": (
                                    "OpenAI messages-to-responses bridge only supports assistant tool_use blocks"
                                ),
                                "type": "invalid_request_error",
                            }
                        },
                    )

                flush_parts()
                call_id = block.get("id")
                name = block.get("name")
                input_data = block.get("input", {})
                if (
                    not isinstance(call_id, str)
                    or not isinstance(name, str)
                    or not isinstance(input_data, dict)
                ):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": (
                                    "OpenAI messages-to-responses bridge requires tool_use blocks to include id, name, and object input"
                                ),
                                "type": "invalid_request_error",
                            }
                        },
                    )

                translated_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": json.dumps(input_data),
                        "status": "completed",
                    }
                )
                continue

            if block_type == "tool_result":
                if role != "user":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": (
                                    "OpenAI messages-to-responses bridge only supports user tool_result blocks"
                                ),
                                "type": "invalid_request_error",
                            }
                        },
                    )

                flush_parts()
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": (
                                    "OpenAI messages-to-responses bridge requires tool_result blocks to include tool_use_id"
                                ),
                                "type": "invalid_request_error",
                            }
                        },
                    )

                translated_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                        "output": self._stringify_messages_tool_result_content(
                            block.get("content")
                        ),
                    }
                )
                continue

            if block_type in {"thinking", "redacted_thinking"} and role == "assistant":
                continue

            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI messages-to-responses bridge does not support content block type "
                            f"'{block_type}'"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        flush_parts()
        return translated_items

    def _translate_messages_to_responses_input(
        self, system: object, messages: object
    ) -> list[dict[str, object]]:
        if not isinstance(messages, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "OpenAI messages-to-responses bridge requires a messages array",
                        "type": "invalid_request_error",
                    }
                },
            )

        translated_input: list[dict[str, object]] = []
        if system is not None:
            translated_input.extend(
                self._translate_messages_content_to_responses_items("system", system)
            )

        for message in messages:
            if not isinstance(message, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "OpenAI messages-to-responses bridge received an invalid message entry"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            role = message.get("role")
            if role not in {"user", "assistant"}:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "OpenAI messages-to-responses bridge does not support message role "
                                f"'{role}'"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            translated_input.extend(
                self._translate_messages_content_to_responses_items(
                    str(role), message.get("content")
                )
            )

        return translated_input

    def _translate_messages_tools_to_responses_tools(
        self, tools: object
    ) -> list[dict[str, object]]:
        if not isinstance(tools, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "OpenAI messages-to-responses bridge requires a tools array",
                        "type": "invalid_request_error",
                    }
                },
            )

        translated_tools: list[dict[str, object]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "OpenAI messages-to-responses bridge received an invalid tool definition"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            tool_type = tool.get("type")
            if tool_type not in (None, "custom"):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "OpenAI messages-to-responses bridge only supports custom function tools"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            name = tool.get("name")
            parameters = tool.get("input_schema", {"type": "object", "properties": {}})
            if not isinstance(name, str) or not isinstance(parameters, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "OpenAI messages-to-responses bridge requires tools to include name and object input_schema"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            translated_tool: dict[str, object] = {
                "type": "function",
                "name": name,
                "parameters": parameters,
            }
            if description := tool.get("description"):
                translated_tool["description"] = description
            translated_tools.append(translated_tool)

        return translated_tools

    def _translate_messages_tool_choice_to_responses(self, tool_choice: object) -> object:
        if isinstance(tool_choice, str):
            mapping = {
                "auto": "auto",
                "any": "required",
                "required": "required",
                "none": "none",
            }
            if tool_choice in mapping:
                return mapping[tool_choice]
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI messages-to-responses bridge does not support tool_choice "
                            f"'{tool_choice}'"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        if isinstance(tool_choice, dict):
            tool_choice_type = tool_choice.get("type")
            if tool_choice_type == "auto":
                return "auto"
            if tool_choice_type == "any":
                return "required"
            if tool_choice_type == "none":
                return "none"
            if tool_choice_type == "tool" and isinstance(tool_choice.get("name"), str):
                return {"type": "function", "name": tool_choice["name"]}

            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI messages-to-responses bridge received an unsupported tool_choice object"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        return tool_choice

    def _apply_messages_output_config(
        self, translated: dict[str, object], output_config: object
    ) -> None:
        if output_config is None:
            return

        if not isinstance(output_config, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI messages-to-responses bridge requires output_config to be an object"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        effort = output_config.get("effort")
        if effort is not None:
            translated["reasoning"] = {"effort": effort}

        format_config = output_config.get("format")
        if format_config is None:
            return

        if not isinstance(format_config, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "OpenAI messages-to-responses bridge requires output_config.format to be an object"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )

        format_type = format_config.get("type")
        if format_type == "json_schema":
            schema = format_config.get("schema")
            if not isinstance(schema, dict):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "OpenAI messages-to-responses bridge requires output_config.format.schema to be an object"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )

            text_format: dict[str, object] = {
                "type": "json_schema",
                "name": str(format_config.get("name") or "anthropic_output"),
                "schema": schema,
            }
            if description := format_config.get("description"):
                text_format["description"] = description
            if "strict" in format_config:
                text_format["strict"] = format_config["strict"]
            translated["text"] = {"format": text_format}
            return

        if format_type == "json_object":
            translated["text"] = {"format": {"type": "json_object"}}
            return

        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "OpenAI messages-to-responses bridge does not support output_config.format type "
                        f"'{format_type}'"
                    ),
                    "type": "invalid_request_error",
                }
            },
        )

    def translate_messages_to_responses_request(
        self, data: dict[str, object], model_obj: Model
    ) -> dict[str, object]:
        translated: dict[str, object] = {
            "model": self.transform_model_name(model_obj.id),
            "input": self._translate_messages_to_responses_input(
                data.get("system"), data.get("messages")
            ),
            "stream": False,
        }

        if "tools" in data:
            translated["tools"] = self._translate_messages_tools_to_responses_tools(
                data.get("tools")
            )

        if "tool_choice" in data:
            translated["tool_choice"] = self._translate_messages_tool_choice_to_responses(
                data.get("tool_choice")
            )

        self._apply_messages_output_config(translated, data.get("output_config"))

        max_output_tokens = data.get("max_tokens")
        if max_output_tokens is not None:
            translated["max_output_tokens"] = max_output_tokens

        passthrough_fields = ("temperature", "top_p", "metadata", "service_tier")
        for field in passthrough_fields:
            if field in data:
                translated[field] = data[field]

        if "stop_sequences" in data:
            translated["stop"] = data["stop_sequences"]

        return translated

    def _responses_usage_to_messages_usage(
        self, usage: object
    ) -> dict[str, object] | None:
        if not isinstance(usage, dict):
            return None

        messages_usage: dict[str, object] = {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        }

        for key in (
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "reasoning_tokens",
            "cost",
            "cost_sats",
            "remaining_balance_msats",
        ):
            if key in usage:
                messages_usage[key] = usage[key]

        return messages_usage

    def convert_responses_to_messages_json(
        self, response_json: dict[str, object]
    ) -> dict[str, object]:
        output = response_json.get("output")
        content_blocks: list[dict[str, object]] = []
        stop_reason = "end_turn"

        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "message" and item.get("role") == "assistant":
                    content = item.get("content")
                    if not isinstance(content, list):
                        continue

                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        part_type = part.get("type")
                        if part_type == "output_text":
                            content_blocks.append(
                                {"type": "text", "text": str(part.get("text", ""))}
                            )
                        elif part_type == "refusal":
                            content_blocks.append(
                                {
                                    "type": "text",
                                    "text": str(part.get("refusal", "")),
                                }
                            )
                elif item_type == "function_call":
                    name = item.get("name")
                    if not isinstance(name, str):
                        continue

                    call_id = item.get("call_id") or item.get("id")
                    if not isinstance(call_id, str):
                        call_id = f"toolu_{uuid.uuid4().hex}"

                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": self._parse_response_function_arguments(
                                item.get("arguments")
                            ),
                        }
                    )
                    stop_reason = "tool_use"

        if stop_reason != "tool_use" and response_json.get("status") == "incomplete":
            incomplete_details = response_json.get("incomplete_details")
            if (
                isinstance(incomplete_details, dict)
                and incomplete_details.get("reason") == "max_output_tokens"
            ):
                stop_reason = "max_tokens"

        message_response: dict[str, object] = {
            "id": response_json.get("id") or f"msg_{uuid.uuid4()}",
            "type": "message",
            "role": "assistant",
            "model": response_json.get("model", "unknown"),
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
        }

        if usage := self._responses_usage_to_messages_usage(response_json.get("usage")):
            message_response["usage"] = usage

        if metadata := response_json.get("metadata"):
            message_response["metadata"] = metadata

        if cost := response_json.get("cost"):
            message_response["cost"] = cost

        return message_response

    def _build_synthetic_messages_stream_headers(
        self, response: Response
    ) -> dict[str, str]:
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower().startswith("access-control-")
            or key.lower() in {"cache-control", "vary"}
        }
        response_headers["content-type"] = "text/event-stream"
        return response_headers

    def _build_synthetic_messages_stream(
        self, message_response: dict[str, object]
    ) -> AsyncGenerator[bytes, None]:
        async def stream() -> AsyncGenerator[bytes, None]:
            message_id = str(message_response.get("id") or f"msg_{uuid.uuid4()}")
            model = str(message_response.get("model", "unknown"))
            usage = message_response.get("usage")
            if not isinstance(usage, dict):
                usage = {}

            start_usage: dict[str, object] = {
                "input_tokens": int(usage.get("input_tokens") or 0)
            }
            for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                if key in usage:
                    start_usage[key] = usage[key]

            message_start = {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": start_usage,
                },
            }
            yield (
                f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode()
            )

            content_blocks = message_response.get("content")
            if not isinstance(content_blocks, list):
                content_blocks = []

            for index, block in enumerate(content_blocks):
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")
                if block_type == "text":
                    yield (
                        "event: content_block_start\ndata: "
                        + json.dumps(
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {"type": "text", "text": ""},
                            }
                        )
                        + "\n\n"
                    ).encode()
                    yield (
                        "event: content_block_delta\ndata: "
                        + json.dumps(
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "text_delta",
                                    "text": str(block.get("text", "")),
                                },
                            }
                        )
                        + "\n\n"
                    ).encode()
                    yield (
                        "event: content_block_stop\ndata: "
                        + json.dumps({"type": "content_block_stop", "index": index})
                        + "\n\n"
                    ).encode()
                    continue

                if block_type == "tool_use":
                    input_data = block.get("input")
                    if not isinstance(input_data, dict):
                        input_data = {}

                    yield (
                        "event: content_block_start\ndata: "
                        + json.dumps(
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": block.get("id"),
                                    "name": block.get("name"),
                                    "input": {},
                                },
                            }
                        )
                        + "\n\n"
                    ).encode()

                    if input_data:
                        yield (
                            "event: content_block_delta\ndata: "
                            + json.dumps(
                                {
                                    "type": "content_block_delta",
                                    "index": index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": json.dumps(
                                            input_data, separators=(",", ":")
                                        ),
                                    },
                                }
                            )
                            + "\n\n"
                        ).encode()

                    yield (
                        "event: content_block_stop\ndata: "
                        + json.dumps({"type": "content_block_stop", "index": index})
                        + "\n\n"
                    ).encode()

            message_delta = {
                "type": "message_delta",
                "delta": {
                    "stop_reason": message_response.get("stop_reason", "end_turn"),
                    "stop_sequence": message_response.get("stop_sequence"),
                },
                "usage": {"output_tokens": int(usage.get("output_tokens") or 0)},
            }
            yield (
                f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode()
            )
            yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        return stream()

    def _build_synthetic_chat_stream_headers(self, response: Response) -> dict[str, str]:
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower().startswith("access-control-") or key.lower() in {"cache-control", "vary"}
        }
        response_headers["content-type"] = "text/event-stream"
        return response_headers

    def _should_include_usage_chunk(self, request_data: dict[str, object]) -> bool:
        stream_options = request_data.get("stream_options")
        if not isinstance(stream_options, dict):
            return True
        return bool(stream_options.get("include_usage", False))

    def _build_synthetic_chat_stream(
        self,
        chat_response: dict[str, object],
        include_usage_chunk: bool,
    ) -> AsyncGenerator[bytes, None]:
        async def stream() -> AsyncGenerator[bytes, None]:
            chunk_id = str(chat_response.get("id") or f"chatcmpl-{uuid.uuid4()}")
            created = int(chat_response.get("created") or time.time())
            model = str(chat_response.get("model", "unknown"))
            choice = chat_response.get("choices", [{}])[0]
            if not isinstance(choice, dict):
                choice = {}
            message = choice.get("message", {})
            if not isinstance(message, dict):
                message = {}

            role_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n".encode()

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                delta_tool_calls = []
                for index, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function", {})
                    if not isinstance(function, dict):
                        function = {}
                    delta_tool_calls.append(
                        {
                            "index": index,
                            "id": tool_call.get("id"),
                            "type": "function",
                            "function": {
                                "name": function.get("name"),
                                "arguments": function.get("arguments", ""),
                            },
                        }
                    )

                tool_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": delta_tool_calls},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(tool_chunk)}\n\n".encode()
            else:
                content = message.get("content")
                if isinstance(content, str) and content:
                    content_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n".encode()

            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": choice.get("finish_reason", "stop"),
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n".encode()

            if include_usage_chunk and isinstance(chat_response.get("usage"), dict):
                usage_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": chat_response["usage"],
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n".encode()

            yield b"data: [DONE]\n\n"

        return stream()

    async def forward_request(
        self,
        request: Request,
        path: str,
        headers: dict,
        request_body: bytes | None,
        key: "ApiKey",
        max_cost_for_model: int,
        session: "AsyncSession",
        model_obj: Model,
    ) -> Response | StreamingResponse:
        if not request_body:
            return await super().forward_request(
                request,
                path,
                headers,
                request_body,
                key,
                max_cost_for_model,
                session,
                model_obj,
            )

        try:
            request_data = json.loads(request_body)
        except json.JSONDecodeError:
            return await super().forward_request(
                request,
                path,
                headers,
                request_body,
                key,
                max_cost_for_model,
                session,
                model_obj,
            )

        if not isinstance(request_data, dict):
            return await super().forward_request(
                request,
                path,
                headers,
                request_body,
                key,
                max_cost_for_model,
                session,
                model_obj,
            )

        if self.should_bridge_messages_to_responses(request_data, path):
            try:
                bridged_request = self.translate_messages_to_responses_request(
                    request_data, model_obj
                )
            except HTTPException as exc:
                error_type, message = self.get_http_exception_error(exc)
                logger.warning(
                    "Rejected OpenAI messages-to-responses bridge request",
                    extra={
                        "path": path,
                        "status_code": exc.status_code,
                        "error_type": error_type,
                        "error_message": message,
                    },
                )
                return create_error_response(
                    error_type,
                    message,
                    exc.status_code,
                    request=request,
                )

            logger.info(
                "Bridging OpenAI messages request to Responses API",
                extra={
                    "path": path,
                    "requested_model": request_data.get("model", "unknown"),
                    "upstream_model": bridged_request.get("model", "unknown"),
                    "client_wants_streaming": bool(request_data.get("stream")),
                    "tool_count": len(request_data.get("tools", []))
                    if isinstance(request_data.get("tools"), list)
                    else 0,
                },
            )

            bridged_response = await self.forward_responses_request(
                request,
                "v1/responses",
                headers,
                json.dumps(bridged_request).encode(),
                key,
                max_cost_for_model,
                session,
                model_obj,
            )

            if bridged_response.status_code != 200:
                return bridged_response

            if isinstance(bridged_response, StreamingResponse):
                return bridged_response

            try:
                bridged_json = json.loads(bridged_response.body)
            except Exception:
                return bridged_response

            if not isinstance(bridged_json, dict):
                return bridged_response

            messages_response = self.convert_responses_to_messages_json(bridged_json)
            if bool(request_data.get("stream")):
                return StreamingResponse(
                    self._build_synthetic_messages_stream(messages_response),
                    status_code=200,
                    headers=self._build_synthetic_messages_stream_headers(
                        bridged_response
                    ),
                )

            response_headers = {
                key: value
                for key, value in bridged_response.headers.items()
                if key.lower() not in {"content-length", "content-encoding"}
            }
            response_headers["content-type"] = "application/json"
            return Response(
                content=json.dumps(messages_response).encode(),
                status_code=200,
                headers=response_headers,
                media_type="application/json",
            )

        if not self.should_bridge_chat_completions_to_responses(request_data, path):
            return await super().forward_request(
                request,
                path,
                headers,
                request_body,
                key,
                max_cost_for_model,
                session,
                model_obj,
            )

        try:
            bridged_request = self.translate_chat_completions_to_responses_request(
                request_data, model_obj
            )
        except HTTPException as exc:
            error_type, message = self.get_http_exception_error(exc)
            logger.warning(
                "Rejected OpenAI chat-to-responses bridge request",
                extra={
                    "path": path,
                    "status_code": exc.status_code,
                    "error_type": error_type,
                    "error_message": message,
                },
            )
            return create_error_response(
                error_type,
                message,
                exc.status_code,
                request=request,
            )

        logger.info(
            "Bridging OpenAI chat/completions request to Responses API",
            extra={
                "path": path,
                "requested_model": request_data.get("model", "unknown"),
                "upstream_model": bridged_request.get("model", "unknown"),
                "client_wants_streaming": bool(request_data.get("stream")),
                "tool_count": len(request_data.get("tools", []))
                if isinstance(request_data.get("tools"), list)
                else 0,
            },
        )

        bridged_response = await self.forward_responses_request(
            request,
            "v1/responses",
            headers,
            json.dumps(bridged_request).encode(),
            key,
            max_cost_for_model,
            session,
            model_obj,
        )

        if bridged_response.status_code != 200:
            return bridged_response

        if isinstance(bridged_response, StreamingResponse):
            return bridged_response

        try:
            bridged_json = json.loads(bridged_response.body)
        except Exception:
            return bridged_response

        if not isinstance(bridged_json, dict):
            return bridged_response

        chat_response = self.convert_responses_to_chat_completion_json(bridged_json)
        client_wants_streaming = bool(request_data.get("stream"))
        if client_wants_streaming:
            return StreamingResponse(
                self._build_synthetic_chat_stream(
                    chat_response,
                    include_usage_chunk=self._should_include_usage_chunk(request_data),
                ),
                status_code=200,
                headers=self._build_synthetic_chat_stream_headers(bridged_response),
            )

        response_headers = {
            key: value
            for key, value in bridged_response.headers.items()
            if key.lower() not in {"content-length", "content-encoding"}
        }
        response_headers["content-type"] = "application/json"
        return Response(
            content=json.dumps(chat_response).encode(),
            status_code=200,
            headers=response_headers,
            media_type="application/json",
        )

    async def fetch_models(self) -> list[Model]:
        """Fetch OpenAI models from OpenRouter API filtered by openai source."""
        models_data = await async_fetch_openrouter_models(source_filter="openai")
        return [Model(**model) for model in models_data]  # type: ignore
