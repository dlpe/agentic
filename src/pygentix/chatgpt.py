"""OpenAI (ChatGPT) LLM backend for agents."""

import base64
import json
import mimetypes
import os
from typing import Any

from .core import Agent, ChatResponse

__all__ = ["ChatGPT"]


# -- OpenAI-format helpers (shared with Copilot) ---------------------------


def _encode_image(path: str) -> tuple[str, str]:
    """Read an image file and return ``(base64_string, mime_type)``."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def _prepare_openai_messages(messages: list[dict]) -> list[dict]:
    """Transform internal message format to the OpenAI Chat Completions format.

    Handles tool-call metadata and multimodal ``images`` that
    :class:`~pygentix.core.Conversation` attaches to messages.
    """
    result: list[dict] = []
    for msg in messages:
        role = msg["role"]

        if role == "user" and msg.get("images"):
            content_parts: list[dict] = [{"type": "text", "text": msg["content"]}]
            for img_path in msg["images"]:
                b64, mime = _encode_image(img_path)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            result.append({"role": "user", "content": content_parts})
        elif role == "assistant" and msg.get("tool_calls"):
            result.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": [
                    {
                        "id": tc.get("id") or f"call_{tc['name']}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in msg["tool_calls"]
                ],
            })
        elif role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id") or f"call_{msg.get('tool_name', '')}",
                "content": msg["content"],
            })
        else:
            result.append({"role": role, "content": msg.get("content", "")})

    return result


def _parse_openai_response(choice: Any) -> ChatResponse:
    """Convert a single OpenAI ``Choice`` into a :class:`ChatResponse`."""
    tool_calls = None
    if choice.message.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            for tc in choice.message.tool_calls
        ]
    return ChatResponse(content=choice.message.content or "", tool_calls=tool_calls)


def _openai_chat(
    client: Any,
    model: str,
    functions: dict,
    messages: list[dict],
    **kwargs: Any,
) -> ChatResponse:
    """Shared chat implementation for OpenAI and Azure OpenAI backends."""
    tools = [f.to_tool_schema() for f in functions.values()] or None

    fmt = kwargs.pop("format", None)
    extra: dict[str, Any] = {}
    if fmt:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": fmt, "strict": False},
        }

    response = client.chat.completions.create(
        model=model,
        messages=_prepare_openai_messages(messages),
        **({"tools": tools} if tools else {}),
        **extra,
    )
    return _parse_openai_response(response.choices[0])


# -- ChatGPT agent --------------------------------------------------------


class ChatGPT(Agent):
    """Agent backed by the `OpenAI <https://platform.openai.com>`_ API.

    Works with any OpenAI chat model (``gpt-4o``, ``gpt-4o-mini``, etc.).

    Parameters
    ----------
    model:
        The model identifier.  Defaults to ``"gpt-4o-mini"``.
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when not provided.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialised OpenAI client."""
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Forward *messages* to the OpenAI API and return a :class:`ChatResponse`."""
        return _openai_chat(self.client, self.model, self.functions, messages, **kwargs)
