"""OpenAI (ChatGPT) LLM backend for agents."""

import base64
import json
import logging
import mimetypes
import os
from typing import Any, Iterator

from .core import Agent, ChatResponse, Usage

__all__ = ["ChatGPT"]

logger = logging.getLogger("pygentix")


# -- OpenAI-format helpers (shared with Copilot) ---------------------------


def encode_image(path: str) -> tuple[str, str]:
    """Read an image file and return ``(base64_string, mime_type)``."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def prepare_openai_messages(messages: list[dict]) -> list[dict]:
    """Transform internal message format to the OpenAI Chat Completions format."""
    result: list[dict] = []
    for msg in messages:
        role = msg["role"]

        if role == "user" and msg.get("images"):
            content_parts: list[dict] = [{"type": "text", "text": msg["content"]}]
            for img_path in msg["images"]:
                b64, mime = encode_image(img_path)
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


def extract_openai_usage(raw_response: Any) -> Usage:
    """Extract token usage from an OpenAI response object."""
    u = getattr(raw_response, "usage", None)
    if u is None:
        return Usage()
    return Usage(
        prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        total_tokens=getattr(u, "total_tokens", 0) or 0,
    )


def parse_openai_response(choice: Any, usage: Usage | None = None) -> ChatResponse:
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
    return ChatResponse(content=choice.message.content or "", tool_calls=tool_calls, usage=usage)


def openai_chat(
    client: Any,
    model: str,
    functions: dict,
    messages: list[dict],
    *,
    temperature: float = 0,
    seed: int | None = 42,
    retry_fn: Any = None,
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

    def do_call() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=prepare_openai_messages(messages),
            temperature=temperature,
            **({"seed": seed} if seed is not None else {}),
            **({"tools": tools} if tools else {}),
            **extra,
        )

    response = retry_fn(do_call) if retry_fn else do_call()
    usage = extract_openai_usage(response)
    return parse_openai_response(response.choices[0], usage=usage)


def openai_chat_stream(
    client: Any,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0,
    seed: int | None = 42,
) -> Iterator[str]:
    """Shared streaming implementation for OpenAI and Azure OpenAI."""
    stream = client.chat.completions.create(
        model=model,
        messages=prepare_openai_messages(messages),
        temperature=temperature,
        **({"seed": seed} if seed is not None else {}),
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# -- ChatGPT agent --------------------------------------------------------


class ChatGPT(Agent):
    """Agent backed by the `OpenAI <https://platform.openai.com>`_ API.

    Parameters
    ----------
    model:
        The model identifier.  Defaults to ``"gpt-4o-mini"``.
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when not provided.
    temperature:
        Sampling temperature.  Defaults to ``0`` for deterministic output.
    seed:
        Fixed seed for reproducibility.  Defaults to ``42``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        temperature: float = 0,
        seed: int | None = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client: Any = None

    def ensure_client(self) -> Any:
        """Return the OpenAI client, creating it on first use."""
        if self.client is None:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key)
        return self.client

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Forward *messages* to the OpenAI API and return a :class:`ChatResponse`."""
        return openai_chat(
            self.ensure_client(), self.model, self.functions, messages,
            temperature=self.temperature, seed=self.seed,
            retry_fn=self.with_retry, **kwargs,
        )

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Yield content chunks via OpenAI's native streaming."""
        yield from openai_chat_stream(
            self.ensure_client(), self.model, messages,
            temperature=self.temperature, seed=self.seed,
        )
