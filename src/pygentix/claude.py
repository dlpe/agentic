"""Anthropic (Claude) LLM backend for agents."""

import base64
import json
import logging
import mimetypes
import os
from typing import Any, Iterator

from .core import Agent, ChatResponse, Usage

__all__ = ["Claude"]

logger = logging.getLogger("pygentix")

# Anthropic prompt cache: tools → system → messages. A breakpoint on the last
# stable block lets later turns read that prefix at ~10% input cost.
STABLE_PROMPT_CACHE = {"type": "ephemeral", "ttl": "1h"}
TURN_PROMPT_CACHE = {"type": "ephemeral"}


def prepare_claude_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split internal messages into an Anthropic system string + message list.

    Anthropic requires the system prompt as a separate parameter and uses a
    different structure for tool calls / tool results than OpenAI.
    """
    system_parts: list[str] = []
    result: list[dict] = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_parts.append(msg["content"])
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content_blocks: list[dict] = []
            text = msg.get("content", "")
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tc in msg["tool_calls"]:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or f"call_{tc['name']}",
                        "name": tc["name"],
                        "input": tc["arguments"],
                    }
                )
            result.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id")
                or f"call_{msg.get('tool_name', '')}",
                "content": msg["content"],
            }
            if (
                result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], list)
            ):
                result[-1]["content"].append(tool_result_block)
            else:
                result.append({"role": "user", "content": [tool_result_block]})

        elif role == "user" and msg.get("images"):
            content_blocks = _build_user_file_blocks(msg)
            result.append({"role": "user", "content": content_blocks})

        else:
            result.append({"role": role, "content": msg.get("content", "")})

    return "\n".join(system_parts), result


IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


def _build_user_file_blocks(msg: dict) -> list[dict]:
    """Build Anthropic content blocks for a user message with attached files."""
    blocks: list[dict] = []

    for path in msg["images"]:
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

        if mime in IMAGE_MIMES:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": data},
                }
            )

        elif mime == "application/pdf":
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            blocks.append(
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": mime, "data": data},
                }
            )

        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                name = os.path.basename(path)
                blocks.append(
                    {
                        "type": "text",
                        "text": f"File contents of {name}:\n{content}",
                    }
                )
            except (UnicodeDecodeError, OSError):
                logger.warning("Cannot read file %s as text, skipping", path)

    text = msg.get("content", "")
    if text:
        blocks.append({"type": "text", "text": text})

    return blocks


def convert_tools_for_claude(functions: dict) -> list[dict] | None:
    """Convert pygentix Function objects to Anthropic tool definitions."""
    if not functions:
        return None
    tools = []
    for f in functions.values():
        schema = f.to_tool_schema()["function"]
        tools.append(
            {
                "name": schema["name"],
                "description": schema["description"],
                "input_schema": schema["parameters"],
            }
        )
    return tools


def cached_system_blocks(system: str) -> list[dict] | str:
    """Wrap the system prompt so Anthropic can cache it across calls."""
    text = (system or "").strip()
    if not text:
        return system
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": dict(STABLE_PROMPT_CACHE),
        }
    ]


def mark_tools_cached(tools: list[dict] | None) -> list[dict] | None:
    """Put a cache breakpoint on the last tool (caches the whole tool list)."""
    if not tools:
        return tools
    marked = [dict(tool) for tool in tools]
    marked[-1] = {**marked[-1], "cache_control": dict(STABLE_PROMPT_CACHE)}
    return marked


def mark_messages_cached(messages: list[dict]) -> list[dict]:
    """Mark the last content block so growing history can be reread from cache."""
    if not messages:
        return messages
    result = list(messages)
    last = dict(result[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": dict(TURN_PROMPT_CACHE),
            }
        ]
    elif isinstance(content, list) and content:
        blocks = [dict(block) for block in content]
        blocks[-1] = {**blocks[-1], "cache_control": dict(TURN_PROMPT_CACHE)}
        last["content"] = blocks
    result[-1] = last
    return result


def build_claude_request(
    messages: list[dict],
    functions: dict,
    *,
    prompt_cache: bool = True,
) -> dict[str, Any]:
    """Shape system, messages, and tools for ``messages.create`` / ``stream``."""
    system, claude_messages = prepare_claude_messages(messages)
    tools = convert_tools_for_claude(functions)
    if prompt_cache:
        tools = mark_tools_cached(tools)
        claude_messages = mark_messages_cached(claude_messages)
        system_arg: Any = cached_system_blocks(system)
    else:
        system_arg = system
    payload: dict[str, Any] = {
        "system": system_arg,
        "messages": claude_messages,
    }
    if tools:
        payload["tools"] = tools
    return payload


def parse_claude_response(response: Any) -> ChatResponse:
    """Convert an Anthropic Message into a pygentix ChatResponse."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                }
            )

    usage = Usage(
        prompt_tokens=getattr(response.usage, "input_tokens", 0),
        completion_tokens=getattr(response.usage, "output_tokens", 0),
        total_tokens=(
            getattr(response.usage, "input_tokens", 0)
            + getattr(response.usage, "output_tokens", 0)
        ),
    )

    return ChatResponse(
        content="".join(text_parts),
        tool_calls=tool_calls or None,
        usage=usage,
    )


class Claude(Agent):
    """Agent backed by the `Anthropic <https://docs.anthropic.com>`_ API.

    Parameters
    ----------
    model:
        Model identifier. Defaults to ``"claude-sonnet-4-6"``.
    api_key:
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    temperature:
        Sampling temperature. Defaults to ``0``.
    max_tokens:
        Maximum tokens in the response. Defaults to ``4096``.
    prompt_cache:
        When True (default), attach Anthropic ``cache_control`` breakpoints
        on tools, the system prompt, and the last message. Set
        ``ANTHROPIC_PROMPT_CACHE=0`` to disable.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        prompt_cache: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client: Any = None
        if prompt_cache is None:
            raw = os.environ.get("ANTHROPIC_PROMPT_CACHE", "1").strip().lower()
            prompt_cache = raw not in {"0", "false", "no", "off"}
        self.prompt_cache = prompt_cache

    def ensure_client(self) -> Any:
        """Return the Anthropic client, creating it on first use."""
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
        return self.client

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Forward messages to the Anthropic API and return a ChatResponse."""
        client = self.ensure_client()
        request = build_claude_request(
            messages, self.functions, prompt_cache=self.prompt_cache
        )

        def do_call() -> Any:
            return client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **request,
            )

        response = self.with_retry(do_call)
        return parse_claude_response(response)

    def stream_chat_turn(
        self,
        messages: list[dict],
        collector: list | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield text deltas; keep tool_use from the completed stream message."""
        client = self.ensure_client()
        request = build_claude_request(
            messages, self.functions, prompt_cache=self.prompt_cache
        )

        with client.messages.stream(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **request,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
            response = parse_claude_response(stream.get_final_message())
        if collector is not None:
            collector.append(response)

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Yield content chunks via Anthropic's native streaming."""
        yield from self.stream_chat_turn(messages, **kwargs)
