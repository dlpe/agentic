"""Anthropic (Claude) LLM backend for agents."""

import json
import logging
import os
from typing import Any, Iterator

from .core import Agent, ChatResponse, Usage

__all__ = ["Claude"]

logger = logging.getLogger("pygentix")


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

        else:
            result.append({"role": role, "content": msg.get("content", "")})

    return "\n".join(system_parts), result


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
        Model identifier. Defaults to ``"claude-sonnet-4-20250514"``.
    api_key:
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    temperature:
        Sampling temperature. Defaults to ``0``.
    max_tokens:
        Maximum tokens in the response. Defaults to ``4096``.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client: Any = None

    def ensure_client(self) -> Any:
        """Return the Anthropic client, creating it on first use."""
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
        return self.client

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Forward messages to the Anthropic API and return a ChatResponse."""
        client = self.ensure_client()
        system, claude_messages = prepare_claude_messages(messages)
        tools = convert_tools_for_claude(self.functions)

        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools

        def do_call() -> Any:
            return client.messages.create(
                model=self.model,
                system=system,
                messages=claude_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **extra,
            )

        response = self.with_retry(do_call)
        return parse_claude_response(response)

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Yield content chunks via Anthropic's native streaming."""
        client = self.ensure_client()
        system, claude_messages = prepare_claude_messages(messages)

        with client.messages.stream(
            model=self.model,
            system=system,
            messages=claude_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text
