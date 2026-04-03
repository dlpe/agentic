"""Ollama LLM backend for agents."""

import logging
from typing import Any, Iterator

from .core import Agent, ChatResponse, Usage

__all__ = ["Ollama"]

logger = logging.getLogger("pygentix")


def prepare_ollama_messages(messages: list[dict]) -> list[dict]:
    """Transform internal message format to what the Ollama library expects.

    The Ollama Pydantic models require tool calls nested under a ``function``
    key and do not accept extra fields like ``tool_call_id`` on tool messages.
    """
    result: list[dict] = []
    for msg in messages:
        role = msg["role"]

        if role == "assistant" and msg.get("tool_calls"):
            result.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": [
                    {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in msg["tool_calls"]
                ],
            })
        elif role == "tool":
            result.append({"role": "tool", "content": msg["content"]})
        else:
            result.append(msg)

    return result


def extract_usage(response: Any) -> Usage:
    """Pull token counts from an Ollama response object."""
    prompt = getattr(response, "prompt_eval_count", 0) or 0
    completion = getattr(response, "eval_count", 0) or 0
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)


DEFAULT_OPTIONS: dict[str, Any] = {"temperature": 0, "top_k": 1, "seed": 42}


class Ollama(Agent):
    """Agent backed by a local `Ollama <https://ollama.com>`_ model.

    Parameters
    ----------
    model:
        The model identifier.  Defaults to ``"qwen2.5:7b"``.
    options:
        Ollama sampling options.  Merged on top of deterministic defaults
        (``temperature=0``, ``top_k=1``, ``seed=42``).
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        *args: Any,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        self.ensure_model_available()

    def ensure_model_available(self) -> None:
        from ollama import list as ollama_list, pull as ollama_pull

        response = ollama_list()
        model_entries = getattr(response, "models", []) or []
        installed = {
            getattr(m, "model", None) or getattr(m, "name", None)
            for m in model_entries
        }
        installed.discard(None)

        if self.model not in installed:
            ollama_pull(self.model)

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Forward *messages* to the Ollama model and return a :class:`ChatResponse`."""
        from ollama import chat as ollama_chat

        fmt = kwargs.pop("format", None)

        def do_call() -> Any:
            return ollama_chat(
                model=self.model,
                messages=prepare_ollama_messages(messages),
                tools=list(self.functions.values()),
                options=self.options,
                **({"format": fmt} if fmt else {}),
            )

        response = self.with_retry(do_call)

        tool_calls = None
        if response.message.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in response.message.tool_calls
            ]

        return ChatResponse(
            content=response.message.content or "",
            tool_calls=tool_calls,
            usage=extract_usage(response),
        )

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Yield content chunks via Ollama's native streaming."""
        from ollama import chat as ollama_chat

        stream = ollama_chat(
            model=self.model,
            messages=prepare_ollama_messages(messages),
            options=self.options,
            stream=True,
        )
        for chunk in stream:
            text = chunk.message.content
            if text:
                yield text
