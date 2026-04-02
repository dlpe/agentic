"""Ollama LLM backend for agents."""

from typing import Any

from .core import Agent, ChatResponse

__all__ = ["Ollama"]


def _prepare_ollama_messages(messages: list[dict]) -> list[dict]:
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


class Ollama(Agent):
    """Agent backed by a local `Ollama <https://ollama.com>`_ model.

    On construction the model is pulled automatically if it isn't already
    installed, so the first instantiation may take a while for large models.
    """

    def __init__(self, model: str = "qwen2.5:7b", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self._ensure_model_available()

    def _ensure_model_available(self) -> None:
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
        response = ollama_chat(
            model=self.model,
            messages=_prepare_ollama_messages(messages),
            tools=list(self.functions.values()),
            **({"format": fmt} if fmt else {}),
        )

        tool_calls = None
        if response.message.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in response.message.tool_calls
            ]

        return ChatResponse(
            content=response.message.content or "",
            tool_calls=tool_calls,
        )
