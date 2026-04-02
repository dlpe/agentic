"""Ollama LLM backend for agents."""

from typing import Any

from .core import Agent

__all__ = ["Ollama"]


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

    def chat(self, messages: list[dict], **kwargs: Any) -> Any:
        """Forward *messages* to the Ollama model and return its response."""
        from ollama import chat as ollama_chat

        fmt = kwargs.pop("format", None)
        return ollama_chat(
            model=self.model,
            messages=messages,
            tools=list(self.functions.values()),
            **({"format": fmt} if fmt else {}),
        )
