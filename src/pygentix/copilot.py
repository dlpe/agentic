"""Azure OpenAI (Microsoft Copilot) LLM backend for agents."""

import os
from typing import Any, Iterator

from .chatgpt import openai_chat, openai_chat_stream
from .core import Agent, ChatResponse

__all__ = ["Copilot"]


class Copilot(Agent):
    """Agent backed by `Azure OpenAI <https://azure.microsoft.com/products/ai-services/openai-service>`_.

    Uses the same wire protocol as :class:`~pygentix.chatgpt.ChatGPT` but
    routes requests through your Azure OpenAI deployment.

    Parameters
    ----------
    model:
        The deployment name (e.g. ``"gpt-4o"``).
    api_key:
        Azure OpenAI API key.  Falls back to ``AZURE_OPENAI_API_KEY``.
    endpoint:
        Azure OpenAI endpoint URL.  Falls back to ``AZURE_OPENAI_ENDPOINT``.
    api_version:
        API version string.  Defaults to ``"2024-10-21"``.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str = "2024-10-21",
        temperature: float = 0,
        seed: int | None = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.client: Any = None

    def ensure_client(self) -> Any:
        """Return the Azure OpenAI client, creating it on first use."""
        if self.client is None:
            import openai

            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
            )
        return self.client

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Forward *messages* to Azure OpenAI and return a :class:`ChatResponse`."""
        return openai_chat(
            self.ensure_client(), self.model, self.functions, messages,
            temperature=self.temperature, seed=self.seed,
            retry_fn=self.with_retry, **kwargs,
        )

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Yield content chunks via Azure OpenAI's native streaming."""
        yield from openai_chat_stream(
            self.ensure_client(), self.model, messages,
            temperature=self.temperature, seed=self.seed,
        )
