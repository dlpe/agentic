"""Azure OpenAI (Microsoft Copilot) LLM backend for agents."""

import os
from typing import Any

from .chatgpt import _openai_chat
from .core import Agent, ChatResponse

__all__ = ["Copilot"]


class Copilot(Agent):
    """Agent backed by `Azure OpenAI <https://azure.microsoft.com/products/ai-services/openai-service>`_.

    Uses the same wire protocol as :class:`~pygentix.chatgpt.ChatGPT` but
    routes requests through your Azure OpenAI deployment, which is the
    engine behind Microsoft Copilot.

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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_version = api_version
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialised Azure OpenAI client."""
        if self._client is None:
            import openai

            self._client = openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._endpoint,
                api_version=self._api_version,
            )
        return self._client

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Forward *messages* to Azure OpenAI and return a :class:`ChatResponse`."""
        return _openai_chat(self.client, self.model, self.functions, messages, **kwargs)
