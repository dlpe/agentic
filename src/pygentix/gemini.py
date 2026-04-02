"""Google Gemini LLM backend for agents."""

import mimetypes
import os
from typing import Any

from .core import Agent, ChatResponse

__all__ = ["Gemini"]


class Gemini(Agent):
    """Agent backed by the `Google Gemini <https://ai.google.dev>`_ API.

    Works with any Gemini chat model (``gemini-2.5-flash``,
    ``gemini-2.5-pro``, etc.).

    Parameters
    ----------
    model:
        The model identifier.  Defaults to ``"gemini-2.5-flash"``.
    api_key:
        Google AI API key.  Falls back to the ``GEMINI_API_KEY`` environment
        variable when not provided.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialised Google GenAI client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # -- chat --------------------------------------------------------------

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Forward *messages* to the Gemini API and return a :class:`ChatResponse`."""
        from google.genai import types

        contents, system_instruction = self._prepare_contents(messages)
        tools = self._build_tools()

        fmt = kwargs.pop("format", None)
        config = types.GenerateContentConfig(
            tools=tools,
            **({"system_instruction": system_instruction} if system_instruction else {}),
            **(
                {
                    "response_mime_type": "application/json",
                    "response_schema": fmt,
                }
                if fmt
                else {}
            ),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return self._parse_response(response)

    # -- internals ---------------------------------------------------------

    def _build_tools(self) -> list | None:
        """Convert registered functions to Gemini tool declarations."""
        if not self.functions:
            return None

        from google.genai import types

        declarations = []
        for func in self.functions.values():
            schema = func.to_tool_schema()["function"]
            declarations.append(
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=schema["parameters"],
                )
            )

        return [types.Tool(function_declarations=declarations)]

    def _prepare_contents(self, messages: list[dict]) -> tuple[list, str | None]:
        """Transform internal messages to the Gemini *contents* format.

        Returns ``(contents_list, system_instruction)``.
        """
        contents: list = []
        system_instruction: str | None = None

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                system_instruction = msg["content"]
            elif role == "user":
                contents.append(self._user_content(msg))
            elif role == "assistant":
                contents.append(self._model_content(msg))
            elif role == "tool":
                content, i = self._tool_contents(messages, i)
                contents.append(content)
                continue

            i += 1

        return contents, system_instruction

    @staticmethod
    def _user_content(msg: dict) -> Any:
        from google.genai import types

        parts = [types.Part(text=msg["content"])]
        for img_path in msg.get("images") or []:
            mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
            with open(img_path, "rb") as f:
                parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime))

        return types.Content(role="user", parts=parts)

    @staticmethod
    def _model_content(msg: dict) -> Any:
        from google.genai import types

        if msg.get("tool_calls"):
            parts = [
                types.Part(
                    function_call=types.FunctionCall(
                        name=tc["name"],
                        args=tc["arguments"],
                        **({"id": tc["id"]} if tc.get("id") else {}),
                    )
                )
                for tc in msg["tool_calls"]
            ]
            return types.Content(role="model", parts=parts)

        return types.Content(
            role="model",
            parts=[types.Part(text=msg.get("content") or "")],
        )

    @staticmethod
    def _tool_contents(messages: list[dict], i: int) -> tuple[Any, int]:
        """Collect consecutive tool messages into a single Gemini Content."""
        from google.genai import types

        parts = []
        while i < len(messages) and messages[i]["role"] == "tool":
            m = messages[i]
            parts.append(
                types.Part.from_function_response(
                    name=m.get("tool_name", ""),
                    response={"result": m["content"]},
                    **({"id": m["tool_call_id"]} if m.get("tool_call_id") else {}),
                )
            )
            i += 1
        return types.Content(role="user", parts=parts), i

    @staticmethod
    def _parse_response(response: Any) -> ChatResponse:
        """Convert a Gemini ``GenerateContentResponse`` into a :class:`ChatResponse`."""
        content = ""
        tool_calls: list[dict] | None = None

        if not response.candidates:
            return ChatResponse(content=content)

        parts = response.candidates[0].content.parts
        tc_list: list[dict] = []

        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tc_list.append({
                    "name": fc.name,
                    "arguments": dict(fc.args) if fc.args else {},
                    "id": getattr(fc, "id", None),
                })
            elif hasattr(part, "text") and part.text:
                content = part.text

        if tc_list:
            tool_calls = tc_list

        return ChatResponse(content=content, tool_calls=tool_calls)
