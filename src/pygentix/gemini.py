"""Google Gemini LLM backend for agents."""

import logging
import mimetypes
import os
from typing import Any, Iterator

from .core import Agent, ChatResponse, Usage

__all__ = ["Gemini"]

logger = logging.getLogger("pygentix")


class Gemini(Agent):
    """Agent backed by the `Google Gemini <https://ai.google.dev>`_ API.

    Parameters
    ----------
    model:
        The model identifier.  Defaults to ``"gemini-2.5-flash"``.
    api_key:
        Google AI API key.  Falls back to the ``GEMINI_API_KEY`` environment
        variable when not provided.
    temperature:
        Sampling temperature.  Defaults to ``0`` for deterministic output.
    top_k:
        Top-k sampling.  Defaults to ``1`` for deterministic output.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        temperature: float = 0,
        top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client: Any = None

    def ensure_client(self) -> Any:
        """Return the Google GenAI client, creating it on first use."""
        if self.client is None:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)
        return self.client

    # -- chat --------------------------------------------------------------

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Forward *messages* to the Gemini API and return a :class:`ChatResponse`."""
        from google.genai import types

        contents, system_instruction = self.prepare_contents(messages)
        tools = self.build_tools()

        fmt = kwargs.pop("format", None)
        config = types.GenerateContentConfig(
            tools=tools,
            temperature=self.temperature,
            top_k=self.top_k,
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

        def do_call() -> Any:
            return self.ensure_client().models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        response = self.with_retry(do_call)
        return self.parse_response(response)

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Yield content chunks via Gemini's native streaming."""
        from google.genai import types

        contents, system_instruction = self.prepare_contents(messages)
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_k=self.top_k,
            **({"system_instruction": system_instruction} if system_instruction else {}),
        )

        stream = self.ensure_client().models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    # -- internals ---------------------------------------------------------

    def build_tools(self) -> list | None:
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

    def prepare_contents(self, messages: list[dict]) -> tuple[list, str | None]:
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
                contents.append(self.user_content(msg))
            elif role == "assistant":
                contents.append(self.model_content(msg))
            elif role == "tool":
                content, i = self.tool_contents(messages, i)
                contents.append(content)
                continue

            i += 1

        return contents, system_instruction

    @staticmethod
    def user_content(msg: dict) -> Any:
        from google.genai import types

        parts = [types.Part(text=msg["content"])]
        for img_path in msg.get("images") or []:
            mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
            with open(img_path, "rb") as f:
                parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime))

        return types.Content(role="user", parts=parts)

    @staticmethod
    def model_content(msg: dict) -> Any:
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
    def tool_contents(messages: list[dict], i: int) -> tuple[Any, int]:
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
    def parse_response(response: Any) -> ChatResponse:
        """Convert a Gemini ``GenerateContentResponse`` into a :class:`ChatResponse`."""
        content = ""
        tool_calls: list[dict] | None = None

        meta = getattr(response, "usage_metadata", None)
        usage = Usage()
        if meta:
            usage = Usage(
                prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
                total_tokens=getattr(meta, "total_token_count", 0) or 0,
            )

        if not response.candidates:
            return ChatResponse(content=content, usage=usage)

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

        return ChatResponse(content=content, tool_calls=tool_calls, usage=usage)
