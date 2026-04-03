"""Core abstractions for building AI agents with tool-calling capabilities."""

import asyncio
import contextvars
import json
import logging
import time
from abc import ABC, abstractmethod
from functools import wraps
import inspect
from typing import Any, Callable, Iterator

__all__ = ["Function", "ChatResponse", "Usage", "Conversation", "Agent", "active_scope", "active_conversation"]

active_scope: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "pygentix_scope", default=None,
)

active_conversation: contextvars.ContextVar["Conversation | None"] = contextvars.ContextVar(
    "pygentix_conversation", default=None,
)

logger = logging.getLogger("pygentix")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful agent. "
    "If you need a tool to answer the question, ALWAYS call tools immediately. "
    "NEVER describe what you intend to do or ask for confirmation — just call the tool. "
    "If you can provide the answer directly, do so. "
    "If your answer doesn't call a tool, it must be the final answer. "
    "No follow-up questions are allowed."
)

TOOL_NUDGE = "Don't describe what you will do. Call the appropriate tool now."

PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}

RETRIABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


# -- Response types --------------------------------------------------------


class Usage:
    """Token usage statistics for a single LLM call.

    Populated automatically by backends that report token counts.
    """
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    def __bool__(self) -> bool:
        return self.total_tokens > 0

    def __repr__(self) -> str:
        return (
            f"Usage(prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, "
            f"total_tokens={self.total_tokens})"
        )


class FunctionCall:
    """Name + arguments of a single function invocation."""
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: dict) -> None:
        self.name = name
        self.arguments = arguments


class ToolCall:
    """A tool call requested by the model, optionally carrying a provider ID."""
    __slots__ = ("id", "function")

    def __init__(self, name: str, arguments: dict, id: str | None = None) -> None:
        self.id = id
        self.function = FunctionCall(name, arguments)


class Message:
    __slots__ = ("content", "tool_calls")

    def __init__(self, content: str, tool_calls: list[ToolCall] | None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class ChatResponse:
    """Normalized response that all LLM backends return.

    Provides a uniform interface so :class:`Conversation` can drive any
    backend without caring which provider produced the response::

        response.message.content        # str
        response.message.tool_calls     # list[ToolCall] | None
        call.function.name              # str
        call.function.arguments         # dict
        call.id                         # str | None  (provider-specific)
        response.usage                  # Usage (token counts)
    """
    __slots__ = ("message", "usage")

    def __init__(
        self,
        content: str = "",
        tool_calls: list[dict] | None = None,
        usage: Usage | None = None,
    ) -> None:
        parsed = None
        if tool_calls:
            parsed = [
                ToolCall(tc["name"], tc["arguments"], tc.get("id"))
                for tc in tool_calls
            ]
        self.message = Message(content, parsed)
        self.usage = usage or Usage()


# -- Function wrapper ------------------------------------------------------


class Function:
    """Introspectable wrapper around a callable, used to expose tools to an LLM.

    Captures the function's signature and source code at wrap time so the
    model backend can generate accurate tool definitions.  Attribute access
    is proxied to the underlying function, which lets libraries like
    ``ollama`` read ``__name__``, ``__doc__``, etc. transparently.
    """

    def __init__(self, func: Callable) -> None:
        self.func = func
        self.signature = inspect.signature(func)
        self.code = inspect.getsource(func)
        self.file = inspect.getfile(func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.func, name)

    @property
    def name(self) -> str:
        return self.func.__name__

    @property
    def docs(self) -> str | None:
        return self.func.__doc__

    @property
    def parameters(self) -> dict:
        return self.signature.parameters

    def __repr__(self) -> str:
        return f"Function({self.name})"

    def to_tool_schema(self) -> dict:
        """Generate an OpenAI-compatible tool definition for this function."""
        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in self.parameters.items():
            if param_name == "self":
                continue
            annotation = param.annotation
            json_type = PYTHON_TO_JSON_TYPE.get(
                annotation if annotation is not inspect.Parameter.empty else str,
                "string",
            )
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.docs or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# -- Conversation ----------------------------------------------------------


class Conversation:
    """Manages a multi-turn conversation between a user and an agent.

    Each call to :meth:`ask` runs a three-stage pipeline:

    1. **Prompt** — send the question; retry with a nudge if the model
       narrates instead of calling a tool.
    2. **Execute** — run every tool the model invokes, looping until it
       stops requesting tools.
    3. **Format** — if the agent defines an ``output_schema``, make one
       final call with a ``format`` constraint so the response is
       guaranteed valid JSON.

    Parameters
    ----------
    max_history:
        When set, keeps only the most recent *N* messages (plus the
        system prompt) to prevent exceeding the model's context window.
    scope:
        Key-value pairs representing the caller's identity context
        (e.g. ``{"current_user": 5}``).  Passed to
        :class:`~pygentix.sqlalchemy.SqlAlchemyAgent` for automatic
        row-level filtering and to the *policy* callback.
    policy:
        Optional callback invoked before every tool execution.
        Signature: ``(tool_name, arguments, scope) -> bool``.
        Return ``False`` to deny execution; the LLM receives a
        *"Permission denied"* tool result instead.
    """

    def __init__(
        self,
        agent: "Agent",
        system: str,
        max_history: int | None = None,
        scope: dict | None = None,
        policy: Callable[..., bool] | None = None,
    ) -> None:
        self.agent = agent
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self.max_history = max_history
        self.scope: dict = scope or {}
        self.policy = policy

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the conversation to a plain dictionary.

        The *policy* callback is not serialized (functions are not JSON-safe).
        """
        return {
            "messages": list(self.messages),
            "max_history": self.max_history,
            "scope": self.scope or None,
        }

    @classmethod
    def from_dict(cls, agent: "Agent", data: dict) -> "Conversation":
        """Restore a conversation from a dictionary produced by :meth:`to_dict`."""
        conv = cls.__new__(cls)
        conv.agent = agent
        conv.messages = list(data["messages"])
        conv.max_history = data.get("max_history")
        conv.scope = data.get("scope") or {}
        conv.policy = None
        return conv

    def to_json(self) -> str:
        """Serialize the conversation to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, agent: "Agent", json_str: str) -> "Conversation":
        """Restore a conversation from a JSON string."""
        return cls.from_dict(agent, json.loads(json_str))

    # -- context management ------------------------------------------------

    def trim_context(self) -> None:
        """Drop old messages when *max_history* is set, preserving the system prompt."""
        if not self.max_history or len(self.messages) <= self.max_history + 1:
            return
        system = self.messages[0]
        self.messages = [system] + self.messages[-self.max_history :]
        logger.debug("Trimmed context to %d messages", len(self.messages))

    # -- public API --------------------------------------------------------

    def ask(
        self,
        question: str,
        images: list[str] | None = None,
        max_retries: int = 3,
    ) -> ChatResponse:
        """Send *question* and return the model's response.

        Parameters
        ----------
        images:
            Optional list of image file paths to include with the question.
        max_retries:
            Maximum retries when the model narrates instead of calling a tool.
        """
        msg: dict[str, Any] = {"role": "user", "content": question}
        if images:
            msg["images"] = images
        self.messages.append(msg)
        self.trim_context()
        logger.info("User: %s", question[:120])

        response = self.prompt_until_actionable(max_retries)
        response = self.execute_tool_calls(response)
        response = self.apply_output_schema(response)

        self.messages.append({"role": "assistant", "content": response.message.content})
        logger.info("Assistant: %s", response.message.content[:120])
        return response

    def ask_stream(
        self,
        question: str,
        images: list[str] | None = None,
        max_retries: int = 3,
    ) -> Iterator[str]:
        """Like :meth:`ask` but yields content chunks as they arrive.

        Tool calls are resolved synchronously mid-conversation.  The final
        text response is streamed via the backend's ``chat_stream`` method.
        When tools are registered and the model chooses to call one, the
        tool loop runs non-streaming and the final answer is yielded whole.
        """
        msg: dict[str, Any] = {"role": "user", "content": question}
        if images:
            msg["images"] = images
        self.messages.append(msg)
        self.trim_context()
        logger.info("User (stream): %s", question[:120])

        if self.agent.functions:
            response = self.prompt_until_actionable(max_retries)
            if response.message.tool_calls:
                response = self.execute_tool_calls(response)
                yield from self.stream_final()
                return
            response = self.apply_output_schema(response)
            self.messages.append({"role": "assistant", "content": response.message.content})
            yield response.message.content
            return

        yield from self.stream_final()

    async def ask_async(
        self,
        question: str,
        images: list[str] | None = None,
        max_retries: int = 3,
    ) -> ChatResponse:
        """Async version of :meth:`ask`.

        Uses the backend's ``chat_async`` method (native async when
        available, ``asyncio.to_thread`` by default).  Tool functions
        are executed via ``asyncio.to_thread`` so they don't block the
        event loop.
        """
        msg: dict[str, Any] = {"role": "user", "content": question}
        if images:
            msg["images"] = images
        self.messages.append(msg)
        self.trim_context()
        logger.info("User (async): %s", question[:120])

        response = await self.prompt_until_actionable_async(max_retries)
        response = await self.execute_tool_calls_async(response)
        response = await self.apply_output_schema_async(response)

        self.messages.append({"role": "assistant", "content": response.message.content})
        logger.info("Assistant (async): %s", response.message.content[:120])
        return response

    # -- private sync helpers ----------------------------------------------

    def prompt_until_actionable(self, max_retries: int) -> ChatResponse:
        """Prompt the model, retrying with a nudge if it narrates instead of acting."""
        for attempt in range(max_retries):
            response = self.agent.chat(messages=self.messages)
            self.agent.fire("response", response)

            if response.message.tool_calls:
                return response

            should_retry = (
                attempt < max_retries - 1
                and self.agent.functions
                and not self.has_prior_tool_result()
            )
            if should_retry:
                self.messages.append({"role": "assistant", "content": response.message.content})
                self.messages.append({"role": "user", "content": TOOL_NUDGE})
            else:
                return response

        return response  # pragma: no cover — loop always returns

    def check_policy(self, tool_name: str, arguments: dict) -> str | None:
        """Run the policy callback; return an error string if denied, else *None*."""
        if not self.policy:
            return None
        try:
            allowed = self.policy(tool_name, arguments, self.scope)
        except Exception as exc:
            logger.warning("Policy callback raised: %s", exc)
            return f"Permission denied: policy error — {exc}"
        if not allowed:
            logger.info("Policy denied %s(%s)", tool_name, arguments)
            return f"Permission denied: {tool_name} blocked by policy"
        return None

    def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a single tool with scope and conversation context propagation."""
        scope_token = active_scope.set(self.scope)
        conv_token = active_conversation.set(self)
        try:
            return str(self.agent.functions[name](**arguments))
        finally:
            active_conversation.reset(conv_token)
            active_scope.reset(scope_token)

    def execute_tool_calls(self, response: ChatResponse) -> ChatResponse:
        """Execute tool calls in a loop until the model stops requesting them."""
        while response.message.tool_calls:
            self.messages.append({
                "role": "assistant",
                "content": response.message.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in response.message.tool_calls
                ],
            })

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments
                self.agent.fire("tool_call", name, args)
                logger.debug("Calling tool %s(%s)", name, args)

                denied = self.check_policy(name, args)
                if denied:
                    result = denied
                else:
                    try:
                        result = self.call_tool(name, args)
                    except Exception as exc:
                        result = f"Tool error: {exc}"

                self.agent.fire("tool_result", name, result)
                logger.debug("Tool %s → %s", name, result[:200])
                self.messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "tool_call_id": call.id,
                    "content": result,
                })

            response = self.agent.chat(messages=self.messages)
            self.agent.fire("response", response)

        return response

    def apply_output_schema(self, response: ChatResponse) -> ChatResponse:
        """Re-prompt with a format constraint if the agent defines an output schema."""
        schema = getattr(self.agent, "output_schema", None)
        if schema:
            return self.agent.chat(messages=self.messages, format=schema)
        return response

    def stream_final(self) -> Iterator[str]:
        """Stream the model's response from the current message state."""
        schema = getattr(self.agent, "output_schema", None)
        if schema:
            response = self.agent.chat(messages=self.messages, format=schema)
            self.messages.append({"role": "assistant", "content": response.message.content})
            yield response.message.content
        else:
            parts: list[str] = []
            for chunk in self.agent.chat_stream(messages=self.messages):
                parts.append(chunk)
                yield chunk
            content = "".join(parts)
            self.messages.append({"role": "assistant", "content": content})
            logger.info("Assistant (stream): %s", content[:120])

    def has_prior_tool_result(self) -> bool:
        """True if a tool result exists after the most recent user message."""
        for msg in reversed(self.messages):
            if msg["role"] == "tool":
                return True
            if msg["role"] == "user":
                return False
        return False

    # -- private async helpers ---------------------------------------------

    async def prompt_until_actionable_async(self, max_retries: int) -> ChatResponse:
        for attempt in range(max_retries):
            response = await self.agent.chat_async(messages=self.messages)
            self.agent.fire("response", response)

            if response.message.tool_calls:
                return response

            should_retry = (
                attempt < max_retries - 1
                and self.agent.functions
                and not self.has_prior_tool_result()
            )
            if should_retry:
                self.messages.append({"role": "assistant", "content": response.message.content})
                self.messages.append({"role": "user", "content": TOOL_NUDGE})
            else:
                return response

        return response  # pragma: no cover

    async def execute_tool_calls_async(self, response: ChatResponse) -> ChatResponse:
        while response.message.tool_calls:
            self.messages.append({
                "role": "assistant",
                "content": response.message.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in response.message.tool_calls
                ],
            })

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments
                self.agent.fire("tool_call", name, args)
                logger.debug("Calling tool %s(%s)", name, args)

                denied = self.check_policy(name, args)
                if denied:
                    result = denied
                else:
                    try:
                        result = str(
                            await asyncio.to_thread(self.call_tool, name, args)
                        )
                    except Exception as exc:
                        result = f"Tool error: {exc}"

                self.agent.fire("tool_result", name, result)
                logger.debug("Tool %s → %s", name, result[:200])
                self.messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "tool_call_id": call.id,
                    "content": result,
                })

            response = await self.agent.chat_async(messages=self.messages)
            self.agent.fire("response", response)

        return response

    async def apply_output_schema_async(self, response: ChatResponse) -> ChatResponse:
        schema = getattr(self.agent, "output_schema", None)
        if schema:
            return await self.agent.chat_async(messages=self.messages, format=schema)
        return response


# -- Agent -----------------------------------------------------------------


class Agent(ABC):
    """Base class for all agents.

    Subclasses must implement :meth:`chat`.  Optionally register tools
    with the :meth:`uses` decorator and start conversations with
    :meth:`start_conversation`.

    Parameters
    ----------
    max_retries:
        How many times to retry on transient API errors (rate-limits,
        connection drops).  Applies to :meth:`chat` calls wrapped with
        :meth:`with_retry`.
    retry_delay:
        Initial delay in seconds between retries (doubles each attempt).
    """

    def __init__(
        self,
        *args: Any,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.functions: dict[str, Function] = {}
        self.conversations: list[Conversation] = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.hooks: dict[str, list[Callable]] = {
            "tool_call": [],
            "tool_result": [],
            "response": [],
        }

    # -- hooks -------------------------------------------------------------

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for a lifecycle event.

        Events
        ------
        ``"tool_call"``
            Fired before a tool executes.  Signature: ``(name, arguments)``.
        ``"tool_result"``
            Fired after a tool returns.  Signature: ``(name, result_str)``.
        ``"response"``
            Fired after every LLM call.  Signature: ``(response,)``.
        """
        if event not in self.hooks:
            raise ValueError(f"Unknown event {event!r}. Valid: {list(self.hooks)}")
        self.hooks[event].append(callback)

    def fire(self, event: str, *args: Any) -> None:
        for cb in self.hooks.get(event, []):
            try:
                cb(*args)
            except Exception:
                logger.exception("Hook error for event %r", event)

    # -- retry helper ------------------------------------------------------

    @staticmethod
    def is_retriable(exc: Exception) -> bool:
        """Return *True* for transient errors worth retrying."""
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return True
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status is not None:
            try:
                return int(status) in RETRIABLE_STATUS_CODES
            except (ValueError, TypeError):
                pass
        return False

    def with_retry(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call *fn* with exponential backoff on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries - 1 or not self.is_retriable(exc):
                    raise
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries, exc, delay,
                )
                time.sleep(delay)
        raise last_exc  # pragma: no cover

    async def with_retry_async(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Async version of :meth:`with_retry`."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries - 1 or not self.is_retriable(exc):
                    raise
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
        raise last_exc  # pragma: no cover

    # -- tool registration -------------------------------------------------

    def uses(self, func: Callable) -> Callable:
        """Register *func* as a tool the agent can invoke."""
        f = Function(func)
        self.functions[f.name] = f

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapped

    # -- chat methods ------------------------------------------------------

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Send *messages* to the model and return a :class:`ChatResponse`."""
        ...

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Yield response content in chunks.

        Override in backends that support native streaming.
        The default implementation falls back to a single-chunk response
        from :meth:`chat`.
        """
        response = self.chat(messages, **kwargs)
        yield response.message.content

    async def chat_async(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        """Async chat.

        Override in backends for native async.  The default runs
        :meth:`chat` in a thread pool via ``asyncio.to_thread``.
        """
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    # -- conversation management -------------------------------------------

    def start_conversation(
        self,
        system: str = DEFAULT_SYSTEM_PROMPT,
        max_history: int | None = None,
        scope: dict | None = None,
        policy: Callable[..., bool] | None = None,
    ) -> Conversation:
        """Begin a new conversation with the given system prompt.

        Parameters
        ----------
        scope:
            Identity context forwarded to row-level security filters
            and the *policy* callback (e.g. ``{"current_user": 5}``).
        policy:
            Optional ``(tool_name, arguments, scope) -> bool`` gate
            evaluated before every tool execution.
        """
        conv = Conversation(
            self, system, max_history=max_history, scope=scope, policy=policy,
        )
        self.conversations.append(conv)
        return conv
