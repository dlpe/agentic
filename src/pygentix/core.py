"""Core abstractions for building AI agents with tool-calling capabilities."""

import asyncio
import contextvars
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import wraps
import inspect
from typing import Any, Callable, Iterator

__all__ = [
    "Function",
    "ChatResponse",
    "Usage",
    "Conversation",
    "Agent",
    "active_scope",
    "active_conversation",
    "normalize_tool_arguments",
    "looks_like_plan",
    "looks_like_finished_answer",
]

active_scope: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "pygentix_scope",
    default=None,
)

active_conversation: contextvars.ContextVar["Conversation | None"] = (
    contextvars.ContextVar(
        "pygentix_conversation",
        default=None,
    )
)

logger = logging.getLogger("pygentix")

DEFAULT_SYSTEM_PROMPT = """You are a tool-driven assistant.

Reply in exactly one of two ways:
- Tool call(s) with empty text, when you still need data or side effects.
- The complete user-facing answer, when you already have the data.

Never announce a plan, list next steps, or say what you are about to do. If you need tools, call them now. Greetings and questions that need no tools get a short direct answer."""

# One-shot only: the model said it would act and then stopped. Never loop.
TOOL_NUDGE = "Call the required tool(s) now. Do not describe the plan."

PLAN_MARKERS = (
    "let me ",
    "let's ",
    "i'll ",
    "i will ",
    "i am going to ",
    "i'm going to ",
    "i need to ",
    "first i ",
    "next i ",
    "now i'll ",
    "now i will ",
    "to answer this",
    "to calculate",
    "to do this",
    "here's what i'll",
    "sure, i'll",
    "okay, i'll",
    "we need to ",
    "we should ",
    "i would run ",
)

DEFAULT_MAX_TOOL_ROUNDS = 8


PLAN_OPENING_CHARS = 160


def tool_call_fingerprint(name: str, arguments: Any) -> str:
    return json.dumps(
        {"name": name, "arguments": arguments},
        sort_keys=True,
        default=str,
    )


def looks_like_finished_answer(text: str) -> bool:
    """True when *text* already delivered the result (link, file, or URL)."""
    content = text or ""
    return (
        "/static/" in content
        or "download_url" in content
        or "http://" in content
        or "https://" in content
    )


def looks_like_plan(text: str) -> bool:
    """True when the opening of *text* is a plan, not a finished answer."""
    content = (text or "").strip()
    if not content or looks_like_finished_answer(content):
        return False
    opening = content[:PLAN_OPENING_CHARS]
    for separator in (". ", "! ", "? ", "\n"):
        index = opening.find(separator)
        if index != -1:
            opening = opening[: index + 1]
            break
    lowered = opening.lower().replace("let me know", " ")
    if any(marker in lowered for marker in PLAN_MARKERS):
        return True
    return content.endswith(":") and len(content) < 400

PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

RETRIABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def normalize_tool_arguments(
    raw: Any,
    *,
    tool_name: str | None = None,
) -> dict:
    """Turn provider-native tool arguments into a plain ``dict``.

    All backends should pass tool call payloads through here (via
    :class:`ChatResponse`) so :class:`Conversation` and :class:`Function`
    always see the same shape — only wire format adapters differ per vendor.
    """
    if raw is None:
        return {}
    if isinstance(raw, Mapping) and not isinstance(raw, str | bytes | bytearray):
        return dict(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        if stripped[0] not in "[{":
            logger.warning(
                "Tool %r arguments look non-JSON (ignored): %r",
                tool_name or "?",
                stripped[:120],
            )
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Tool %r arguments JSON parse failed: %s — raw %r",
                tool_name or "?",
                exc,
                stripped[:120],
            )
            return {}
        if isinstance(parsed, dict):
            return parsed
        logger.warning(
            "Tool %r arguments JSON must be an object, got %s",
            tool_name or "?",
            type(parsed).__name__,
        )
        return {}
    logger.warning(
        "Tool %r arguments have unexpected type %s",
        tool_name or "?",
        type(raw).__name__,
    )
    return {}


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
                ToolCall(
                    tc["name"],
                    normalize_tool_arguments(
                        tc.get("arguments"),
                        tool_name=tc.get("name"),
                    ),
                    tc.get("id"),
                )
                for tc in tool_calls
            ]
        self.message = Message(content, parsed)
        self.usage = usage or Usage()


# -- Function wrapper ------------------------------------------------------


# Always omitted from the tool schema so the model cannot pick another tenant,
# even if schema is built before scope is set.
IDENTITY_SCOPE_KEYS = frozenset({"enterprise_id", "user_id", "username", "role"})


class Function:
    """Introspectable wrapper around a callable, used to expose tools to an LLM.

    Captures the function's signature at wrap time so the backend can generate
    accurate tool definitions.  Any parameter whose name matches a key in the
    active :data:`active_scope` is filled from the scope at call time and
    hidden from the LLM-visible tool schema, so the model cannot craft
    arguments that widen access (e.g. by passing a different ``enterprise_id``).

    A *serializer* may be provided to post-process the return value, and
    *name* / *description* may be supplied to override ``func.__name__`` /
    ``func.__doc__`` in the schema.
    """

    def __init__(
        self,
        func: Callable,
        *,
        serializer: Callable[[Any], Any] | None = None,
        description: str | None = None,
        name: str | None = None,
    ) -> None:
        self.func = func
        self.signature = inspect.signature(func)
        try:
            self.code = inspect.getsource(func)
        except (OSError, TypeError):
            self.code = ""
        try:
            self.file = inspect.getfile(func)
        except TypeError:
            self.file = ""
        self._serializer = serializer
        self._name_override = name
        self._description_override = description

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        scope = active_scope.get() or {}
        merged = dict(kwargs)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in self.signature.parameters.values()
        )
        for param_name, param in self.signature.parameters.items():
            if param_name == "self" or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if param_name in scope:
                merged[param_name] = scope[param_name]
        if not has_var_kw:
            allowed = {
                n
                for n, p in self.signature.parameters.items()
                if n != "self" and p.kind != inspect.Parameter.VAR_KEYWORD
            }
            merged = {k: v for k, v in merged.items() if k in allowed}
        bound = self.signature.bind_partial(*args, **merged)
        result = self.func(*bound.args, **bound.kwargs)
        if self._serializer is not None:
            result = self._serializer(result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.func, name)

    @property
    def name(self) -> str:
        return self._name_override or self.func.__name__

    @property
    def docs(self) -> str | None:
        return self._description_override or self.func.__doc__

    @property
    def parameters(self) -> dict:
        return self.signature.parameters

    def __repr__(self) -> str:
        return f"Function({self.name})"

    def to_tool_schema(self) -> dict:
        """Generate an OpenAI-compatible tool definition for this function.

        Parameters whose name matches a key in the active scope are omitted —
        they are filled automatically at call time and must never be part of
        the LLM's visible surface.
        """
        properties: dict[str, dict] = {}
        required: list[str] = []
        scope_keys = set((active_scope.get() or {}).keys()) | IDENTITY_SCOPE_KEYS

        for param_name, param in self.parameters.items():
            if param_name == "self":
                continue
            if param_name in scope_keys:
                continue
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                annotation = str
            origin = getattr(annotation, "__origin__", None)
            json_type = PYTHON_TO_JSON_TYPE.get(
                origin or annotation,
                "string",
            )
            prop: dict[str, Any] = {"type": json_type}
            if json_type == "array":
                prop["items"] = {"type": "object"}
            properties[param_name] = prop
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

        1. **Prompt** — one model call for this user turn. If that reply is
       "I'll do X" with no tool call, nudge **once** so it actually calls
       the tool. Never nudge again on that turn.
        2. **Execute** — run tools until the model stops or
       :attr:`max_tool_rounds` is hit. Same tool+args is not re-run.
    3. **Format** — if the agent defines an ``output_schema``, optionally
       re-prompt with a ``format`` constraint when the reply is not already
       valid JSON (see :meth:`apply_output_schema`).

    The system message is always anchored to :data:`DEFAULT_SYSTEM_PROMPT`.
    Mixins append context via :meth:`append_system_supplement`; the full
    system text is recomposed before every model call so accidental edits
    to ``messages[0]`` cannot replace that policy.

    Parameters
    ----------
    max_history:
        When set, keeps only the most recent *N* messages (plus the
        system prompt) to prevent exceeding the model's context window.
    max_tool_rounds:
        Cap on tool-call batches per user turn. Prevents an unbounded
        tool loop from hanging the conversation.
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
        *,
        max_history: int | None = None,
        scope: dict | None = None,
        policy: Callable[..., bool] | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ) -> None:
        self.agent = agent
        self.system_supplements: list[str] = []
        self.max_history = max_history
        self.max_tool_rounds = max(1, max_tool_rounds)
        self.turn_tool_cache: dict[str, str] = {}
        self.scope: dict = scope or {}
        self.policy = policy
        self.messages: list[dict] = []
        self.sync_system_message()

    def compose_system_content(self) -> str:
        """Full system text: locked core plus registered supplements."""
        parts = [DEFAULT_SYSTEM_PROMPT]
        parts.extend(s.strip() for s in self.system_supplements if s and s.strip())
        return "\n\n".join(parts)

    def append_system_supplement(self, text: str) -> None:
        """Append immutable context (schemas, DB hints) after the locked core."""
        chunk = (text or "").strip()
        if not chunk:
            return
        self.system_supplements.append(chunk)
        self.sync_system_message()

    def sync_system_message(self) -> None:
        """Force ``messages[0]`` to the composed locked system instruction."""
        payload = {"role": "system", "content": self.compose_system_content()}
        if not self.messages:
            self.messages = [payload]
            return
        if self.messages[0].get("role") != "system":
            self.messages.insert(0, payload)
            return
        self.messages[0] = payload

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the conversation to a plain dictionary.

        The *policy* callback is not serialized (functions are not JSON-safe).
        """
        return {
            "messages": list(self.messages),
            "system_supplements": list(self.system_supplements),
            "max_history": self.max_history,
            "max_tool_rounds": self.max_tool_rounds,
            "scope": self.scope or None,
        }

    @classmethod
    def from_dict(cls, agent: "Agent", data: dict) -> "Conversation":
        """Restore a conversation from a dictionary produced by :meth:`to_dict`."""
        conv = cls.__new__(cls)
        conv.agent = agent
        conv.messages = list(data["messages"])
        conv.system_supplements = list(data.get("system_supplements") or [])
        conv.max_history = data.get("max_history")
        conv.max_tool_rounds = max(
            1, int(data.get("max_tool_rounds") or DEFAULT_MAX_TOOL_ROUNDS)
        )
        conv.scope = data.get("scope") or {}
        conv.policy = None
        if not conv.system_supplements and conv.messages:
            first = conv.messages[0]
            if first.get("role") == "system":
                blob = first.get("content") or ""
                if blob.startswith(DEFAULT_SYSTEM_PROMPT):
                    tail = blob[len(DEFAULT_SYSTEM_PROMPT) :].strip()
                    if tail:
                        conv.system_supplements.append(tail)
        conv.sync_system_message()
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
        non_system = [m for m in self.messages if m.get("role") != "system"]
        self.messages = [
            {"role": "system", "content": self.compose_system_content()}
        ] + non_system[-self.max_history :]
        logger.debug("Trimmed context to %d messages", len(self.messages))

    # -- public API --------------------------------------------------------

    def ask(
        self,
        question: str,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Send *question* and return the model's response.

        Parameters
        ----------
        images:
            Optional list of image file paths to include with the question.
        """
        self.sync_system_message()
        msg: dict[str, Any] = {"role": "user", "content": question}
        if images:
            msg["images"] = images
        self.messages.append(msg)
        self.trim_context()
        self.sync_system_message()
        logger.info("User: %s", question[:120])
        self.turn_tool_cache = {}

        scope_token = active_scope.set(self.scope)
        try:
            response = self.complete_turn(self.call_model())
            response = self.apply_output_schema(response)
        finally:
            active_scope.reset(scope_token)

        self.messages.append({"role": "assistant", "content": response.message.content})
        logger.info("Assistant: %s", response.message.content[:120])
        return response

    def ask_stream(
        self,
        question: str,
        images: list[str] | None = None,
    ) -> Iterator[str]:
        """Like :meth:`ask` but yields content chunks as they arrive.

        Each model turn is streamed. Tool calls pause the text stream, run
        synchronously, then the next model turn streams again.
        """
        self.sync_system_message()
        msg: dict[str, Any] = {"role": "user", "content": question}
        if images:
            msg["images"] = images
        self.messages.append(msg)
        self.trim_context()
        self.sync_system_message()
        logger.info("User (stream): %s", question[:120])
        self.turn_tool_cache = {}

        # Snapshot-restore instead of Token.reset(): a streaming generator is
        # resumed by the caller (e.g. Starlette's threadpool) in a different
        # Context on each iteration, so Token.reset() would raise ValueError
        # when the finally block runs outside the Context the token came from.
        previous_scope = active_scope.get()
        active_scope.set(self.scope)
        try:
            yield from self.stream_complete_turn()
        finally:
            active_scope.set(previous_scope)

    async def ask_async(
        self,
        question: str,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Async version of :meth:`ask`.

        Uses the backend's ``chat_async`` method (native async when
        available, ``asyncio.to_thread`` by default).  Tool functions
        are executed via ``asyncio.to_thread`` so they don't block the
        event loop.
        """
        self.sync_system_message()
        msg: dict[str, Any] = {"role": "user", "content": question}
        if images:
            msg["images"] = images
        self.messages.append(msg)
        self.trim_context()
        self.sync_system_message()
        logger.info("User (async): %s", question[:120])
        self.turn_tool_cache = {}

        scope_token = active_scope.set(self.scope)
        try:
            response = await self.complete_turn_async(await self.call_model_async())
            response = await self.apply_output_schema_async(response)
        finally:
            active_scope.reset(scope_token)

        self.messages.append({"role": "assistant", "content": response.message.content})
        logger.info("Assistant (async): %s", response.message.content[:120])
        return response

    # -- private sync helpers ----------------------------------------------

    def call_model(self) -> ChatResponse:
        """Single provider call for the current message list."""
        response = self.agent.chat(messages=self.messages)
        self.agent.fire("response", response)
        return response

    def should_insist(self, response: ChatResponse) -> bool:
        """True when the model narrated a plan instead of calling a tool."""
        if response.message.tool_calls:
            return False
        if not self.agent.functions:
            return False
        if looks_like_finished_answer(response.message.content):
            return False
        return looks_like_plan(response.message.content)

    def complete_turn(self, response: ChatResponse) -> ChatResponse:
        """Run tools. Nudge at most once if the model announced work and stopped."""
        already_insisted = self.should_insist(response)
        if already_insisted:
            response = self.insist_if_plan(response)
        response = self.execute_tool_calls(response)
        if already_insisted:
            return response
        response = self.insist_if_plan(response)
        return self.execute_tool_calls(response)

    def insist_if_plan(self, response: ChatResponse) -> ChatResponse:
        """Re-prompt once if the reply is a plan with no tool calls.

        Used only so "I'll look that up" does not end the turn empty.
        The draft and nudge are not kept in history.
        """
        if not self.should_insist(response):
            return response
        mark = len(self.messages)
        self.messages.append(
            {"role": "assistant", "content": response.message.content}
        )
        self.messages.append({"role": "user", "content": TOOL_NUDGE})
        logger.info("Narration without tools; insisting once")
        retry = self.call_model()
        del self.messages[mark:]
        return retry

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

    def run_one_tool_round(self, response: ChatResponse) -> None:
        """Append the tool request and execute each call. Does not call the model."""
        self.messages.append(
            {
                "role": "assistant",
                "content": response.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in response.message.tool_calls
                ],
            }
        )

        for call in response.message.tool_calls:
            name = call.function.name
            args = call.function.arguments
            self.agent.fire("tool_call", name, args)
            logger.debug("Calling tool %s(%s)", name, args)

            denied = self.check_policy(name, args)
            cache_key = tool_call_fingerprint(name, args)
            if denied:
                result = denied
            elif cache_key in self.turn_tool_cache:
                result = self.turn_tool_cache[cache_key]
            else:
                try:
                    result = self.call_tool(name, args)
                except Exception as exc:
                    result = f"Tool error: {exc}"
                self.turn_tool_cache[cache_key] = result

            self.agent.fire("tool_result", name, result)
            logger.debug("Tool %s → %s", name, result[:200])
            self.messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "tool_call_id": call.id,
                    "content": result,
                }
            )

    def execute_tool_calls(self, response: ChatResponse) -> ChatResponse:
        """Execute tool calls in a loop until the model stops requesting them."""
        rounds = 0
        previous_keys: frozenset[str] | None = None
        while response.message.tool_calls:
            keys = frozenset(
                tool_call_fingerprint(call.function.name, call.function.arguments)
                for call in response.message.tool_calls
            )
            if previous_keys is not None and keys == previous_keys:
                logger.warning("Identical tool calls repeated; stopping tool loop")
                content = (response.message.content or "").strip()
                if not content:
                    content = (
                        "Stopped after too many tool steps. "
                        "Try a narrower question."
                    )
                return ChatResponse(content=content, usage=response.usage)
            if rounds >= self.max_tool_rounds:
                logger.warning(
                    "Stopped after %d tool rounds", self.max_tool_rounds
                )
                fallback = (
                    "Stopped after too many tool steps. "
                    "Try a narrower question."
                )
                content = (response.message.content or "").strip()
                return ChatResponse(content=content or fallback)
            rounds += 1
            previous_keys = keys
            self.run_one_tool_round(response)
            response = self.agent.chat(messages=self.messages)
            self.agent.fire("response", response)

        return response

    def apply_output_schema(self, response: ChatResponse) -> ChatResponse:
        """Re-prompt with a format constraint if the agent defines an output schema.

        When the last turn is already a JSON object (common after tools), skip a
        redundant ``chat(..., format=schema)`` call to save a full model round-trip.
        """
        schema = getattr(self.agent, "output_schema", None)
        if not schema:
            return response
        text = (response.message.content or "").strip()
        if text:
            try:
                if isinstance(json.loads(text), dict):
                    return response
            except json.JSONDecodeError:
                pass
            self.messages.append({"role": "assistant", "content": response.message.content})
        return self.agent.chat(messages=self.messages, format=schema)

    def stream_complete_turn(self) -> Iterator[str]:
        """Stream model turns, run tools between them, persist the final text."""
        response = ChatResponse(content="")
        parts: list[str] = []
        insisted = False
        rounds = 0
        previous_keys: frozenset[str] | None = None

        def take_turn() -> Iterator[str]:
            nonlocal response, parts
            collector: list[ChatResponse] = []
            parts = []
            buffered: list[str] = []
            for chunk in self.agent.stream_chat_turn(self.messages, collector):
                if not chunk:
                    continue
                parts.append(chunk)
                buffered.append(chunk)
            response = (
                collector[-1] if collector else ChatResponse(content="".join(parts))
            )
            self.agent.fire("response", response)
            if not response.message.tool_calls:
                yield from buffered

        while True:
            yield from take_turn()
            if not insisted and self.should_insist(response):
                insisted = True
                mark = len(self.messages)
                self.messages.append(
                    {"role": "assistant", "content": response.message.content}
                )
                self.messages.append({"role": "user", "content": TOOL_NUDGE})
                logger.info("Narration without tools; insisting once")
                yield from take_turn()
                del self.messages[mark:]

            if response.message.tool_calls:
                keys = frozenset(
                    tool_call_fingerprint(
                        call.function.name, call.function.arguments
                    )
                    for call in response.message.tool_calls
                )
                if previous_keys is not None and keys == previous_keys:
                    logger.warning("Identical tool calls repeated; stopping tool loop")
                    content = "".join(parts).strip()
                    if content:
                        yield content
                        self.messages.append({"role": "assistant", "content": content})
                    return
                if rounds >= self.max_tool_rounds:
                    logger.warning(
                        "Stopped after %d tool rounds", self.max_tool_rounds
                    )
                    content = "".join(parts).strip()
                    if not content:
                        fallback = (
                            "Stopped after too many tool steps. "
                            "Try a narrower question."
                        )
                        yield fallback
                        self.messages.append(
                            {"role": "assistant", "content": fallback}
                        )
                    else:
                        self.messages.append(
                            {"role": "assistant", "content": "".join(parts)}
                        )
                    return
                self.run_one_tool_round(response)
                previous_keys = keys
                rounds += 1
                continue

            content = "".join(parts) or (response.message.content or "")
            boxed = ChatResponse(
                content=content,
                usage=response.usage,
            )
            boxed = self.apply_output_schema(boxed)
            extra = boxed.message.content or ""
            if extra and extra != content:
                if extra.startswith(content):
                    tail = extra[len(content) :]
                    if tail:
                        yield tail
                else:
                    yield extra
            self.messages.append(
                {"role": "assistant", "content": boxed.message.content}
            )
            logger.info("Assistant (stream): %s", (boxed.message.content or "")[:120])
            return

    def stream_final(self) -> Iterator[str]:
        """Stream the model's response from the current message state."""
        schema = getattr(self.agent, "output_schema", None)
        if schema:
            response = self.agent.chat(messages=self.messages, format=schema)
            self.messages.append(
                {"role": "assistant", "content": response.message.content}
            )
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

    async def call_model_async(self) -> ChatResponse:
        """Async single provider call."""
        response = await self.agent.chat_async(messages=self.messages)
        self.agent.fire("response", response)
        return response

    async def insist_if_plan_async(self, response: ChatResponse) -> ChatResponse:
        """Async twin of :meth:`insist_if_plan`."""
        if not self.should_insist(response):
            return response
        mark = len(self.messages)
        self.messages.append(
            {"role": "assistant", "content": response.message.content}
        )
        self.messages.append({"role": "user", "content": TOOL_NUDGE})
        logger.info("Narration without tools; insisting once")
        retry = await self.call_model_async()
        del self.messages[mark:]
        return retry

    async def complete_turn_async(self, response: ChatResponse) -> ChatResponse:
        """Async twin of :meth:`complete_turn`."""
        already_insisted = self.should_insist(response)
        if already_insisted:
            response = await self.insist_if_plan_async(response)
        response = await self.execute_tool_calls_async(response)
        if already_insisted:
            return response
        response = await self.insist_if_plan_async(response)
        return await self.execute_tool_calls_async(response)

    async def execute_tool_calls_async(self, response: ChatResponse) -> ChatResponse:
        rounds = 0
        previous_keys: frozenset[str] | None = None
        while response.message.tool_calls:
            keys = frozenset(
                tool_call_fingerprint(call.function.name, call.function.arguments)
                for call in response.message.tool_calls
            )
            if previous_keys is not None and keys == previous_keys:
                logger.warning("Identical tool calls repeated; stopping tool loop")
                content = (response.message.content or "").strip()
                if not content:
                    content = (
                        "Stopped after too many tool steps. "
                        "Try a narrower question."
                    )
                return ChatResponse(content=content, usage=response.usage)
            if rounds >= self.max_tool_rounds:
                logger.warning(
                    "Stopped after %d tool rounds", self.max_tool_rounds
                )
                fallback = (
                    "Stopped after too many tool steps. "
                    "Try a narrower question."
                )
                content = (response.message.content or "").strip()
                return ChatResponse(content=content or fallback)
            rounds += 1
            previous_keys = keys
            self.run_one_tool_round(response)
            response = await self.call_model_async()

        return response

    async def apply_output_schema_async(self, response: ChatResponse) -> ChatResponse:
        """Async twin of :meth:`apply_output_schema`."""
        schema = getattr(self.agent, "output_schema", None)
        if not schema:
            return response
        text = (response.message.content or "").strip()
        if text:
            try:
                if isinstance(json.loads(text), dict):
                    return response
            except json.JSONDecodeError:
                pass
            self.messages.append({"role": "assistant", "content": response.message.content})
        return await self.agent.chat_async(messages=self.messages, format=schema)


# -- Agent -----------------------------------------------------------------


class Agent(ABC):
    """Base class for all agents.

    **Single workflow (all vendors):** :class:`Conversation` owns prompting,
    tool execution, and output formatting. A subclass only
    implements :meth:`chat` / :meth:`chat_stream` — transport plus parsing into
    :class:`ChatResponse`. Message shaping for each API lives in that backend's
    ``prepare_*_messages`` (or equivalent); tool argument coercion to plain dicts
    is centralized in :func:`normalize_tool_arguments` via :class:`ChatResponse`.

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

    system_prompt: str | None = None
    temperature: float = 0

    # Registry: lets tools declared near the methods they expose reach the
    # agent via :meth:`by_name` without importing the concrete instance.
    # Populated when an agent is constructed with ``name=``.
    registry: dict[str, "Agent"] = {}

    # Registrations queued against a name that does not yet exist in the
    # registry.  They are applied when an agent with that name is constructed.
    pending_uses: dict[str, list[tuple[Callable, dict]]] = {}

    def __init__(
        self,
        *args: Any,
        name: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.functions: dict[str, Function] = {}
        self.conversations: list[Conversation] = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.name = name
        self.hooks: dict[str, list[Callable]] = {
            "tool_call": [],
            "tool_result": [],
            "response": [],
        }
        if name is not None:
            if name in Agent.registry:
                raise ValueError(f"Agent with name {name!r} is already registered")
            Agent.registry[name] = self
            for target, use_kwargs in Agent.pending_uses.pop(name, []):
                self.add_tool(target, **use_kwargs)

    @classmethod
    def by_name(cls, name: str) -> "AgentRef":
        """Return a lazy handle that forwards :meth:`uses` to the named agent.

        Adapter: lets model modules declare their tools with
        ``@Agent.by_name("CRMAgent").uses(...)`` without importing the agent
        instance itself.  If the agent has not been constructed yet, the
        registration is queued and flushed on construction; this sidesteps
        import-ordering problems between the agent module and the modules
        that contribute tools to it.
        """
        return AgentRef(name)

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
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
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
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        raise last_exc  # pragma: no cover

    # -- tool registration -------------------------------------------------

    def add_tool(
        self,
        func: Callable,
        *,
        serializer: Callable[[Any], Any] | None = None,
        description: str | None = None,
        name: str | None = None,
    ) -> None:
        """Wrap *func* as a :class:`Function` and store it on this agent."""
        f = Function(func, serializer=serializer, description=description, name=name)
        self.functions[f.name] = f

    def uses(
        self,
        func: Callable | None = None,
        *,
        serializer: Callable[[Any], Any] | None = None,
        description: str | None = None,
        name: str | None = None,
    ) -> Callable:
        """Register *func* as a tool the agent can invoke.

        Accepts all three decorator shapes (bare, parameterised, direct call).
        Also bound on :class:`AgentRef` so ``@Agent.by_name(...).uses(...)``
        runs this exact body against a lazy handle — the only thing that
        differs between the two code paths is :meth:`add_tool`.

        Any parameter of *func* whose name matches a key in the conversation
        :data:`active_scope` is auto-injected at call time and hidden from
        the LLM-visible tool schema, so the model can't widen access by
        passing its own ``enterprise_id`` / ``user_id`` / etc.
        """
        kwargs = {"serializer": serializer, "description": description, "name": name}

        def apply(target: Callable) -> Callable:
            self.add_tool(target, **kwargs)

            @wraps(target)
            def passthrough(*args: Any, **kw: Any) -> Any:
                return target(*args, **kw)

            return passthrough

        return apply if func is None else apply(func)

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
        if response.message.content:
            yield response.message.content

    def stream_chat_turn(
        self,
        messages: list[dict],
        collector: list | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield text deltas and store the completed :class:`ChatResponse`.

        *collector* receives the parsed response so a streaming caller can
        see tool calls without storing state on the shared agent instance.
        Backends that stream natively should override this (not only
        :meth:`chat_stream`) so tool_use survives the stream.
        """
        if not self.functions:
            parts: list[str] = []
            for chunk in self.chat_stream(messages, **kwargs):
                if not chunk:
                    continue
                parts.append(chunk)
                yield chunk
            if collector is not None:
                collector.append(ChatResponse(content="".join(parts)))
            return
        response = self.chat(messages, **kwargs)
        if collector is not None:
            collector.append(response)
        if response.message.content:
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
        prompt: str | None = None,
        max_history: int | None = None,
        scope: dict | None = None,
        policy: Callable[..., bool] | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ) -> Conversation:
        """Begin a new conversation.

        The system prompt is always anchored to :data:`DEFAULT_SYSTEM_PROMPT`.
        Mixins add context via :meth:`Conversation.append_system_supplement`
        so the locked policy cannot be replaced by accident.

        Parameters
        ----------
        prompt:
            Optional text added as the first user message (not the
            system prompt).
        scope:
            Identity context forwarded to row-level security filters
            and the *policy* callback (e.g. ``{"current_user": 5}``).
        policy:
            Optional ``(tool_name, arguments, scope) -> bool`` gate
            evaluated before every tool execution.
        """
        conv = Conversation(
            self,
            max_history=max_history,
            scope=scope,
            policy=policy,
            max_tool_rounds=max_tool_rounds,
        )
        if prompt:
            conv.messages.append({"role": "user", "content": prompt})
        self.conversations.append(conv)
        return conv


class AgentRef:
    """Lazy handle to an :class:`Agent`, resolved by name at registration time.

    Reuses :meth:`Agent.uses` wholesale — only :meth:`add_tool` differs,
    forwarding to the live agent if it exists or queuing on
    :attr:`Agent.pending_uses` otherwise.
    """

    __slots__ = ("target_name",)

    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

    def add_tool(self, func: Callable, **kwargs: Any) -> None:
        agent = Agent.registry.get(self.target_name)
        if agent is not None:
            agent.add_tool(func, **kwargs)
        else:
            Agent.pending_uses.setdefault(self.target_name, []).append((func, kwargs))

    uses = Agent.uses
