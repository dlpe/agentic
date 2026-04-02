"""Core abstractions for building AI agents with tool-calling capabilities."""

from abc import ABC, abstractmethod
from functools import wraps
import inspect
from typing import Any, Callable

__all__ = ["Function", "Conversation", "Agent"]

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful agent. "
    "If you need a tool to answer the question, ALWAYS call tools immediately. "
    "NEVER describe what you intend to do or ask for confirmation — just call the tool. "
    "If you can provide the answer directly, do so. "
    "If your answer doesn't call a tool, it must be the final answer. "
    "No follow-up questions are allowed."
)

_TOOL_NUDGE = "Don't describe what you will do. Call the appropriate tool now."


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
    """

    def __init__(self, agent: "Agent", system: str) -> None:
        self.agent = agent
        self.messages: list[dict] = [{"role": "system", "content": system}]

    # -- public API --------------------------------------------------------

    def ask(self, question: str, max_retries: int = 3) -> Any:
        """Send *question* and return the model's response."""
        self.messages.append({"role": "user", "content": question})

        response = self._prompt_until_actionable(max_retries)
        response = self._execute_tool_calls(response)
        response = self._apply_output_schema(response)

        self.messages.append({"role": "assistant", "content": response.message.content})
        return response

    # -- private helpers ---------------------------------------------------

    def _prompt_until_actionable(self, max_retries: int) -> Any:
        """Prompt the model, retrying with a nudge if it narrates instead of acting."""
        for attempt in range(max_retries):
            response = self.agent.chat(messages=self.messages)

            if response.message.tool_calls:
                return response

            should_retry = (
                attempt < max_retries - 1
                and self.agent.functions
                and not self._has_prior_tool_result()
            )
            if should_retry:
                self.messages.append({"role": "assistant", "content": response.message.content})
                self.messages.append({"role": "user", "content": _TOOL_NUDGE})
            else:
                return response

        return response  # pragma: no cover — loop always returns

    def _execute_tool_calls(self, response: Any) -> Any:
        """Execute tool calls in a loop until the model stops requesting them."""
        while response.message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.message.content})

            for call in response.message.tool_calls:
                try:
                    result = str(
                        self.agent.functions[call.function.name](
                            **call.function.arguments
                        )
                    )
                except Exception as exc:
                    result = f"Tool error: {exc}"

                self.messages.append({
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": result,
                })

            response = self.agent.chat(messages=self.messages)

        return response

    def _apply_output_schema(self, response: Any) -> Any:
        """Re-prompt with a format constraint if the agent defines an output schema."""
        schema = getattr(self.agent, "output_schema", None)
        if schema:
            return self.agent.chat(messages=self.messages, format=schema)
        return response

    def _has_prior_tool_result(self) -> bool:
        """True if a tool result exists after the most recent user message."""
        for msg in reversed(self.messages):
            if msg["role"] == "tool":
                return True
            if msg["role"] == "user":
                return False
        return False


class Agent(ABC):
    """Base class for all agents.

    Subclasses must implement :meth:`chat`.  Optionally register tools
    with the :meth:`uses` decorator and start conversations with
    :meth:`start_conversation`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.functions: dict[str, Function] = {}
        self.conversations: list[Conversation] = []

    def uses(self, func: Callable) -> Callable:
        """Register *func* as a tool the agent can invoke."""
        f = Function(func)
        self.functions[f.name] = f

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapped

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs: Any) -> Any:
        """Send *messages* to the model and return its response."""
        ...

    def start_conversation(self, system: str = _DEFAULT_SYSTEM_PROMPT) -> Conversation:
        """Begin a new conversation with the given system prompt."""
        conv = Conversation(self, system)
        self.conversations.append(conv)
        return conv
