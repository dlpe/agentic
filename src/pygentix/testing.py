"""Testing utilities — a mock agent for unit-testing application code."""

from typing import Any, Iterator

from .core import Agent, ChatResponse, Usage

__all__ = ["MockAgent"]


class MockAgent(Agent):
    """A fake LLM backend that returns pre-configured responses.

    Useful for unit-testing application code that depends on pygentix
    without making real LLM calls.

    Parameters
    ----------
    responses:
        A list of responses to return in order.  Each entry can be:

        * A **string** — returned as the ``content`` of a :class:`ChatResponse`.
        * A **dict** with optional keys ``content``, ``tool_calls``, ``usage``.

        When all responses are consumed the agent cycles back to the start.

    Example::

        from pygentix.testing import MockAgent

        agent = MockAgent(responses=["Hello!", "Goodbye!"])
        conv = agent.start_conversation()
        r1 = conv.ask("Hi")     # → "Hello!"
        r2 = conv.ask("Bye")    # → "Goodbye!"
    """

    def __init__(
        self,
        responses: list[str | dict] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.responses: list[str | dict] = list(responses or [""])
        self.index = 0

    def next_response(self) -> ChatResponse:
        entry = self.responses[self.index % len(self.responses)]
        self.index += 1

        if isinstance(entry, str):
            return ChatResponse(content=entry)

        usage = None
        if entry.get("usage"):
            u = entry["usage"]
            usage = Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )

        return ChatResponse(
            content=entry.get("content", ""),
            tool_calls=entry.get("tool_calls"),
            usage=usage,
        )

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        return self.next_response()

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """Yield the next response one word at a time to simulate streaming."""
        response = self.next_response()
        words = response.message.content.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "
