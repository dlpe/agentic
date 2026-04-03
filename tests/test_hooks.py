"""Tests for agent event hooks."""

import logging

from pygentix.testing import MockAgent


class TestHooks:
    def test_register_valid_event(self):
        agent = MockAgent(responses=["ok"])
        calls = []
        agent.on("response", lambda r: calls.append(r))
        conv = agent.start_conversation()
        conv.ask("hi")
        assert len(calls) >= 1

    def test_invalid_event_raises(self):
        agent = MockAgent()
        try:
            agent.on("nonexistent", lambda: None)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)

    def test_tool_call_hook_fires(self):
        agent = MockAgent(responses=[
            {"tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 2}}]},
            "3",
        ])

        @agent.uses
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        calls = []
        agent.on("tool_call", lambda name, args: calls.append((name, args)))

        conv = agent.start_conversation()
        conv.ask("add 1 + 2")
        assert len(calls) == 1
        assert calls[0] == ("add", {"a": 1, "b": 2})

    def test_tool_result_hook_fires(self):
        agent = MockAgent(responses=[
            {"tool_calls": [{"name": "greet", "arguments": {"name": "world"}}]},
            "done",
        ])

        @agent.uses
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello {name}"

        results = []
        agent.on("tool_result", lambda name, result: results.append((name, result)))

        conv = agent.start_conversation()
        conv.ask("greet")
        assert len(results) == 1
        assert results[0] == ("greet", "Hello world")

    def test_response_hook_receives_chat_response(self):
        agent = MockAgent(responses=["hello"])
        responses = []
        agent.on("response", lambda r: responses.append(r.message.content))

        conv = agent.start_conversation()
        conv.ask("hi")
        assert "hello" in responses

    def test_hook_exception_does_not_crash(self, caplog):
        agent = MockAgent(responses=["ok"])

        def bad_hook(r):
            raise RuntimeError("hook broke")

        agent.on("response", bad_hook)

        conv = agent.start_conversation()
        with caplog.at_level(logging.ERROR, logger="pygentix"):
            resp = conv.ask("hi")
        assert resp.message.content == "ok"
        assert "hook broke" in caplog.text

    def test_multiple_hooks_on_same_event(self):
        agent = MockAgent(responses=["result"])
        log1, log2 = [], []
        agent.on("response", lambda r: log1.append(1))
        agent.on("response", lambda r: log2.append(2))

        conv = agent.start_conversation()
        conv.ask("x")
        assert len(log1) >= 1
        assert len(log2) >= 1
