"""Tests for pygentix.testing.MockAgent."""

from pygentix.testing import MockAgent


class TestMockAgent:
    def test_returns_responses_in_order(self):
        agent = MockAgent(responses=["first", "second", "third"])
        conv = agent.start_conversation()
        assert conv.ask("a").message.content == "first"
        assert conv.ask("b").message.content == "second"
        assert conv.ask("c").message.content == "third"

    def test_cycles_when_exhausted(self):
        agent = MockAgent(responses=["alpha", "beta"])
        conv = agent.start_conversation()
        conv.ask("1")
        conv.ask("2")
        assert conv.ask("3").message.content == "alpha"

    def test_dict_response_with_content(self):
        agent = MockAgent(responses=[{"content": "from dict"}])
        conv = agent.start_conversation()
        assert conv.ask("x").message.content == "from dict"

    def test_dict_response_with_usage(self):
        agent = MockAgent(responses=[{
            "content": "ok",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }])
        conv = agent.start_conversation()
        resp = conv.ask("x")
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.usage.total_tokens == 15

    def test_tool_simulation(self):
        agent = MockAgent(responses=[
            {"tool_calls": [{"name": "greet", "arguments": {"name": "world"}}]},
            "Hello, world!",
        ])

        @agent.uses
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}"

        conv = agent.start_conversation()
        resp = conv.ask("greet the world")
        assert resp.message.content == "Hello, world!"

    def test_stream_yields_words(self):
        agent = MockAgent(responses=["hello beautiful world"])
        conv = agent.start_conversation()
        chunks = list(conv.ask_stream("hi"))
        assert "".join(chunks) == "hello beautiful world"
        assert len(chunks) == 3

    def test_default_empty_response(self):
        agent = MockAgent()
        conv = agent.start_conversation()
        assert conv.ask("anything").message.content == ""
