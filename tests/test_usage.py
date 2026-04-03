"""Tests for token usage tracking."""

from pygentix.core import ChatResponse, Usage
from pygentix.testing import MockAgent


class TestUsage:
    def test_default_values(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_custom_values(self):
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.total_tokens == 150

    def test_bool_false_when_empty(self):
        assert not Usage()

    def test_bool_true_when_populated(self):
        assert Usage(total_tokens=1)

    def test_repr(self):
        u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        r = repr(u)
        assert "10" in r
        assert "5" in r
        assert "15" in r

    def test_chat_response_default_usage(self):
        resp = ChatResponse(content="hello")
        assert resp.usage is not None
        assert resp.usage.total_tokens == 0

    def test_chat_response_with_usage(self):
        u = Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        resp = ChatResponse(content="hello", usage=u)
        assert resp.usage.prompt_tokens == 20
        assert resp.usage.total_tokens == 30

    def test_mock_agent_with_usage(self):
        agent = MockAgent(responses=[{
            "content": "ok",
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        }])
        conv = agent.start_conversation()
        resp = conv.ask("anything")
        assert resp.usage.prompt_tokens == 50
        assert resp.usage.total_tokens == 75
