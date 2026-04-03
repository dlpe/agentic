"""Tests for context window management (max_history)."""

from pygentix.testing import MockAgent


class TestContextManagement:
    def test_no_trim_when_under_limit(self):
        agent = MockAgent(responses=["a", "b"])
        conv = agent.start_conversation(max_history=10)
        conv.ask("q1")
        conv.ask("q2")
        assert len(conv.messages) == 5  # system + 2*(user + assistant)

    def test_trims_when_over_limit(self):
        agent = MockAgent(responses=["r"] * 20)
        conv = agent.start_conversation(max_history=4)
        for i in range(10):
            conv.ask(f"question {i}")
        # system + max_history kept after trim + 1 assistant appended after
        assert len(conv.messages) <= 4 + 2

    def test_system_prompt_always_preserved(self):
        agent = MockAgent(responses=["r"] * 10)
        conv = agent.start_conversation(max_history=2)
        for i in range(5):
            conv.ask(f"q{i}")
        assert conv.messages[0]["role"] == "system"

    def test_max_history_none_no_trim(self):
        agent = MockAgent(responses=["r"] * 20)
        conv = agent.start_conversation()  # max_history=None by default
        for i in range(15):
            conv.ask(f"q{i}")
        assert len(conv.messages) == 31  # system + 15*(user + assistant)

    def test_trim_preserves_recent_content(self):
        agent = MockAgent(responses=["r"] * 10)
        conv = agent.start_conversation(max_history=2)
        conv.ask("old question")
        conv.ask("recent question")
        conv.ask("latest question")
        user_msgs = [m["content"] for m in conv.messages if m["role"] == "user"]
        assert "latest question" in user_msgs
        assert "old question" not in user_msgs
