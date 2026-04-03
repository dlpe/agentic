"""Tests for built-in structured logging."""

import logging

from pygentix.testing import MockAgent


class TestLogging:
    def test_logs_user_message(self, caplog):
        agent = MockAgent(responses=["ok"])
        conv = agent.start_conversation()
        with caplog.at_level(logging.INFO, logger="pygentix"):
            conv.ask("What is the meaning of life?")
        assert "What is the meaning of life?" in caplog.text

    def test_logs_assistant_response(self, caplog):
        agent = MockAgent(responses=["42"])
        conv = agent.start_conversation()
        with caplog.at_level(logging.INFO, logger="pygentix"):
            conv.ask("question")
        assert "42" in caplog.text

    def test_logs_tool_execution(self, caplog):
        agent = MockAgent(responses=[
            {"tool_calls": [{"name": "calc", "arguments": {"x": 5}}]},
            "25",
        ])

        @agent.uses
        def calc(x: int) -> int:
            """Square it."""
            return x * x

        conv = agent.start_conversation()
        with caplog.at_level(logging.DEBUG, logger="pygentix"):
            conv.ask("calculate")
        assert "calc" in caplog.text

    def test_logs_context_trim(self, caplog):
        agent = MockAgent(responses=["r"] * 10)
        conv = agent.start_conversation(max_history=2)
        with caplog.at_level(logging.DEBUG, logger="pygentix"):
            for i in range(5):
                conv.ask(f"q{i}")
        assert "Trimmed" in caplog.text

    def test_log_level_filtering(self, caplog):
        agent = MockAgent(responses=["ok"])
        conv = agent.start_conversation()
        with caplog.at_level(logging.WARNING, logger="pygentix"):
            conv.ask("hi")
        assert "User:" not in caplog.text
