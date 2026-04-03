"""Tests for conversation serialization (save/load)."""

import json

from pygentix.core import Conversation
from pygentix.testing import MockAgent


class TestPersistence:
    def test_to_dict_contains_messages(self):
        agent = MockAgent(responses=["pong"])
        conv = agent.start_conversation()
        conv.ask("ping")
        data = conv.to_dict()
        assert "messages" in data
        assert len(data["messages"]) == 3  # system + user + assistant

    def test_round_trip_dict(self):
        agent = MockAgent(responses=["one", "two"])
        conv = agent.start_conversation()
        conv.ask("first")

        data = conv.to_dict()
        restored = Conversation.from_dict(agent, data)

        assert restored.messages == conv.messages
        assert restored.max_history == conv.max_history

    def test_round_trip_json(self):
        agent = MockAgent(responses=["hello"])
        conv = agent.start_conversation()
        conv.ask("hi")

        json_str = conv.to_json()
        assert isinstance(json_str, str)
        json.loads(json_str)  # must be valid JSON

        restored = Conversation.from_json(agent, json_str)
        assert restored.messages == conv.messages

    def test_max_history_preserved(self):
        agent = MockAgent(responses=["x"])
        conv = agent.start_conversation(max_history=10)
        conv.ask("a")

        data = conv.to_dict()
        assert data["max_history"] == 10

        restored = Conversation.from_dict(agent, data)
        assert restored.max_history == 10

    def test_from_dict_restores_agent(self):
        agent = MockAgent(responses=["reply"])
        conv = agent.start_conversation()
        conv.ask("q")

        restored = Conversation.from_dict(agent, conv.to_dict())
        assert restored.agent is agent

    def test_restored_conversation_can_continue(self):
        agent = MockAgent(responses=["first", "second", "third"])
        conv = agent.start_conversation()
        conv.ask("1")

        restored = Conversation.from_json(agent, conv.to_json())
        resp = restored.ask("2")
        assert resp.message.content in ("first", "second", "third")
        assert len(restored.messages) == 5  # system + q1 + a1 + q2 + a2
