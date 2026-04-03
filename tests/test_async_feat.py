"""Tests for async conversation support (ask_async / chat_async)."""

import asyncio

import pytest

from pygentix.testing import MockAgent


class TestAsync:
    def test_ask_async_returns_response(self):
        agent = MockAgent(responses=["async hello"])
        conv = agent.start_conversation()
        resp = asyncio.run(conv.ask_async("hi"))
        assert resp.message.content == "async hello"

    def test_ask_async_records_history(self):
        agent = MockAgent(responses=["reply"])
        conv = agent.start_conversation()
        asyncio.run(conv.ask_async("question"))
        assert conv.messages[-1]["role"] == "assistant"
        assert conv.messages[-1]["content"] == "reply"

    def test_ask_async_with_tools(self):
        agent = MockAgent(responses=[
            {"tool_calls": [{"name": "double", "arguments": {"n": 7}}]},
            "14",
        ])

        @agent.uses
        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        conv = agent.start_conversation()
        resp = asyncio.run(conv.ask_async("double 7"))
        assert resp.message.content == "14"

    def test_chat_async_default_thread_pool(self):
        agent = MockAgent(responses=["threaded"])
        resp = asyncio.run(agent.chat_async([{"role": "user", "content": "hi"}]))
        assert resp.message.content == "threaded"

    def test_multiple_async_turns(self):
        agent = MockAgent(responses=["a", "b", "c"])
        conv = agent.start_conversation()

        async def multi():
            r1 = await conv.ask_async("1")
            r2 = await conv.ask_async("2")
            r3 = await conv.ask_async("3")
            return r1, r2, r3

        r1, r2, r3 = asyncio.run(multi())
        assert r1.message.content == "a"
        assert r2.message.content == "b"
        assert r3.message.content == "c"
