"""Tests for streaming responses (ask_stream / chat_stream)."""

from pygentix.testing import MockAgent


class TestStreaming:
    def test_ask_stream_yields_content(self):
        agent = MockAgent(responses=["the quick brown fox"])
        conv = agent.start_conversation()
        chunks = list(conv.ask_stream("tell me something"))
        full = "".join(chunks)
        assert full == "the quick brown fox"

    def test_stream_records_in_history(self):
        agent = MockAgent(responses=["streamed answer"])
        conv = agent.start_conversation()
        list(conv.ask_stream("question"))
        assert conv.messages[-1]["role"] == "assistant"
        assert conv.messages[-1]["content"] == "streamed answer"

    def test_stream_without_tools(self):
        agent = MockAgent(responses=["hello world"])
        conv = agent.start_conversation()
        chunks = list(conv.ask_stream("hi"))
        assert len(chunks) == 2  # "hello " and "world"

    def test_stream_with_tools_yields_final(self):
        agent = MockAgent(responses=[
            {"tool_calls": [{"name": "calc", "arguments": {"x": 5}}]},
            "result is 25",
        ])

        @agent.uses
        def calc(x: int) -> int:
            """Square a number."""
            return x * x

        conv = agent.start_conversation()
        chunks = list(conv.ask_stream("square 5"))
        full = "".join(chunks)
        assert "25" in full
        assert len(chunks) >= 2
        assert any(role["role"] == "tool" for role in conv.messages)

    def test_chat_stream_yields_all_content(self):
        """MockAgent.chat_stream splits into words; content is preserved."""
        agent = MockAgent(responses=["hello world"])
        chunks = list(agent.chat_stream([{"role": "user", "content": "hi"}]))
        assert "".join(chunks) == "hello world"

    def test_stream_hides_text_from_tool_turns(self):
        agent = MockAgent(
            responses=[
                {
                    "content": "Working on it, I'll fetch sales next.",
                    "tool_calls": [{"name": "calc", "arguments": {"x": 2}}],
                },
                "result is 4",
            ]
        )

        @agent.uses
        def calc(x: int) -> int:
            """Square a number."""
            return x * x

        conv = agent.start_conversation()
        full = "".join(conv.ask_stream("square 2"))
        assert "Working on it" not in full
        assert "result is 4" in full

    def test_stream_does_not_reexport_after_let_me_know(self):
        calls = {"n": 0}
        agent = MockAgent(
            responses=[
                {"tool_calls": [{"name": "query_product_sales", "arguments": {}}]},
                (
                    "Your Excel file with all 109 sold items is ready! "
                    "Let me know if you'd like to filter by client."
                ),
                {"tool_calls": [{"name": "query_product_sales", "arguments": {}}]},
                "looped again",
            ]
        )

        @agent.uses
        def query_product_sales() -> str:
            """Export sold items."""
            calls["n"] += 1
            return "download_url=/static/chat_files/sold.xlsx"

        conv = agent.start_conversation()
        full = "".join(conv.ask_stream("all items sold"))
        assert calls["n"] == 1
        assert "looped again" not in full
        assert "109 sold items" in full

    def test_multiple_streamed_turns(self):
        agent = MockAgent(responses=["one", "two", "three"])
        conv = agent.start_conversation()
        r1 = "".join(conv.ask_stream("a"))
        r2 = "".join(conv.ask_stream("b"))
        r3 = "".join(conv.ask_stream("c"))
        assert r1 == "one"
        assert r2 == "two"
        assert r3 == "three"
