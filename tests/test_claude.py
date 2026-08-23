"""Tests for Anthropic request shaping and prompt-cache breakpoints."""

from pygentix.claude import (
    STABLE_PROMPT_CACHE,
    TURN_PROMPT_CACHE,
    build_claude_request,
    cached_system_blocks,
    convert_tools_for_claude,
    mark_messages_cached,
    mark_tools_cached,
)
from pygentix.core import Function


def add_fn(n: int) -> int:
    """Add helper used as a tool."""
    return n + 1


class TestPromptCacheBlocks:
    def test_system_block_has_one_hour_cache(self):
        blocks = cached_system_blocks("You are a CRM assistant.")
        assert isinstance(blocks, list)
        assert blocks[0]["cache_control"] == STABLE_PROMPT_CACHE
        assert blocks[0]["text"] == "You are a CRM assistant."

    def test_empty_system_stays_a_string(self):
        assert cached_system_blocks("  ") == "  "

    def test_last_tool_is_marked(self):
        tools = [
            {"name": "a", "description": "", "input_schema": {}},
            {"name": "b", "description": "", "input_schema": {}},
        ]
        marked = mark_tools_cached(tools)
        assert marked is not None
        assert "cache_control" not in marked[0]
        assert marked[-1]["cache_control"] == STABLE_PROMPT_CACHE

    def test_string_message_becomes_cached_text_block(self):
        out = mark_messages_cached([{"role": "user", "content": "hello"}])
        assert out[-1]["content"] == [
            {"type": "text", "text": "hello", "cache_control": TURN_PROMPT_CACHE}
        ]

    def test_last_tool_result_block_is_marked(self):
        out = mark_messages_cached(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "c1", "content": "ok"}
                    ],
                }
            ]
        )
        assert out[-1]["content"][-1]["cache_control"] == TURN_PROMPT_CACHE


class TestBuildClaudeRequest:
    def test_request_includes_all_breakpoints(self):
        functions = {"add_fn": Function(add_fn)}
        payload = build_claude_request(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
            functions,
        )
        assert payload["system"][0]["cache_control"] == STABLE_PROMPT_CACHE
        assert payload["tools"][-1]["cache_control"] == STABLE_PROMPT_CACHE
        assert payload["messages"][-1]["content"][-1]["cache_control"] == (
            TURN_PROMPT_CACHE
        )

    def test_cache_can_be_disabled(self):
        functions = {"add_fn": Function(add_fn)}
        payload = build_claude_request(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
            functions,
            prompt_cache=False,
        )
        assert payload["system"] == "Be brief."
        assert isinstance(payload["messages"][-1]["content"], str)
        assert "cache_control" not in payload["tools"][-1]

    def test_convert_tools_keeps_schema(self):
        tools = convert_tools_for_claude({"add_fn": Function(add_fn)})
        assert tools is not None
        assert tools[0]["name"] == "add_fn"
