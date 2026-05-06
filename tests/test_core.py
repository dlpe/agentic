"""Tests for pygentix.core — Function, Conversation, and Agent."""

from unittest.mock import MagicMock

from pygentix.core import ChatResponse, Conversation, Function


# -- Function --------------------------------------------------------------


class TestFunction:
    def test_call_delegates_to_wrapped_function(self):
        def add(a: int, b: int) -> int:
            return a + b

        f = Function(add)
        assert f(2, 3) == 5

    def test_name_reflects_wrapped_function(self):
        def my_tool():
            return None

        assert Function(my_tool).name == "my_tool"

    def test_parameters_match_signature(self):
        def greet(name: str, excited: bool = False):
            return f"{name}{'!' if excited else ''}"

        params = Function(greet).parameters
        assert "name" in params
        assert "excited" in params

    def test_repr_is_readable(self):
        def example():
            return None

        assert repr(Function(example)) == "Function(example)"


# -- Conversation.has_prior_tool_result ------------------------------------


class TestHasPriorToolResult:
    def make_conversation(self):
        mock_agent = MagicMock()
        mock_agent.functions = {"some_tool": lambda: None}
        return Conversation(mock_agent)

    def test_false_when_no_messages(self):
        conv = self.make_conversation()
        assert conv.has_prior_tool_result() is False

    def test_false_after_user_message(self):
        conv = self.make_conversation()
        conv.messages.append({"role": "user", "content": "hello"})
        assert conv.has_prior_tool_result() is False

    def test_true_after_tool_result(self):
        conv = self.make_conversation()
        conv.messages.append({"role": "user", "content": "hello"})
        conv.messages.append({"role": "tool", "content": "result"})
        assert conv.has_prior_tool_result() is True

    def test_false_after_new_user_message_following_tool(self):
        conv = self.make_conversation()
        conv.messages.append({"role": "user", "content": "first"})
        conv.messages.append({"role": "tool", "content": "result"})
        conv.messages.append({"role": "user", "content": "second"})
        assert conv.has_prior_tool_result() is False


# -- Conversation single-shot prompt ---------------------------------------


class TestConversationPrompting:
    def make_mock_response(self, content="", tool_calls=None):
        resp = MagicMock()
        resp.message.content = content
        resp.message.tool_calls = tool_calls
        return resp

    def test_no_retry_when_tool_called(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        tool_call = MagicMock()
        tool_call.function.name = "tool"
        tool_call.function.arguments = {}

        mock_agent.chat.side_effect = [
            self.make_mock_response(tool_calls=[tool_call]),
            self.make_mock_response(content="done"),
        ]
        mock_agent.functions = {"tool": lambda: "ok"}

        conv = Conversation(mock_agent)
        conv.ask("do something")
        assert mock_agent.chat.call_count == 2

    def test_no_retry_when_no_functions(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        mock_agent.functions = {}
        mock_agent.chat.return_value = self.make_mock_response(content="answer")

        conv = Conversation(mock_agent)
        resp = conv.ask("question")
        assert resp.message.content == "answer"
        assert mock_agent.chat.call_count == 1

    def test_single_prompt_when_model_narrates_without_tools(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        mock_agent.functions = {"tool": lambda: "ok"}
        mock_agent.chat.return_value = self.make_mock_response(content="I will...")

        conv = Conversation(mock_agent)
        conv.ask("do it")
        assert mock_agent.chat.call_count == 1
        assert not any(
            "Don't describe" in str(m.get("content", "")) for m in conv.messages
        )

    def test_single_prompt_after_prior_tool_messages(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        mock_agent.functions = {"tool": lambda: "ok"}
        mock_agent.chat.return_value = self.make_mock_response(content="Let me think...")

        conv = Conversation(mock_agent)
        conv.messages.append({"role": "user", "content": "first"})
        conv.messages.append({"role": "assistant", "content": ""})
        conv.messages.append(
            {"role": "tool", "tool_name": "tool", "content": "prior data"}
        )

        conv.ask("new question")
        assert mock_agent.chat.call_count == 1
        assert not any(
            "Don't describe" in str(m.get("content", "")) for m in conv.messages
        )

    def test_final_response_is_recorded_in_history(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        mock_agent.functions = {}
        mock_agent.chat.return_value = self.make_mock_response(content="the answer")

        conv = Conversation(mock_agent)
        conv.ask("question")

        last = conv.messages[-1]
        assert last["role"] == "assistant"
        assert last["content"] == "the answer"


class TestApplyOutputSchemaShortcut:
    def test_skips_second_chat_when_final_content_is_json_object(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = {"type": "object"}
        conv = Conversation(mock_agent)
        payload = '{"answer":"done","data":[]}'
        out = conv.apply_output_schema(ChatResponse(content=payload))
        mock_agent.chat.assert_not_called()
        assert out.message.content == payload

    def test_appends_non_json_draft_then_calls_chat(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = {"type": "object"}
        conv = Conversation(mock_agent)
        conv.messages.append({"role": "user", "content": "go"})
        mock_agent.chat.return_value = ChatResponse(content='{"answer":"ok","data":[]}')
        out = conv.apply_output_schema(ChatResponse(content="thinking aloud"))
        mock_agent.chat.assert_called_once()
        assert conv.messages[-1]["content"] == "thinking aloud"
        assert out.message.content == '{"answer":"ok","data":[]}'
