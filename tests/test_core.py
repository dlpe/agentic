"""Tests for pygentix.core — Function, Conversation, and Agent."""

from unittest.mock import MagicMock

from pygentix.core import Conversation, Function


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
        return Conversation(mock_agent, "system prompt")

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


# -- Conversation retry logic ----------------------------------------------


class TestConversationRetry:
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

        conv = Conversation(mock_agent, "system")
        conv.ask("do something")
        assert mock_agent.chat.call_count == 2

    def test_no_retry_when_no_functions(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        mock_agent.functions = {}
        mock_agent.chat.return_value = self.make_mock_response(content="answer")

        conv = Conversation(mock_agent, "system")
        resp = conv.ask("question")
        assert resp.message.content == "answer"
        assert mock_agent.chat.call_count == 1

    def test_retries_when_text_only_and_functions_exist(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        tool_call = MagicMock()
        tool_call.function.name = "tool"
        tool_call.function.arguments = {}

        mock_agent.chat.side_effect = [
            self.make_mock_response(content="I will..."),
            self.make_mock_response(tool_calls=[tool_call]),
            self.make_mock_response(content="done"),
        ]
        mock_agent.functions = {"tool": lambda: "ok"}

        conv = Conversation(mock_agent, "system")
        conv.ask("do it")
        nudge_msgs = [m for m in conv.messages if "Don't describe" in m.get("content", "")]
        assert len(nudge_msgs) == 1

    def test_retries_on_new_question_after_prior_tool_turn(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        tool_call = MagicMock()
        tool_call.function.name = "tool"
        tool_call.function.arguments = {}

        mock_agent.chat.side_effect = [
            self.make_mock_response(content="Let me think..."),
            self.make_mock_response(tool_calls=[tool_call]),
            self.make_mock_response(content="done"),
        ]
        mock_agent.functions = {"tool": lambda: "ok"}

        conv = Conversation(mock_agent, "system")
        conv.messages.append({"role": "user", "content": "first"})
        conv.messages.append({"role": "assistant", "content": ""})
        conv.messages.append({"role": "tool", "tool_name": "tool", "content": "prior data"})

        conv.ask("new question")
        nudge_msgs = [m for m in conv.messages if "Don't describe" in m.get("content", "")]
        assert len(nudge_msgs) == 1

    def test_final_response_is_recorded_in_history(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        mock_agent.functions = {}
        mock_agent.chat.return_value = self.make_mock_response(content="the answer")

        conv = Conversation(mock_agent, "system")
        conv.ask("question")

        last = conv.messages[-1]
        assert last["role"] == "assistant"
        assert last["content"] == "the answer"
