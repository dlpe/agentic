"""Tests for pygentix.core — Function, Conversation, and Agent."""

from unittest.mock import MagicMock

from pygentix.core import (
    ChatResponse,
    Conversation,
    Function,
    active_scope,
    looks_like_finished_answer,
    looks_like_plan,
)


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

    def test_scope_overwrites_model_supplied_enterprise_id(self):
        def list_rows(enterprise_id: str) -> str:
            return enterprise_id

        f = Function(list_rows)
        token = active_scope.set({"enterprise_id": "current-ent"})
        try:
            assert f(enterprise_id="other-ent") == "current-ent"
        finally:
            active_scope.reset(token)

    def test_enterprise_id_hidden_from_schema_without_scope(self):
        def list_rows(enterprise_id: str, limit: int = 10) -> str:
            return enterprise_id

        schema = Function(list_rows).to_tool_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "enterprise_id" not in props
        assert "limit" in props


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

    def test_single_insist_when_model_narrates_without_tools(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        mock_agent.functions = {"tool": lambda: "ok"}
        mock_agent.chat.side_effect = [
            self.make_mock_response(content="I will look that up."),
            self.make_mock_response(content="3 open opportunities"),
        ]

        conv = Conversation(mock_agent)
        conv.ask("do it")
        assert mock_agent.chat.call_count == 2
        assert not any(
            "Call the required tool" in str(m.get("content", "")) for m in conv.messages
        )
        assert conv.messages[-1]["content"] == "3 open opportunities"

    def test_no_insist_on_finished_answer(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        mock_agent.functions = {"tool": lambda: "ok"}
        mock_agent.chat.return_value = self.make_mock_response(
            content="You have 3 opportunities."
        )

        conv = Conversation(mock_agent)
        conv.ask("how many?")
        assert mock_agent.chat.call_count == 1

    def test_no_insist_on_finished_answer_after_prior_tool_messages(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        mock_agent.functions = {"tool": lambda: "ok"}
        mock_agent.chat.return_value = self.make_mock_response(
            content="Based on the earlier data, the total is 10."
        )

        conv = Conversation(mock_agent)
        conv.messages.append({"role": "user", "content": "first"})
        conv.messages.append({"role": "assistant", "content": ""})
        conv.messages.append(
            {"role": "tool", "tool_name": "tool", "content": "prior data"}
        )

        conv.ask("new question")
        assert mock_agent.chat.call_count == 1
        assert not any(
            "Call the required tool" in str(m.get("content", "")) for m in conv.messages
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


class TestLooksLikePlan:
    def test_detects_plan_phrases(self):
        assert looks_like_plan("I will fetch the invoices next.")
        assert looks_like_plan("We need to do A, B and C")
        assert looks_like_plan("Let me look that up:")

    def test_ignores_finished_answers(self):
        assert looks_like_plan("You have 3 opportunities.") is False
        assert looks_like_plan("hello") is False
        assert looks_like_plan("") is False
        assert looks_like_plan(
            "Your Excel file is ready! Let me know if you'd like to filter."
        ) is False
        assert looks_like_plan(
            "Your Excel file is ready.\nI'll include Sale Date and Client."
        ) is False
        assert looks_like_finished_answer(
            "Download /static/chat_files/sold.xlsx"
        ) is True


class TestInsistAfterTools:
    def test_insists_when_model_plans_after_a_tool_result(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        tool_call = MagicMock()
        tool_call.id = "c1"
        tool_call.function.name = "tool"
        tool_call.function.arguments = {}
        first = MagicMock()
        first.message.content = ""
        first.message.tool_calls = [tool_call]
        after_tool = MagicMock()
        after_tool.message.content = "Now I need to sum the values."
        after_tool.message.tool_calls = None
        final = MagicMock()
        final.message.content = "The total is 10."
        final.message.tool_calls = None
        mock_agent.chat.side_effect = [first, after_tool, final]
        mock_agent.functions = {"tool": lambda: "ok"}

        conv = Conversation(mock_agent)
        resp = conv.ask("total?")
        assert mock_agent.chat.call_count == 3
        assert resp.message.content == "The total is 10."
        assert not any(
            "Call the required tool" in str(m.get("content", "")) for m in conv.messages
        )

    def test_insists_at_most_once_per_turn(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        tool_call = MagicMock()
        tool_call.id = "c1"
        tool_call.function.name = "tool"
        tool_call.function.arguments = {}
        announced = MagicMock()
        announced.message.content = "I will fetch the invoices next."
        announced.message.tool_calls = None
        with_tool = MagicMock()
        with_tool.message.content = ""
        with_tool.message.tool_calls = [tool_call]
        announced_again = MagicMock()
        announced_again.message.content = "I will format the file next."
        announced_again.message.tool_calls = None
        unused = MagicMock()
        unused.message.content = "should not run"
        unused.message.tool_calls = None
        mock_agent.chat.side_effect = [
            announced,
            with_tool,
            announced_again,
            unused,
        ]
        mock_agent.functions = {"tool": lambda: "ok"}

        conv = Conversation(mock_agent)
        resp = conv.ask("invoices?")
        assert mock_agent.chat.call_count == 3
        assert resp.message.content == "I will format the file next."


class TestMaxToolRounds:
    def test_stops_unbounded_tool_loop(self):
        mock_agent = MagicMock()
        mock_agent.output_schema = None

        def looping_response(*_args, **_kwargs):
            tool_call = MagicMock()
            tool_call.id = "c1"
            tool_call.function.name = "tool"
            tool_call.function.arguments = {"n": mock_agent.chat.call_count}
            response = MagicMock()
            response.message.content = ""
            response.message.tool_calls = [tool_call]
            return response

        mock_agent.chat.side_effect = looping_response
        mock_agent.functions = {"tool": lambda n=0: "ok"}

        conv = Conversation(mock_agent, max_tool_rounds=2)
        resp = conv.ask("loop")
        assert mock_agent.chat.call_count == 3
        assert "too many tool steps" in resp.message.content


class TestRepeatedToolCalls:
    def test_same_tool_args_do_not_run_twice(self):
        calls = {"n": 0}
        mock_agent = MagicMock()
        mock_agent.output_schema = None
        tool_call = MagicMock()
        tool_call.id = "c1"
        tool_call.function.name = "export"
        tool_call.function.arguments = {}
        first = MagicMock()
        first.message.content = "Your file: /static/chat_files/a.xlsx"
        first.message.tool_calls = [tool_call]
        again = MagicMock()
        again.message.content = "Your file: /static/chat_files/a.xlsx I'll add more."
        again.message.tool_calls = [tool_call]
        mock_agent.chat.side_effect = [first, again]

        def export() -> str:
            calls["n"] += 1
            return "/static/chat_files/a.xlsx"

        mock_agent.functions = {"export": export}
        conv = Conversation(mock_agent)
        resp = conv.ask("export")
        assert calls["n"] == 1
        assert "/static/chat_files/a.xlsx" in resp.message.content
