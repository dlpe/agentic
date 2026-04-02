"""Tests for pygentix.output — OutputAgent schema generation and parsing."""

import pytest
from unittest.mock import MagicMock

from pygentix.output import OutputAgent
from pygentix.sqlalchemy import SqlAlchemyAgent
from tests.conftest import Author


class StubOutputAgent(OutputAgent):
    def chat(self, messages, **kwargs):
        raise NotImplementedError


# -- output() with a raw dict ---------------------------------------------


class TestOutputDict:
    def test_accepts_dict_schema(self):
        a = StubOutputAgent()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        a.output(schema)
        assert a.output_schema == schema

    def test_output_cls_is_none_for_dict(self):
        a = StubOutputAgent()
        a.output({"type": "object", "properties": {}})
        assert a._output_cls is None


# -- output() with a class ------------------------------------------------


class TestOutputFromClass:
    def test_basic_types(self):
        a = StubOutputAgent()

        @a.output
        class Schema:
            name: str
            age: int
            score: float
            active: bool

        props = a.output_schema["properties"]
        assert props["name"] == {"type": "string"}
        assert props["age"] == {"type": "integer"}
        assert props["score"] == {"type": "number"}
        assert props["active"] == {"type": "boolean"}

    def test_all_annotated_fields_required_by_default(self):
        a = StubOutputAgent()

        @a.output
        class Schema:
            name: str
            age: int

        assert sorted(a.output_schema["required"]) == ["age", "name"]

    def test_fields_with_defaults_are_optional(self):
        a = StubOutputAgent()

        @a.output
        class Schema:
            name: str
            nickname: str = ""
            score: float = 0.0

        assert a.output_schema["required"] == ["name"]

    def test_list_type(self):
        a = StubOutputAgent()

        @a.output
        class Schema:
            tags: list[str]
            scores: list[int]

        assert a.output_schema["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
        assert a.output_schema["properties"]["scores"] == {"type": "array", "items": {"type": "integer"}}

    def test_bare_list(self):
        a = StubOutputAgent()

        @a.output
        class Schema:
            items: list

        assert a.output_schema["properties"]["items"]["type"] == "array"

    def test_dict_type(self):
        a = StubOutputAgent()

        @a.output
        class Schema:
            metadata: dict

        assert a.output_schema["properties"]["metadata"] == {"type": "object"}

    def test_decorator_returns_class(self):
        a = StubOutputAgent()

        @a.output
        class Schema:
            x: int

        assert Schema.__annotations__["x"] is int

    def test_output_cls_stored(self):
        a = StubOutputAgent()

        @a.output
        class Schema:
            name: str

        assert a._output_cls is Schema


# -- parse_output ----------------------------------------------------------


class TestParseOutput:
    def test_parse_to_dict_without_class(self):
        a = StubOutputAgent()
        a.output({"type": "object", "properties": {"x": {"type": "integer"}}})
        resp = MagicMock()
        resp.message.content = '{"x": 42}'
        assert a.parse_output(resp) == {"x": 42}

    def test_parse_to_class_instance(self):
        a = StubOutputAgent()

        @a.output
        class Result:
            answer: str
            confidence: float

            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        resp = MagicMock()
        resp.message.content = '{"answer": "yes", "confidence": 0.95}'
        result = a.parse_output(resp)
        assert isinstance(result, Result)
        assert result.answer == "yes"
        assert result.confidence == pytest.approx(0.95)

    def test_invalid_json_returns_raw_content(self):
        a = StubOutputAgent()
        a.output({"type": "object"})
        resp = MagicMock()
        resp.message.content = "not json"
        assert a.parse_output(resp) == "not json"

    def test_truncated_json_returns_raw_content(self):
        a = StubOutputAgent()
        a.output({"type": "object", "properties": {"answer": {"type": "string"}}})
        resp = MagicMock()
        resp.message.content = '{"answer": "The wea'
        assert a.parse_output(resp) == '{"answer": "The wea'

    def test_parse_plain_class_without_init(self):
        a = StubOutputAgent()

        @a.output
        class Result:
            answer: str
            details: str = ""

        resp = MagicMock()
        resp.message.content = '{"answer": "hello", "details": "world"}'
        result = a.parse_output(resp)
        assert isinstance(result, Result)
        assert result.answer == "hello"
        assert result.details == "world"


# -- _type_to_json edge cases ----------------------------------------------


class TestTypeToJson:
    def test_unknown_type_falls_back_to_string(self):
        assert OutputAgent._type_to_json(bytes) == {"type": "string"}

    def test_nested_list(self):
        result = OutputAgent._type_to_json(list[list[int]])
        assert result == {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}


# -- multiple inheritance --------------------------------------------------


class CombinedAgent(OutputAgent, SqlAlchemyAgent):
    def chat(self, messages, **kwargs):
        raise NotImplementedError


class TestOutputWithSqlAlchemy:
    def test_has_both_output_schema_and_db_tools(self, engine):
        a = CombinedAgent(engine=engine)
        a.reads(Author)
        a.writes(Author)

        @a.output
        class Resp:
            answer: str

        assert a.output_schema is not None
        assert "run_query" in a.functions
        assert "run_insert" in a.functions

    def test_output_schema_coexists_with_entities(self, engine):
        a = CombinedAgent(engine=engine)
        a.reads(Author)

        @a.output
        class Resp:
            answer: str

        assert Author in a.entities
        assert a.output_schema["properties"]["answer"] == {"type": "string"}

    def test_start_conversation_injects_schema_hint(self, engine):
        a = CombinedAgent(engine=engine)
        a.reads(Author)

        @a.output
        class Resp:
            answer: str

        conv = a.start_conversation()
        system_msg = conv.messages[0]["content"]
        assert "JSON" in system_msg
        assert '"answer"' in system_msg
