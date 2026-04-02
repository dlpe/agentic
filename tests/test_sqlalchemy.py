"""Tests for pygentix.sqlalchemy — SqlAlchemyAgent CRUD, coercion, and query building."""

import pytest
from datetime import date, datetime
from sqlalchemy.orm import Session

from pygentix.core import Function
from pygentix.sqlalchemy import describe_model
from tests.conftest import Author, Book, Event, StubSqlAgent


# -- reads / writes decorators ---------------------------------------------


class TestReads:
    def test_registers_entity(self, agent):
        assert Author in agent.entities
        assert Book in agent.entities

    def test_adds_run_query_function(self, agent):
        assert "run_query" in agent.functions

    def test_resolve_populates_descriptions(self, agent):
        agent._resolve_entities()
        for cls in (Author, Book):
            desc = agent.entities[cls]
            assert desc["model"] == cls.__name__
            assert len(desc["columns"]) > 0


class TestWrites:
    def test_registers_writable(self, agent):
        assert "Author" in agent._writable
        assert "Book" in agent._writable

    def test_adds_write_functions(self, agent):
        for name in ("run_insert", "run_update", "run_delete"):
            assert name in agent.functions


class TestFunctionWrapping:
    def test_reads_wraps_run_query(self, agent):
        assert isinstance(agent.functions["run_query"], Function)

    def test_writes_wraps_crud_tools(self, agent):
        for name in ("run_insert", "run_update", "run_delete"):
            assert isinstance(agent.functions[name], Function)

    def test_all_tools_are_callable(self, agent):
        for func in agent.functions.values():
            assert callable(func)


# -- run_insert ------------------------------------------------------------


class TestRunInsert:
    def test_single_row(self, agent, engine):
        agent.run_insert("Author", {"name": "Tolkien"})
        with Session(engine) as s:
            assert s.query(Author).first().name == "Tolkien"

    def test_with_foreign_key(self, agent, engine):
        agent.run_insert("Author", {"name": "Tolkien"})
        agent.run_insert("Book", {"title": "The Hobbit", "price": 12.99, "author_id": 1})
        with Session(engine) as s:
            book = s.query(Book).first()
        assert book.title == "The Hobbit"
        assert book.author_id == 1

    def test_batch_insert(self, agent, engine):
        agent.run_insert("Author", [{"name": "A"}, {"name": "B"}, {"name": "C"}])
        with Session(engine) as s:
            assert s.query(Author).count() == 3

    def test_returns_confirmation_string(self, agent):
        result = agent.run_insert("Author", {"name": "Tolkien"})
        assert "Inserted" in result and "Author" in result


# -- run_update ------------------------------------------------------------


class TestRunUpdate:
    def test_updates_matching_rows(self, agent, engine):
        agent.run_insert("Author", {"name": "Tolkein"})
        agent.run_update("Author", {"name": "Tolkein"}, {"name": "Tolkien"})
        with Session(engine) as s:
            assert s.query(Author).first().name == "Tolkien"

    def test_returns_count(self, agent):
        agent.run_insert("Author", {"name": "A"})
        agent.run_insert("Author", {"name": "A"})
        assert "2" in agent.run_update("Author", {"name": "A"}, {"name": "B"})

    def test_no_match_updates_zero(self, agent):
        assert "0" in agent.run_update("Author", {"name": "Nobody"}, {"name": "X"})


# -- run_delete ------------------------------------------------------------


class TestRunDelete:
    def test_deletes_matching_rows(self, agent, engine):
        agent.run_insert("Author", {"name": "A"})
        agent.run_insert("Author", {"name": "B"})
        agent.run_delete("Author", {"name": "A"})
        with Session(engine) as s:
            remaining = s.query(Author).all()
        assert len(remaining) == 1
        assert remaining[0].name == "B"

    def test_returns_count(self, agent):
        agent.run_insert("Author", {"name": "A"})
        agent.run_insert("Author", {"name": "A"})
        assert "2" in agent.run_delete("Author", {"name": "A"})

    def test_no_match_deletes_zero(self, agent):
        assert "0" in agent.run_delete("Author", {"name": "Nobody"})


# -- describe_model --------------------------------------------------------


class TestDescribeModel:
    def test_columns(self):
        col_names = [c["name"] for c in describe_model(Author)["columns"]]
        assert "id" in col_names
        assert "name" in col_names

    def test_primary_key_flagged(self):
        id_col = next(c for c in describe_model(Author)["columns"] if c["name"] == "id")
        assert id_col["primary_key"] is True

    def test_foreign_keys_described(self):
        fk_col = next(c for c in describe_model(Book)["columns"] if c["name"] == "author_id")
        assert fk_col["foreign_keys"][0]["target"] == "authors.id"

    def test_relationships_described(self):
        rel_names = [r["name"] for r in describe_model(Author)["relationships"]]
        assert "books" in rel_names

    def test_table_name(self):
        assert describe_model(Book)["table"] == "books"


# -- _coerce_values --------------------------------------------------------


class TestCoerceValues:
    @pytest.fixture
    def agent_with_events(self, engine):
        a = StubSqlAgent(engine=engine)
        a.writes(Event)
        return a

    def test_string_to_int(self, agent):
        result = agent._coerce_values(Book, {"author_id": "42"})
        assert result["author_id"] == 42

    def test_string_to_float(self, agent):
        result = agent._coerce_values(Book, {"price": "19.99"})
        assert result["price"] == pytest.approx(19.99)

    def test_string_to_date(self, agent_with_events):
        result = agent_with_events._coerce_values(Event, {"event_date": "2026-04-01"})
        assert result["event_date"] == date(2026, 4, 1)

    def test_string_to_datetime(self, agent_with_events):
        result = agent_with_events._coerce_values(Event, {"created_at": "2026-04-01T10:30:00"})
        assert result["created_at"] == datetime(2026, 4, 1, 10, 30, 0)

    def test_string_to_bool_true(self, agent_with_events):
        for val in ("true", "True", "1", "yes"):
            assert agent_with_events._coerce_values(Event, {"active": val})["active"] is True

    def test_string_to_bool_false(self, agent_with_events):
        for val in ("false", "0", "no"):
            assert agent_with_events._coerce_values(Event, {"active": val})["active"] is False

    def test_leaves_non_string_values_unchanged(self, agent):
        result = agent._coerce_values(Book, {"price": 12.99, "author_id": 1})
        assert result["price"] == pytest.approx(12.99)
        assert result["author_id"] == 1

    def test_leaves_string_columns_unchanged(self, agent):
        assert agent._coerce_values(Author, {"name": "Tolkien"})["name"] == "Tolkien"

    def test_end_to_end_insert(self, agent_with_events, engine):
        agent_with_events.run_insert("Event", {
            "name": "Launch",
            "event_date": "2026-06-15",
            "active": "true",
            "score": "9.5",
        })
        with Session(engine) as s:
            event = s.query(Event).first()
        assert event.event_date == date(2026, 6, 15)
        assert event.active is True
        assert event.score == pytest.approx(9.5)

    def test_end_to_end_update(self, agent_with_events, engine):
        agent_with_events.run_insert("Event", {"name": "Old", "score": "5.0"})
        agent_with_events.run_update("Event", {"id": "1"}, {"score": "8.0"})
        with Session(engine) as s:
            assert s.query(Event).first().score == pytest.approx(8.0)

    def test_end_to_end_delete(self, agent_with_events, engine):
        agent_with_events.run_insert("Event", {"name": "Temp", "score": "1.0"})
        agent_with_events.run_delete("Event", {"id": "1"})
        with Session(engine) as s:
            assert s.query(Event).count() == 0


# -- operator filters -----------------------------------------------------


class TestOperatorMap:
    def test_eq(self, agent, engine):
        agent.run_insert("Author", [{"name": "Tolkien"}, {"name": "Rowling"}])
        assert len(agent.run_query([{"entity": "Author", "field": "name", "op": "eq", "value": "Tolkien"}])) == 1

    def test_gt(self, agent, engine):
        agent.run_insert("Book", [
            {"title": "Cheap", "price": 5.0, "author_id": None},
            {"title": "Pricey", "price": 50.0, "author_id": None},
        ])
        assert len(agent.run_query([{"entity": "Book", "field": "price", "op": "gt", "value": 10.0}])) == 1

    def test_lt(self, agent, engine):
        agent.run_insert("Book", [
            {"title": "Cheap", "price": 5.0, "author_id": None},
            {"title": "Pricey", "price": 50.0, "author_id": None},
        ])
        assert len(agent.run_query([{"entity": "Book", "field": "price", "op": "lt", "value": 10.0}])) == 1

    def test_gte(self, agent, engine):
        agent.run_insert("Book", [
            {"title": "A", "price": 10.0, "author_id": None},
            {"title": "B", "price": 20.0, "author_id": None},
        ])
        assert len(agent.run_query([{"entity": "Book", "field": "price", "op": "gte", "value": 10.0}])) == 2

    def test_lte(self, agent, engine):
        agent.run_insert("Book", [
            {"title": "A", "price": 10.0, "author_id": None},
            {"title": "B", "price": 20.0, "author_id": None},
        ])
        assert len(agent.run_query([{"entity": "Book", "field": "price", "op": "lte", "value": 10.0}])) == 1


# -- run_query: multi-step and edge cases ----------------------------------


class TestRunQuery:
    def test_entity_only_returns_all(self, agent, engine):
        agent.run_insert("Author", [{"name": "A"}, {"name": "B"}])
        assert len(agent.run_query([{"entity": "Author"}])) == 2

    def test_entity_step_plus_filter(self, agent, engine):
        agent.run_insert("Author", [{"name": "A"}, {"name": "B"}])
        rows = agent.run_query([
            {"entity": "Author"},
            {"field": "name", "op": "eq", "value": "A"},
        ])
        assert len(rows) == 1

    def test_step_without_op_is_skipped(self, agent, engine):
        agent.run_insert("Author", {"name": "A"})
        assert len(agent.run_query([{"entity": "Author", "field": "name"}])) == 1


# -- deferred entity resolution -------------------------------------------


class TestDeferredResolve:
    def test_none_before_resolve(self, agent):
        assert agent.entities[Author] is None

    def test_populated_after_resolve(self, agent):
        agent._resolve_entities()
        assert "columns" in agent.entities[Author]

    def test_idempotent(self, agent):
        agent._resolve_entities()
        first = agent.entities[Author]
        agent._resolve_entities()
        assert agent.entities[Author] is first
