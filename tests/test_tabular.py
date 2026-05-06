"""Tests for pygentix.tabular — register/reduce row helpers."""

import json
import uuid

import pytest

from pygentix.core import Agent, active_conversation
from pygentix.tabular import TabularStore, reduce_tabular_rows, register_tabular_tools
from pygentix.testing import MockAgent


class TestReduceTabularRows:
    def test_sum_without_group(self):
        rows = [{"a": 1, "v": 10}, {"a": 2, "v": 20}]
        spec = {"metrics": [{"op": "sum", "field": "v", "as": "total"}]}
        out = reduce_tabular_rows(rows, spec)
        assert out == [{"total": 30.0}]

    def test_count_rows(self):
        rows = [{"x": 1}, {"x": 2}, {"x": None}]
        spec = {"metrics": [{"op": "count", "as": "n"}]}
        out = reduce_tabular_rows(rows, spec)
        assert out == [{"n": 3}]

    def test_group_by_sum(self):
        rows = [
            {"owner": "u1", "value": 100},
            {"owner": "u1", "value": 50},
            {"owner": "u2", "value": 30},
        ]
        spec = {
            "group_by": ["owner"],
            "metrics": [
                {"op": "sum", "field": "value", "as": "total_value"},
                {"op": "count", "as": "n"},
            ],
        }
        out = reduce_tabular_rows(rows, spec)
        by_owner = {r["owner"]: r for r in out}
        assert by_owner["u1"]["total_value"] == 150.0
        assert by_owner["u1"]["n"] == 2
        assert by_owner["u2"]["total_value"] == 30.0
        assert by_owner["u2"]["n"] == 1

    def test_where_gte(self):
        rows = [{"p": 0.1}, {"p": 0.9}, {"p": 0.5}]
        spec = {
            "where": [{"field": "p", "op": "gte", "value": 0.5}],
            "metrics": [{"op": "count", "as": "n"}],
        }
        out = reduce_tabular_rows(rows, spec)
        assert out == [{"n": 2}]

    def test_invalid_field_name(self):
        with pytest.raises(ValueError, match="Invalid"):
            reduce_tabular_rows(
                [{"ok": 1}],
                {"metrics": [{"op": "sum", "field": "bad-field", "as": "x"}]},
            )


class TestTabularStore:
    def test_register_and_get(self):
        s = TabularStore(max_rows_per_dataset=10, max_datasets=5)
        did, n, cols = s.register([{"a": 1}, {"a": 2}])
        assert n == 2
        assert "a" in cols
        assert did.startswith("ds_")
        assert len(s.get(did)) == 2


class TestRegisterTabularTools:
    def test_tools_register_and_roundtrip(self):
        name = f"_tabular_test_{uuid.uuid4().hex[:12]}"
        Agent.registry.pop(name, None)
        Agent.pending_uses.pop(name, None)
        try:
            agent = MockAgent(name=name)
            register_tabular_tools(agent)
            assert "tabular_register" in agent.functions
            assert "tabular_reduce" in agent.functions

            conv = agent.start_conversation()
            tok = active_conversation.set(conv)
            try:
                reg = agent.functions["tabular_register"]
                raw = reg([{"owner": "a", "value": 10}, {"owner": "b", "value": 5}])
                meta = json.loads(raw)
                assert "dataset_id" in meta
                did = meta["dataset_id"]

                red = agent.functions["tabular_reduce"]
                spec = json.dumps(
                    {
                        "group_by": ["owner"],
                        "metrics": [{"op": "sum", "field": "value", "as": "t"}],
                    }
                )
                raw2 = red(did, spec)
                out = json.loads(raw2)
                assert "rows" in out
                assert len(out["rows"]) == 2
            finally:
                active_conversation.reset(tok)
        finally:
            Agent.registry.pop(name, None)
            Agent.pending_uses.pop(name, None)

    def test_register_by_name_before_construct_queues(self):
        pending_name = f"_tabular_pending_{uuid.uuid4().hex[:12]}"
        Agent.registry.pop(pending_name, None)
        Agent.pending_uses.pop(pending_name, None)
        try:
            register_tabular_tools(pending_name)
            assert pending_name in Agent.pending_uses
            assert len(Agent.pending_uses[pending_name]) == 2

            agent = MockAgent(name=pending_name)
            assert "tabular_register" in agent.functions
        finally:
            Agent.registry.pop(pending_name, None)
            Agent.pending_uses.pop(pending_name, None)

    def test_idempotent_second_register(self):
        name = f"_tabular_idem_{uuid.uuid4().hex[:12]}"
        Agent.registry.pop(name, None)
        try:
            agent = MockAgent(name=name)
            register_tabular_tools(agent)
            register_tabular_tools(agent)
            assert len([k for k in agent.functions if k.startswith("tabular_")]) == 2
        finally:
            Agent.registry.pop(name, None)
