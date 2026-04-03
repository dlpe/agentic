"""Tests for row-level security: direct scope (A), scope chains (B), and policy callbacks (C)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from pygentix.core import ChatResponse, Conversation, active_scope
from pygentix.sqlalchemy import SqlAlchemyAgent
from tests.conftest import Base, Tenant, Project, Task, StubSqlAgent


# -- helpers ---------------------------------------------------------------


@pytest.fixture
def scoped_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _seed(engine):
    """Insert two tenants with projects and tasks."""
    with Session(engine) as s:
        t1 = Tenant(id=1, name="Acme", owner_id=10)
        t2 = Tenant(id=2, name="Globex", owner_id=20)
        s.add_all([t1, t2])
        s.flush()
        p1 = Project(id=1, name="Alpha", tenant_id=1)
        p2 = Project(id=2, name="Beta", tenant_id=2)
        s.add_all([p1, p2])
        s.flush()
        s.add_all([
            Task(id=1, title="Task A1", project_id=1),
            Task(id=2, title="Task A2", project_id=1),
            Task(id=3, title="Task B1", project_id=2),
        ])
        s.commit()


# ==========================================================================
# A: Direct scope
# ==========================================================================


class TestDirectScope:
    @pytest.fixture
    def agent(self, scoped_engine):
        a = StubSqlAgent(engine=scoped_engine)
        a.reads(Tenant, scope={"owner_id": "current_user"})
        a.writes(Tenant, scope={"owner_id": "current_user"})
        _seed(scoped_engine)
        return a

    def test_query_returns_only_scoped_rows(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            rows = agent.run_query([{"entity": "Tenant"}])
        finally:
            active_scope.reset(token)
        assert len(rows) == 1
        assert rows[0]["name"] == "Acme"

    def test_query_without_scope_returns_all(self, agent):
        rows = agent.run_query([{"entity": "Tenant"}])
        assert len(rows) == 2

    def test_insert_auto_sets_scope_column(self, agent, scoped_engine):
        token = active_scope.set({"current_user": 10})
        try:
            agent.run_insert("Tenant", {"name": "NewCo"})
        finally:
            active_scope.reset(token)
        with Session(scoped_engine) as s:
            t = s.query(Tenant).filter_by(name="NewCo").one()
        assert t.owner_id == 10

    def test_insert_rejects_wrong_scope_value(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            with pytest.raises(PermissionError, match="scope requires"):
                agent.run_insert("Tenant", {"name": "Evil", "owner_id": 99})
        finally:
            active_scope.reset(token)

    def test_update_only_affects_scoped_rows(self, agent, scoped_engine):
        token = active_scope.set({"current_user": 10})
        try:
            result = agent.run_update("Tenant", {"name": "Acme"}, {"name": "AcmeV2"})
        finally:
            active_scope.reset(token)
        assert "1" in result
        with Session(scoped_engine) as s:
            assert s.query(Tenant).filter_by(name="AcmeV2").count() == 1
            assert s.query(Tenant).filter_by(name="Globex").count() == 1

    def test_update_rejects_cross_user_filter(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            with pytest.raises(PermissionError, match="scope restricts"):
                agent.run_update("Tenant", {"owner_id": 20}, {"name": "Hacked"})
        finally:
            active_scope.reset(token)

    def test_delete_only_affects_scoped_rows(self, agent, scoped_engine):
        token = active_scope.set({"current_user": 10})
        try:
            result = agent.run_delete("Tenant", {"name": "Acme"})
        finally:
            active_scope.reset(token)
        assert "1" in result
        with Session(scoped_engine) as s:
            assert s.query(Tenant).count() == 1
            assert s.query(Tenant).first().name == "Globex"

    def test_delete_rejects_cross_user_filter(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            with pytest.raises(PermissionError, match="scope restricts"):
                agent.run_delete("Tenant", {"owner_id": 20})
        finally:
            active_scope.reset(token)


# ==========================================================================
# B: Scope chains
# ==========================================================================


class TestScopeChain:
    @pytest.fixture
    def agent(self, scoped_engine):
        a = StubSqlAgent(engine=scoped_engine)
        a.reads(Tenant, scope={"owner_id": "current_user"})
        a.writes(Tenant, scope={"owner_id": "current_user"})
        a.reads(Project, scope_chain=[
            ("tenant_id", Tenant),
            ("owner_id", "current_user"),
        ])
        a.writes(Project, scope_chain=[
            ("tenant_id", Tenant),
            ("owner_id", "current_user"),
        ])
        a.reads(Task, scope_chain=[
            ("project_id", Project),
            ("tenant_id", Tenant),
            ("owner_id", "current_user"),
        ])
        a.writes(Task, scope_chain=[
            ("project_id", Project),
            ("tenant_id", Tenant),
            ("owner_id", "current_user"),
        ])
        _seed(scoped_engine)
        return a

    def test_query_project_scoped_by_chain(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            rows = agent.run_query([{"entity": "Project"}])
        finally:
            active_scope.reset(token)
        assert len(rows) == 1
        assert rows[0]["name"] == "Alpha"

    def test_query_task_two_level_chain(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            rows = agent.run_query([{"entity": "Task"}])
        finally:
            active_scope.reset(token)
        assert len(rows) == 2
        titles = {r["title"] for r in rows}
        assert titles == {"Task A1", "Task A2"}

    def test_insert_project_validates_fk_ownership(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            agent.run_insert("Project", {"name": "Good", "tenant_id": 1})
            with pytest.raises(PermissionError, match="not owned"):
                agent.run_insert("Project", {"name": "Bad", "tenant_id": 2})
        finally:
            active_scope.reset(token)

    def test_insert_task_validates_deep_chain(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            agent.run_insert("Task", {"title": "Good Task", "project_id": 1})
            with pytest.raises(PermissionError, match="not owned"):
                agent.run_insert("Task", {"title": "Bad Task", "project_id": 2})
        finally:
            active_scope.reset(token)

    def test_update_chain_blocks_foreign_rows(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            with pytest.raises(PermissionError, match="not owned by the current scope"):
                agent.run_update("Task", {"id": 3}, {"title": "Hacked"})
        finally:
            active_scope.reset(token)

    def test_delete_chain_blocks_foreign_rows(self, agent):
        token = active_scope.set({"current_user": 10})
        try:
            with pytest.raises(PermissionError, match="not owned by the current scope"):
                agent.run_delete("Task", {"id": 3})
        finally:
            active_scope.reset(token)


# ==========================================================================
# C: Policy callbacks
# ==========================================================================


class TestPolicy:
    def test_policy_blocks_tool_call(self):
        def deny_all(tool_name, arguments, scope):
            return False

        from pygentix.testing import MockAgent
        agent = MockAgent(responses=[
            {"content": "", "tool_calls": [{"name": "greet", "arguments": {"name": "X"}}]},
            {"content": "blocked"},
        ])

        @agent.uses
        def greet(name: str) -> str:
            return f"Hello {name}"

        conv = agent.start_conversation(policy=deny_all, scope={"user": 1})
        resp = conv.ask("Say hi")
        tool_msg = [m for m in conv.messages if m["role"] == "tool"]
        assert len(tool_msg) == 1
        assert "Permission denied" in tool_msg[0]["content"]

    def test_policy_allows_tool_call(self):
        def allow_all(tool_name, arguments, scope):
            return True

        from pygentix.testing import MockAgent
        agent = MockAgent(responses=[
            {"content": "", "tool_calls": [{"name": "greet", "arguments": {"name": "X"}}]},
            {"content": "Hello X"},
        ])

        @agent.uses
        def greet(name: str) -> str:
            return f"Hello {name}"

        conv = agent.start_conversation(policy=allow_all, scope={"user": 1})
        resp = conv.ask("Say hi")
        assert resp.message.content == "Hello X"
        tool_msg = [m for m in conv.messages if m["role"] == "tool"]
        assert "Hello X" in tool_msg[0]["content"]

    def test_policy_receives_scope(self):
        received = {}

        def capture_policy(tool_name, arguments, scope):
            received.update(scope)
            return True

        from pygentix.testing import MockAgent
        agent = MockAgent(responses=[
            {"content": "", "tool_calls": [{"name": "noop", "arguments": {}}]},
            {"content": "done"},
        ])

        @agent.uses
        def noop() -> str:
            return "ok"

        conv = agent.start_conversation(policy=capture_policy, scope={"tenant": 42})
        conv.ask("do it")
        assert received == {"tenant": 42}

    def test_policy_exception_becomes_denial(self):
        def bad_policy(tool_name, arguments, scope):
            raise RuntimeError("oops")

        from pygentix.testing import MockAgent
        agent = MockAgent(responses=[
            {"content": "", "tool_calls": [{"name": "noop", "arguments": {}}]},
            {"content": "ok"},
        ])

        @agent.uses
        def noop() -> str:
            return "ok"

        conv = agent.start_conversation(policy=bad_policy)
        conv.ask("go")
        tool_msg = [m for m in conv.messages if m["role"] == "tool"]
        assert "Permission denied" in tool_msg[0]["content"]
        assert "oops" in tool_msg[0]["content"]


# ==========================================================================
# A + C combined: both run
# ==========================================================================


class TestScopeAndPolicy:
    def test_policy_runs_before_scope(self, scoped_engine):
        policy_log = []

        def log_policy(tool_name, arguments, scope):
            policy_log.append(tool_name)
            return True

        a = StubSqlAgent(engine=scoped_engine)
        a.reads(Tenant, scope={"owner_id": "current_user"})
        a.writes(Tenant, scope={"owner_id": "current_user"})
        _seed(scoped_engine)

        conv = Conversation(
            a, "test", scope={"current_user": 10}, policy=log_policy,
        )

        token = active_scope.set({"current_user": 10})
        try:
            rows = a.run_query([{"entity": "Tenant"}])
        finally:
            active_scope.reset(token)
        assert len(rows) == 1

    def test_policy_denial_prevents_scope_execution(self, scoped_engine):
        def deny_all(tool_name, arguments, scope):
            return False

        from pygentix.testing import MockAgent

        a = StubSqlAgent(engine=scoped_engine)
        a.reads(Tenant, scope={"owner_id": "current_user"})
        _seed(scoped_engine)

        from pygentix.core import Conversation as Conv
        mock = MockAgent(responses=[
            {"content": "", "tool_calls": [{"name": "run_query", "arguments": {"query_steps": [{"entity": "Tenant"}]}}]},
            {"content": "denied"},
        ])
        mock.functions = dict(a.functions)

        conv = mock.start_conversation(
            policy=deny_all, scope={"current_user": 10},
        )
        resp = conv.ask("list tenants")
        tool_msg = [m for m in conv.messages if m["role"] == "tool"]
        assert "Permission denied" in tool_msg[0]["content"]


# ==========================================================================
# Backward compat: no scope = unrestricted
# ==========================================================================


class TestNoScope:
    def test_reads_without_scope_returns_all(self, scoped_engine):
        a = StubSqlAgent(engine=scoped_engine)
        a.reads(Tenant)
        _seed(scoped_engine)
        rows = a.run_query([{"entity": "Tenant"}])
        assert len(rows) == 2

    def test_writes_without_scope_are_unrestricted(self, scoped_engine):
        a = StubSqlAgent(engine=scoped_engine)
        a.reads(Tenant)
        a.writes(Tenant)
        _seed(scoped_engine)
        a.run_insert("Tenant", {"name": "New", "owner_id": 99})
        with Session(scoped_engine) as s:
            assert s.query(Tenant).count() == 3

    def test_conversation_without_scope_or_policy(self):
        from pygentix.testing import MockAgent
        agent = MockAgent(responses=[{"content": "hello"}])
        conv = agent.start_conversation()
        assert conv.scope == {}
        assert conv.policy is None


# ==========================================================================
# Serialization preserves scope
# ==========================================================================


class TestScopeSerialization:
    def test_to_dict_includes_scope(self):
        from pygentix.testing import MockAgent
        agent = MockAgent(responses=[])
        conv = agent.start_conversation(scope={"current_user": 5})
        d = conv.to_dict()
        assert d["scope"] == {"current_user": 5}

    def test_from_dict_restores_scope(self):
        from pygentix.testing import MockAgent
        agent = MockAgent(responses=[])
        conv = agent.start_conversation(scope={"current_user": 5})
        data = conv.to_dict()

        restored = Conversation.from_dict(agent, data)
        assert restored.scope == {"current_user": 5}
        assert restored.policy is None

    def test_to_json_roundtrip(self):
        from pygentix.testing import MockAgent
        agent = MockAgent(responses=[])
        conv = agent.start_conversation(scope={"role": "admin"})
        json_str = conv.to_json()
        restored = Conversation.from_json(agent, json_str)
        assert restored.scope == {"role": "admin"}
