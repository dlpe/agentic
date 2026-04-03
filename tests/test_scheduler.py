"""Tests for pygentix.scheduler — SchedulerAgent scheduling, persistence, and execution."""

import json
import time
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pygentix.scheduler import SchedulerAgent, ScheduledTask
from pygentix.testing import MockAgent


class StubSchedulerAgent(SchedulerAgent):
    """Concrete agent combining MockAgent responses with SchedulerAgent."""

    def __init__(self, schedule_file, responses=None, **kwargs):
        self.mock_responses = list(responses or [""])
        self.mock_index = 0
        super().__init__(schedule_file=schedule_file, **kwargs)

    def chat(self, messages, **kwargs):
        from pygentix.core import ChatResponse
        entry = self.mock_responses[self.mock_index % len(self.mock_responses)]
        self.mock_index += 1
        if isinstance(entry, str):
            return ChatResponse(content=entry)
        return ChatResponse(
            content=entry.get("content", ""),
            tool_calls=entry.get("tool_calls"),
        )


@pytest.fixture
def schedule_file(tmp_path):
    return str(tmp_path / "tasks.json")


@pytest.fixture
def agent(schedule_file):
    a = StubSchedulerAgent(schedule_file=schedule_file, responses=["done"])
    return a


# -- ScheduledTask ---------------------------------------------------------


class TestScheduledTask:
    def test_one_shot_due(self):
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        task = ScheduledTask(id="t1", type="tool_call", run_at=past)
        assert task.is_due()

    def test_one_shot_not_due(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        task = ScheduledTask(id="t2", type="tool_call", run_at=future)
        assert not task.is_due()

    def test_non_pending_never_due(self):
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        task = ScheduledTask(id="t3", type="tool_call", run_at=past, status="done")
        assert not task.is_due()

    def test_cron_due(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        task = ScheduledTask(id="t4", type="tool_call", cron="* * * * *", created_at=old)
        assert task.is_due()

    def test_roundtrip(self):
        task = ScheduledTask(id="t5", type="conversation", prompt="hello")
        restored = ScheduledTask.from_dict(task.to_dict())
        assert restored.id == "t5"
        assert restored.prompt == "hello"


# -- JSON persistence -----------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, agent, schedule_file):
        agent.schedule_task("schedule_task", {}, run_at="2099-01-01T00:00:00")
        tasks = agent.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].type == "tool_call"

    def test_load_empty_file(self, agent):
        tasks = agent.load_tasks()
        assert tasks == []


# -- schedule_task ---------------------------------------------------------


class TestScheduleTask:
    def test_creates_task_in_file(self, agent, schedule_file):
        result = agent.schedule_task("schedule_task", {"x": 1}, run_at="2099-01-01T00:00:00")
        assert "Scheduled task" in result
        raw = json.loads(Path(schedule_file).read_text())
        assert len(raw) == 1
        assert raw[0]["function_name"] == "schedule_task"

    def test_rejects_missing_schedule(self, agent):
        result = agent.schedule_task("schedule_task", {})
        assert "Error" in result

    def test_rejects_unknown_function(self, agent):
        result = agent.schedule_task("nonexistent_func", {}, run_at="2099-01-01T00:00:00")
        assert "Error" in result


# -- schedule_conversation -------------------------------------------------


class TestScheduleConversation:
    def test_creates_conversation_task(self, agent, schedule_file):
        result = agent.schedule_conversation("send report", run_at="2099-01-01T00:00:00")
        assert "Scheduled conversation" in result
        raw = json.loads(Path(schedule_file).read_text())
        assert raw[0]["type"] == "conversation"
        assert raw[0]["prompt"] == "send report"

    def test_rejects_missing_schedule(self, agent):
        result = agent.schedule_conversation("hello")
        assert "Error" in result


# -- list_scheduled_tasks --------------------------------------------------


class TestListTasks:
    def test_empty(self, agent):
        result = agent.list_scheduled_tasks()
        assert "No pending" in result

    def test_lists_pending(self, agent):
        agent.schedule_task("schedule_task", {}, run_at="2099-01-01T00:00:00")
        agent.schedule_conversation("hello", run_at="2099-01-01T00:00:00")
        result = agent.list_scheduled_tasks()
        assert "2 pending" in result


# -- cancel_scheduled_task -------------------------------------------------


class TestCancelTask:
    def test_cancel_existing(self, agent):
        agent.schedule_task("schedule_task", {}, run_at="2099-01-01T00:00:00")
        task_id = agent.load_tasks()[0].id
        result = agent.cancel_scheduled_task(task_id)
        assert "Cancelled" in result
        assert agent.load_tasks()[0].status == "cancelled"

    def test_cancel_nonexistent(self, agent):
        result = agent.cancel_scheduled_task("nosuchid")
        assert "No pending task" in result


# -- tick() execution ------------------------------------------------------


class TestTick:
    def test_executes_due_tool_call(self, agent):
        call_log = []

        @agent.uses
        def greet(name: str) -> str:
            call_log.append(name)
            return f"Hello {name}"

        past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        agent.schedule_task("greet", {"name": "Alice"}, run_at=past)
        results = agent.tick()
        assert len(results) == 1
        assert "Hello Alice" in results[0]
        assert call_log == ["Alice"]

        tasks = agent.load_tasks()
        assert tasks[0].status == "done"

    def test_executes_due_conversation(self, schedule_file):
        a = StubSchedulerAgent(
            schedule_file=schedule_file,
            responses=["Report sent!"],
        )
        past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        a.schedule_conversation("send report", run_at=past)
        results = a.tick()
        assert len(results) == 1
        assert "Report sent!" in results[0]

    def test_skips_future_tasks(self, agent):
        agent.schedule_task("schedule_task", {}, run_at="2099-01-01T00:00:00")
        results = agent.tick()
        assert results == []

    def test_cron_task_stays_pending_after_fire(self, agent):
        call_log = []

        @agent.uses
        def ping() -> str:
            call_log.append(1)
            return "pong"

        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        task = ScheduledTask(
            id="cron1", type="tool_call", function_name="ping",
            arguments={}, cron="* * * * *", created_at=old,
        )
        agent.add_task(task)
        agent.tick()
        assert len(call_log) == 1
        tasks = agent.load_tasks()
        assert tasks[0].status == "pending"


# -- conversation context --------------------------------------------------


class TestConversationContext:
    def test_schedule_conversation_captures_messages(self, schedule_file):
        a = StubSchedulerAgent(
            schedule_file=schedule_file,
            responses=[
                {"tool_calls": [{"name": "schedule_conversation", "arguments": {"prompt": "follow up", "run_at": "2099-01-01T00:00:00"}}]},
                "Scheduled!",
            ],
        )
        conv = a.start_conversation()
        conv.messages.append({"role": "user", "content": "My name is Alice"})
        conv.messages.append({"role": "assistant", "content": "Hello Alice!"})
        conv.ask("Schedule a follow-up conversation for tomorrow")

        tasks = a.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].context_messages is not None
        contents = [m.get("content", "") for m in tasks[0].context_messages]
        assert "My name is Alice" in contents
        assert "Hello Alice!" in contents

    def test_execute_conversation_restores_context(self, schedule_file):
        received_messages = []

        class CapturingAgent(SchedulerAgent):
            def chat(self, messages, **kwargs):
                from pygentix.core import ChatResponse
                received_messages.extend(messages)
                return ChatResponse(content="Done")

        a = CapturingAgent(schedule_file=schedule_file)
        context = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "My name is Bob"},
            {"role": "assistant", "content": "Hi Bob!"},
        ]
        past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        task = ScheduledTask(
            id="ctx1", type="conversation", prompt="What is my name?",
            context_messages=context, run_at=past,
        )
        a.add_task(task)
        results = a.tick()
        assert len(results) == 1
        assert "Done" in results[0]
        prior_contents = [m.get("content", "") for m in received_messages]
        assert "My name is Bob" in prior_contents
        assert "Hi Bob!" in prior_contents
        assert "What is my name?" in prior_contents

    def test_context_persists_to_json(self, schedule_file):
        context = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        task = ScheduledTask(
            id="ctx2", type="conversation", prompt="hi",
            context_messages=context, run_at="2099-01-01T00:00:00",
        )
        raw = json.dumps([task.to_dict()])
        restored = ScheduledTask.from_dict(json.loads(raw)[0])
        assert restored.context_messages == context


# -- missed tasks ----------------------------------------------------------


class TestMissedTasks:
    def test_mark_missed_on_startup(self, agent, schedule_file):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        task = ScheduledTask(
            id="old1", type="tool_call", function_name="schedule_task",
            arguments={}, run_at=past,
        )
        agent.add_task(task)
        agent.mark_missed_tasks()
        tasks = agent.load_tasks()
        assert tasks[0].status == "missed"

    def test_cron_not_marked_missed(self, agent):
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        task = ScheduledTask(
            id="cron2", type="tool_call", function_name="schedule_task",
            arguments={}, cron="0 9 * * 1", created_at=old,
        )
        agent.add_task(task)
        agent.mark_missed_tasks()
        tasks = agent.load_tasks()
        assert tasks[0].status == "pending"


# -- background thread lifecycle -------------------------------------------


class TestBackgroundThread:
    def test_start_and_stop(self, agent):
        agent.start_scheduler()
        assert agent.scheduler_thread is not None
        assert agent.scheduler_thread.is_alive()
        agent.stop_scheduler()
        assert agent.scheduler_thread is None

    def test_start_is_idempotent(self, agent):
        agent.start_scheduler()
        thread1 = agent.scheduler_thread
        agent.start_scheduler()
        assert agent.scheduler_thread is thread1
        agent.stop_scheduler()

    def test_thread_executes_due_task(self, agent):
        call_log = []

        @agent.uses
        def ping() -> str:
            call_log.append(1)
            return "pong"

        agent.poll_interval = 0.2
        agent.start_scheduler()
        past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        agent.schedule_task("ping", {}, run_at=past)
        time.sleep(1)
        agent.stop_scheduler()
        assert len(call_log) >= 1
