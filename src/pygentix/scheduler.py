"""Task scheduling for agents — one-shot and recurring via cron expressions."""

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import Agent, Function, active_conversation

__all__ = ["SchedulerAgent"]

logger = logging.getLogger("pygentix")


@dataclass
class ScheduledTask:
    """A single scheduled task persisted to JSON."""

    id: str
    type: str  # "tool_call" or "conversation"
    status: str = "pending"  # "pending", "done", "missed", "cancelled"
    function_name: str | None = None
    arguments: dict | None = None
    prompt: str | None = None
    context_messages: list[dict] | None = None
    run_at: str | None = None  # ISO-8601 datetime for one-shot
    cron: str | None = None  # cron expression for recurring
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def is_due(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if self.status != "pending":
            return False
        if self.run_at:
            target = datetime.fromisoformat(self.run_at)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return now >= target
        if self.cron:
            return self.cron_is_due(now)
        return False

    def cron_is_due(self, now: datetime) -> bool:
        try:
            from croniter import croniter
        except ImportError:
            logger.warning("croniter not installed — cannot evaluate cron expression %r", self.cron)
            return False
        base = datetime.fromisoformat(self.created_at)
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        it = croniter(self.cron, base)
        next_fire = it.get_next(datetime)
        if next_fire.tzinfo is None:
            next_fire = next_fire.replace(tzinfo=timezone.utc)
        return now >= next_fire

    def advance_cron(self) -> None:
        """After firing a cron task, update created_at so the next occurrence is computed correctly."""
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class SchedulerAgent(Agent):
    """Mixin that lets an agent schedule tool calls and conversations for future execution.

    Tasks are persisted to a JSON file and executed by a background thread
    or manual :meth:`tick` calls.

    Parameters
    ----------
    schedule_file:
        Path to the JSON file for task persistence.  Defaults to ``"scheduled_tasks.json"``.
    poll_interval:
        Seconds between background-thread polls.  Defaults to ``10``.
    """

    def __init__(
        self,
        *args: Any,
        schedule_file: str = "scheduled_tasks.json",
        poll_interval: float = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.schedule_file = Path(schedule_file)
        self.poll_interval = poll_interval
        self.scheduler_thread: threading.Thread | None = None
        self.scheduler_stop_event = threading.Event()
        self.scheduler_lock = threading.Lock()

        self.functions["schedule_task"] = Function(self.schedule_task)
        self.functions["schedule_conversation"] = Function(self.schedule_conversation)
        self.functions["list_scheduled_tasks"] = Function(self.list_scheduled_tasks)
        self.functions["cancel_scheduled_task"] = Function(self.cancel_scheduled_task)

    # -- JSON persistence --------------------------------------------------

    def load_tasks(self) -> list[ScheduledTask]:
        with self.scheduler_lock:
            if not self.schedule_file.exists():
                return []
            try:
                data = json.loads(self.schedule_file.read_text())
                return [ScheduledTask.from_dict(t) for t in data]
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt schedule file %s — starting empty", self.schedule_file)
                return []

    def save_tasks(self, tasks: list[ScheduledTask]) -> None:
        with self.scheduler_lock:
            self.schedule_file.write_text(
                json.dumps([t.to_dict() for t in tasks], indent=2, ensure_ascii=False)
            )

    def add_task(self, task: ScheduledTask) -> None:
        tasks = self.load_tasks()
        tasks.append(task)
        self.save_tasks(tasks)

    # -- LLM-facing tools --------------------------------------------------

    def schedule_task(
        self,
        function_name: str,
        arguments: dict,
        run_at: str = "",
        cron: str = "",
    ) -> str:
        """Schedule a tool call for future execution.

        Provide *run_at* (ISO datetime like '2026-04-04T09:00:00') for a
        one-shot task, or *cron* (e.g. '0 9 * * 1' for every Monday at 9am)
        for recurring execution.
        """
        if not run_at and not cron:
            return "Error: provide either run_at or cron"
        if function_name not in self.functions:
            return f"Error: unknown function {function_name!r}"

        task = ScheduledTask(
            id=uuid.uuid4().hex[:8],
            type="tool_call",
            function_name=function_name,
            arguments=arguments,
            run_at=run_at or None,
            cron=cron or None,
        )
        self.add_task(task)
        when = f"at {run_at}" if run_at else f"recurring ({cron})"
        logger.info("Scheduled %s(%s) %s [id=%s]", function_name, arguments, when, task.id)
        return f"Scheduled task {task.id}: {function_name} {when}"

    def schedule_conversation(
        self,
        prompt: str,
        run_at: str = "",
        cron: str = "",
    ) -> str:
        """Schedule a future conversation — the agent will reason about *prompt* at execution time.

        Useful when the data needed isn't available yet (e.g. 'check latest
        sales and email a report').  Provide *run_at* or *cron*.
        The conversation's message history is captured so the agent has
        full context when the task fires.
        """
        if not run_at and not cron:
            return "Error: provide either run_at or cron"

        conv = active_conversation.get(None)
        messages = list(conv.messages) if conv else None

        task = ScheduledTask(
            id=uuid.uuid4().hex[:8],
            type="conversation",
            prompt=prompt,
            context_messages=messages,
            run_at=run_at or None,
            cron=cron or None,
        )
        self.add_task(task)
        when = f"at {run_at}" if run_at else f"recurring ({cron})"
        logger.info("Scheduled conversation %r %s [id=%s]", prompt[:60], when, task.id)
        return f"Scheduled conversation {task.id}: {when}"

    def list_scheduled_tasks(self) -> str:
        """List all pending scheduled tasks."""
        tasks = [t for t in self.load_tasks() if t.status == "pending"]
        if not tasks:
            return "No pending scheduled tasks."
        lines = []
        for t in tasks:
            when = t.run_at or t.cron or "unknown"
            if t.type == "tool_call":
                lines.append(f"  [{t.id}] {t.function_name}({t.arguments}) — {when}")
            else:
                lines.append(f"  [{t.id}] conversation: {t.prompt!r} — {when}")
        return f"{len(tasks)} pending task(s):\n" + "\n".join(lines)

    def cancel_scheduled_task(self, task_id: str) -> str:
        """Cancel a scheduled task by its ID."""
        tasks = self.load_tasks()
        for t in tasks:
            if t.id == task_id and t.status == "pending":
                t.status = "cancelled"
                self.save_tasks(tasks)
                logger.info("Cancelled task %s", task_id)
                return f"Cancelled task {task_id}"
        return f"No pending task found with id {task_id!r}"

    # -- Execution ---------------------------------------------------------

    def tick(self) -> list[str]:
        """Check for due tasks and execute them.  Returns a list of result strings.

        Call this manually from a cron job or let :meth:`start_scheduler`
        run it automatically in a background thread.
        """
        tasks = self.load_tasks()
        now = datetime.now(timezone.utc)
        results: list[str] = []
        changed = False

        for task in tasks:
            if not task.is_due(now):
                continue

            if task.type == "tool_call":
                result = self.execute_tool_task(task)
            elif task.type == "conversation":
                result = self.execute_conversation_task(task)
            else:
                result = f"Unknown task type: {task.type}"

            results.append(result)
            changed = True

            if task.cron:
                task.advance_cron()
            else:
                task.status = "done"

        if changed:
            self.save_tasks(tasks)
        return results

    def execute_tool_task(self, task: ScheduledTask) -> str:
        logger.info("Executing scheduled tool call %s [id=%s]", task.function_name, task.id)
        try:
            func = self.functions.get(task.function_name or "")
            if not func:
                return f"Error: function {task.function_name!r} not found"
            result = str(func(**(task.arguments or {})))
            logger.info("Scheduled task %s result: %s", task.id, result[:200])
            return result
        except Exception as exc:
            logger.exception("Scheduled task %s failed", task.id)
            return f"Error executing {task.function_name}: {exc}"

    def execute_conversation_task(self, task: ScheduledTask) -> str:
        logger.info("Executing scheduled conversation [id=%s]: %s", task.id, (task.prompt or "")[:80])
        try:
            conv = self.start_conversation()
            if task.context_messages:
                conv.messages = list(task.context_messages)
            response = conv.ask(task.prompt or "")
            logger.info("Scheduled conversation %s result: %s", task.id, response.message.content[:200])
            return response.message.content
        except Exception as exc:
            logger.exception("Scheduled conversation %s failed", task.id)
            return f"Error in scheduled conversation: {exc}"

    def mark_missed_tasks(self) -> None:
        """Mark one-shot tasks whose run_at has passed as 'missed'."""
        tasks = self.load_tasks()
        now = datetime.now(timezone.utc)
        changed = False
        for task in tasks:
            if task.status != "pending" or not task.run_at:
                continue
            target = datetime.fromisoformat(task.run_at)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            if now > target:
                task.status = "missed"
                logger.info("Marking overdue task %s as missed", task.id)
                changed = True
        if changed:
            self.save_tasks(tasks)

    # -- Background thread -------------------------------------------------

    def start_scheduler(self) -> None:
        """Start the background thread that polls for due tasks."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return
        self.scheduler_stop_event.clear()
        self.mark_missed_tasks()
        self.scheduler_thread = threading.Thread(
            target=self.scheduler_loop, daemon=True, name="pygentix-scheduler",
        )
        self.scheduler_thread.start()
        logger.info("Scheduler started (poll_interval=%.1fs)", self.poll_interval)

    def stop_scheduler(self) -> None:
        """Stop the background thread."""
        self.scheduler_stop_event.set()
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=self.poll_interval + 2)
            self.scheduler_thread = None
        logger.info("Scheduler stopped")

    def scheduler_loop(self) -> None:
        while not self.scheduler_stop_event.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("Scheduler tick failed")
            self.scheduler_stop_event.wait(self.poll_interval)
