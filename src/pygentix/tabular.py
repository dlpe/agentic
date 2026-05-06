"""Conversation-scoped tabular data: register rows once, reduce without re-sending bulk JSON.

**Pattern:** A heavy tool (or your own code) builds ``list[dict]`` rows on the server.
Call ``tabular_register`` to store them under the active :data:`pygentix.core.active_conversation`
and receive a short ``dataset_id``. The model then calls ``tabular_reduce`` with a small JSON
spec (group_by, metrics, optional where clauses) and only sees the aggregated result.

This keeps tool *results* small while preserving flexibility: the LLM chooses ops and fields,
not raw rows. Reduction runs locally (no extra LLM round-trip for the bulk payload).

Adapter: storage hangs off the current :class:`~pygentix.core.Conversation` instance so
multi-tenant / multi-user isolation follows whatever scope you already enforce in tools.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable

from .core import Agent, active_conversation

if TYPE_CHECKING:
    from .core import Conversation

__all__ = [
    "TabularStore",
    "reduce_tabular_rows",
    "register_tabular_tools",
]

logger = logging.getLogger("pygentix")

_TABULAR_ATTR = "_pygentix_tabular_store"

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_ident(name: str, ctx: str) -> None:
    if not IDENT_RE.match(name):
        raise ValueError(
            f"Invalid {ctx} field name {name!r} (allowed: letters, digits, underscore)"
        )


class TabularStore:
    """In-memory row store for one conversation (not shared across workers)."""

    def __init__(
        self,
        *,
        max_rows_per_dataset: int = 100_000,
        max_datasets: int = 32,
    ) -> None:
        self.max_rows_per_dataset = max_rows_per_dataset
        self.max_datasets = max_datasets
        self.datasets: dict[str, list[dict[str, Any]]] = {}

    def register(self, rows: Any) -> tuple[str, int, list[str]]:
        """*rows* is validated at runtime (LLM tool calls may not match annotations)."""
        if not isinstance(rows, list):
            raise TypeError("tabular_register: rows must be a list of dict objects")
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"tabular_register: rows[{i}] must be a dict")
        if len(rows) > self.max_rows_per_dataset:
            raise ValueError(
                f"tabular_register: {len(rows)} rows exceeds max_rows_per_dataset={self.max_rows_per_dataset}"
            )

        while len(self.datasets) >= self.max_datasets:
            first = next(iter(self.datasets))
            del self.datasets[first]
            logger.debug("tabular_store evicted oldest dataset %s", first)

        did = f"ds_{uuid.uuid4().hex[:24]}"
        self.datasets[did] = [dict(r) for r in rows]
        columns: list[str] = []
        if rows:
            columns = sorted({k for r in rows for k in r.keys()})
        return did, len(rows), columns

    def get(self, dataset_id: str) -> list[dict[str, Any]]:
        if dataset_id not in self.datasets:
            raise KeyError(
                f"Unknown dataset_id {dataset_id!r} (expired, wrong id, or not this conversation)"
            )
        return self.datasets[dataset_id]


def store_for_conversation(
    conv: "Conversation",
    *,
    max_rows_per_dataset: int,
    max_datasets: int,
) -> TabularStore:
    store = getattr(conv, _TABULAR_ATTR, None)
    if store is None:
        store = TabularStore(
            max_rows_per_dataset=max_rows_per_dataset,
            max_datasets=max_datasets,
        )
        setattr(conv, _TABULAR_ATTR, store)
    return store


def _coerce_num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _apply_where(
    rows: list[dict[str, Any]], clauses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not clauses:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        ok = True
        for c in clauses:
            field = c.get("field")
            op = (c.get("op") or "eq").lower()
            val = c.get("value")
            if not isinstance(field, str):
                ok = False
                break
            validate_ident(field, "where")
            rv = row.get(field)
            if op == "eq":
                if rv != val:
                    ok = False
            elif op == "ne":
                if rv == val:
                    ok = False
            elif op in ("gt", "gte", "lt", "lte"):
                a = _coerce_num(rv)
                b = _coerce_num(val)
                if a is None or b is None:
                    ok = False
                elif op == "gt" and not (a > b):
                    ok = False
                elif op == "gte" and not (a >= b):
                    ok = False
                elif op == "lt" and not (a < b):
                    ok = False
                elif op == "lte" and not (a <= b):
                    ok = False
            elif op == "in":
                if not isinstance(val, list) or rv not in val:
                    ok = False
            else:
                raise ValueError(f"Unsupported where op {op!r}")
            if not ok:
                break
        if ok:
            out.append(row)
    return out


def _metric_values(rows: list[dict[str, Any]], field: str | None) -> list[Any]:
    if field is None:
        return []
    validate_ident(field, "metric")
    return [r.get(field) for r in rows]


def _run_metric(rows: list[dict[str, Any]], op: str, field: str | None) -> Any:
    op = (op or "").lower()
    if op == "count" and field is None:
        return len(rows)
    vals = _metric_values(rows, field)
    if op == "count":
        return sum(1 for v in vals if v is not None)
    nums = [_coerce_num(v) for v in vals]
    nums = [n for n in nums if n is not None]
    if op == "sum":
        return float(sum(nums)) if nums else None
    if op == "avg":
        return float(sum(nums) / len(nums)) if nums else None
    if op == "min":
        return min(nums) if nums else None
    if op == "max":
        return max(nums) if nums else None
    raise ValueError(f"Unsupported metric op {op!r}")


def reduce_tabular_rows(
    rows: list[dict[str, Any]], spec: dict[str, Any]
) -> list[dict[str, Any]]:
    """Pure reduction over in-memory row dicts (no conversation required).

    *spec* keys:

    - ``where`` (optional): list of ``{field, op, value}`` with ``op`` in
      ``eq``, ``ne``, ``gt``, ``gte``, ``lt``, ``lte``, ``in``.
    - ``group_by`` (optional): list of field names. Empty = one aggregate row.
    - ``metrics`` (required): list of ``{op, field?, as?}`` with ``op`` in
      ``sum``, ``count``, ``avg``, ``min``, ``max``. For ``count``, omit ``field``
      to count rows; with ``field``, count non-null values.

    Field names must match ``^[A-Za-z_][A-Za-z0-9_]*$``.
    """
    if not isinstance(spec, dict):
        raise TypeError("reduce_tabular_rows: spec must be a dict")
    metrics = spec.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("reduce_tabular_rows: spec.metrics must be a non-empty list")

    where = spec.get("where") or []
    if not isinstance(where, list):
        raise TypeError("reduce_tabular_rows: spec.where must be a list")
    group_by = spec.get("group_by") or []
    if not isinstance(group_by, list):
        raise TypeError("reduce_tabular_rows: spec.group_by must be a list")
    for g in group_by:
        if not isinstance(g, str):
            raise TypeError("reduce_tabular_rows: group_by entries must be strings")
        validate_ident(g, "group_by")

    filtered = _apply_where(list(rows), where)

    if not group_by:
        row_out: dict[str, Any] = {}
        for m in metrics:
            if not isinstance(m, dict):
                raise TypeError("reduce_tabular_rows: each metric must be a dict")
            op = m.get("op")
            field = m.get("field")
            label = m.get("as") or (f"{op}_{field}" if field is not None else str(op))
            row_out[str(label)] = _run_metric(filtered, str(op), field)
        return [row_out]

    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in filtered:
        key = tuple(r.get(g) for g in group_by)
        buckets[key].append(r)

    out_rows: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda kv: kv[0]):
        row_out = {g: key[i] for i, g in enumerate(group_by)}
        for m in metrics:
            if not isinstance(m, dict):
                raise TypeError("reduce_tabular_rows: each metric must be a dict")
            op = m.get("op")
            field = m.get("field")
            label = m.get("as") or (f"{op}_{field}" if field is not None else str(op))
            row_out[str(label)] = _run_metric(bucket, str(op), field)
        out_rows.append(row_out)
    return out_rows


def register_tabular_tools(
    agent: Agent | str,
    *,
    max_rows_per_dataset: int = 100_000,
    max_datasets: int = 32,
    register_name: str = "tabular_register",
    reduce_name: str = "tabular_reduce",
) -> None:
    """Register two tools on *agent* for register-then-reduce workflows.

    Call after the agent instance exists (e.g. immediately after ``Claude(name=...)``).

    Parameters
    ----------
    agent:
        Concrete :class:`~pygentix.core.Agent` instance or registry ``name`` string.
    """
    if isinstance(agent, str):
        target = Agent.registry.get(agent)
        if target is None:
            for tool_fn, meta in _tabular_tool_definitions(
                max_rows_per_dataset=max_rows_per_dataset,
                max_datasets=max_datasets,
                register_name=register_name,
                reduce_name=reduce_name,
            ):
                Agent.pending_uses.setdefault(agent, []).append((tool_fn, meta))
            return
        agent = target

    if register_name in agent.functions:
        return

    for tool_fn, meta in _tabular_tool_definitions(
        max_rows_per_dataset=max_rows_per_dataset,
        max_datasets=max_datasets,
        register_name=register_name,
        reduce_name=reduce_name,
    ):
        agent.add_tool(tool_fn, **meta)


def _tabular_tool_definitions(
    *,
    max_rows_per_dataset: int,
    max_datasets: int,
    register_name: str,
    reduce_name: str,
) -> list[tuple[Callable[..., str], dict[str, Any]]]:
    def tabular_register(rows: list[dict[str, Any]]) -> str:
        conv = active_conversation.get()
        if conv is None:
            return json.dumps(
                {
                    "error": "tabular_register requires an active Conversation (call from agent tool loop).",
                }
            )
        store = store_for_conversation(
            conv,
            max_rows_per_dataset=max_rows_per_dataset,
            max_datasets=max_datasets,
        )
        try:
            did, n, columns = store.register(rows)
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(
            {
                "dataset_id": did,
                "row_count": n,
                "columns": columns,
                "hint": "Call tabular_reduce with this dataset_id and a metrics spec; do not paste all rows back.",
            },
            default=str,
        )

    tabular_register.__doc__ = (
        "Store a list of row dicts for the current chat turn. Returns JSON with "
        "`dataset_id`, `row_count`, and `columns`. Use `tabular_reduce` next — never "
        "re-send the full row list to the model."
    )

    def tabular_reduce(dataset_id: str, spec_json: str) -> str:
        conv = active_conversation.get()
        if conv is None:
            return json.dumps(
                {
                    "error": "tabular_reduce requires an active Conversation.",
                }
            )
        store = getattr(conv, _TABULAR_ATTR, None)
        if store is None:
            return json.dumps(
                {"error": "No datasets registered in this conversation yet."}
            )
        try:
            rows = store.get(dataset_id)
            parsed = json.loads(spec_json)
            if not isinstance(parsed, dict):
                raise TypeError("spec_json must be a JSON object")
            out = reduce_tabular_rows(rows, parsed)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"rows": out}, default=str)

    tabular_reduce.__doc__ = (
        "Run group_by / metrics / where over a registered dataset. "
        "`spec_json` is a JSON object string with `metrics` (required), optional `group_by` "
        "and `where`. See pygentix.tabular.reduce_tabular_rows."
    )

    return [
        (
            tabular_register,
            {
                "name": register_name,
                "description": tabular_register.__doc__ or "",
            },
        ),
        (
            tabular_reduce,
            {
                "name": reduce_name,
                "description": tabular_reduce.__doc__ or "",
            },
        ),
    ]
