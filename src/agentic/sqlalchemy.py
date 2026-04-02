"""SQLAlchemy database integration for agents."""

from datetime import date, datetime
from typing import Any

from sqlalchemy import inspect as sa_inspect, select, types as sa_types
from sqlalchemy.orm import Session

from .core import Agent, Function

__all__ = ["SqlAlchemyAgent", "describe_model"]

_QUERY_REFERENCE = """\
Supported query operators: eq, gt, lt, gte, lte, like, ilike, in, not_in, is_null, is_not_null
Supported joins: left, right, inner, outer
Supported modifiers: asc, desc, limit, offset

Examples:
  run_query([{{"entity": "Author", "field": "name", "op": "eq", "value": "Tolkien"}}])
  run_query([{{"entity": "Author"}}, {{"field": "name", "op": "like", "value": "%Tolk%"}}])
  run_insert("Author", {{"name": "Tolkien"}})
  run_insert("Author", [{{"name": "A"}}, {{"name": "B"}}])  # batch insert
  run_update("Author", {{"name": "Tolkien"}}, {{"name": "J.R.R. Tolkien"}})
  run_delete("Author", {{"name": "Tolkien"}})
"""


class SqlAlchemyAgent(Agent):
    """Mixin that gives an agent read/write access to SQLAlchemy models.

    Register models with the :meth:`reads` and :meth:`writes` decorators.
    The agent automatically gains ``run_query``, ``run_insert``,
    ``run_update``, and ``run_delete`` tools that the LLM can call.
    """

    OPERATOR_MAP: dict[str, Any] = {
        "eq": "__eq__",
        "gt": "__gt__",
        "lt": "__lt__",
        "gte": "__ge__",
        "lte": "__le__",
        "like": "like",
        "ilike": "ilike",
        "in": "in_",
        "not_in": "notin_",
        "is": "is_",
        "is_not": "is_not",
        "is_null": lambda col: col.is_(None),
        "is_not_null": lambda col: col.is_not(None),
    }

    def __init__(self, engine: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.entities: dict[type, dict | None] = {}
        self.entities_by_name: dict[str, type] = {}
        self._writable: dict[str, type] = {}

    # -- decorators --------------------------------------------------------

    def reads(self, cls: type) -> type:
        """Register a SQLAlchemy model for read access (``run_query``)."""
        self.entities[cls] = None
        self.entities_by_name[cls.__name__] = cls
        self.functions["run_query"] = Function(self.run_query)
        return cls

    def writes(self, cls: type) -> type:
        """Register a SQLAlchemy model for write access (insert/update/delete)."""
        self.entities[cls] = None
        self._writable[cls.__name__] = cls
        for name in ("run_insert", "run_update", "run_delete"):
            self.functions[name] = Function(getattr(self, name))
        return cls

    # -- CRUD tools --------------------------------------------------------

    def run_query(self, query_steps: list[dict]) -> list[dict]:
        """Query an entity using a list of filter/modifier steps."""
        entity_cls = self.entities_by_name[query_steps[0]["entity"]]
        query = select(entity_cls)

        for step in query_steps:
            if not step.get("op"):
                continue
            query = self._apply_step(query, step.get("field"), step["op"], step.get("value"))

        columns = sa_inspect(entity_cls).columns
        with Session(self.engine) as session:
            rows = session.execute(query).fetchall()
            return [self._row_to_dict(row, entity_cls, columns) for row in rows]

    def run_insert(self, entity: str, values: dict | list[dict]) -> str:
        """Insert one or many rows.  Accepts a dict or a list of dicts."""
        cls = self._writable[entity]
        records = values if isinstance(values, list) else [values]
        columns = sa_inspect(cls).columns
        inserted = []

        with Session(self.engine) as session:
            for record in records:
                obj = cls(**self._coerce_values(cls, record))
                session.add(obj)
                session.flush()
                inserted.append({c.key: getattr(obj, c.key) for c in columns})
            session.commit()

        return f"Inserted {len(inserted)} row(s) into {entity}: {inserted}"

    def run_update(self, entity: str, filters: dict, values: dict) -> str:
        """Update rows matching *filters* with new *values*."""
        cls = self._writable[entity]
        filters = self._coerce_values(cls, filters)
        values = self._coerce_values(cls, values)

        with Session(self.engine) as session:
            count = session.query(cls).filter_by(**filters).update(values)
            session.commit()

        return f"Updated {count} row(s) in {entity} where {filters}, set {values}"

    def run_delete(self, entity: str, filters: dict) -> str:
        """Delete rows matching *filters*."""
        cls = self._writable[entity]
        filters = self._coerce_values(cls, filters)

        with Session(self.engine) as session:
            count = session.query(cls).filter_by(**filters).delete()
            session.commit()

        return f"Deleted {count} row(s) from {entity} where {filters}"

    # -- conversation setup ------------------------------------------------

    def start_conversation(self, **kwargs: Any):
        """Build a system prompt with entity descriptions and query reference."""
        self._resolve_entities()

        entity_lines = []
        for cls, desc in self.entities.items():
            cols = ", ".join(c["name"] for c in desc["columns"])
            rels = ", ".join(r["name"] for r in desc["relationships"])
            line = f"  {cls.__name__} — columns: [{cols}]"
            if rels:
                line += f"  relationships: [{rels}]"
            entity_lines.append(line)

        base = kwargs.pop("system", "")
        system = (
            f"{base}\n"
            f"Database entities:\n" + "\n".join(entity_lines) + "\n\n"
            f"{_QUERY_REFERENCE}"
        )
        return super().start_conversation(system=system, **kwargs)

    # -- internals ---------------------------------------------------------

    def _apply_step(self, query: Any, field: str | None, op: str, value: Any) -> Any:
        """Apply a single filter / modifier step to a SQLAlchemy query."""
        model = query.column_descriptions[0]["entity"]
        col = getattr(model, field) if field else None

        if op in self.OPERATOR_MAP:
            mapped = self.OPERATOR_MAP[op]
            if callable(mapped) and not isinstance(mapped, str):
                return query.filter(mapped(col))
            return query.filter(getattr(col, mapped)(value))
        if op in ("left", "right", "inner", "outer"):
            return query.join(getattr(model, value))
        if op in ("asc", "desc"):
            return query.order_by(getattr(col, op)())
        if op == "limit":
            return query.limit(value)
        if op == "offset":
            return query.offset(value)

        raise ValueError(f"Unsupported operator: {op}")

    @staticmethod
    def _row_to_dict(row: Any, entity_cls: type, columns: Any) -> dict:
        """Convert a SQLAlchemy result row to a plain dict."""
        mapping = row._mapping
        entity_name = entity_cls.__name__

        if entity_name in mapping and hasattr(mapping[entity_name], "__table__"):
            obj = mapping[entity_name]
            result = {}
            for col in columns:
                val = getattr(obj, col.key)
                if isinstance(val, (date, datetime)):
                    val = val.isoformat()
                result[col.key] = val
            return result

        return dict(mapping)

    def _coerce_values(self, cls: type, values: dict) -> dict:
        """Cast string values to the types expected by the model's columns."""
        mapper = sa_inspect(cls)
        col_map = {c.key: c for c in mapper.columns}
        coerced = {}

        for key, val in values.items():
            col = col_map.get(key)
            if col is not None and isinstance(val, str):
                col_type = type(col.type)
                if col_type in (sa_types.Integer, sa_types.BigInteger, sa_types.SmallInteger):
                    val = int(val)
                elif col_type in (sa_types.Float, sa_types.Numeric):
                    val = float(val)
                elif col_type is sa_types.Date:
                    val = date.fromisoformat(val)
                elif col_type is sa_types.DateTime:
                    val = datetime.fromisoformat(val)
                elif col_type is sa_types.Boolean:
                    val = val.lower() in ("true", "1", "yes")
            coerced[key] = val

        return coerced

    def _resolve_entities(self) -> None:
        """Populate deferred entity descriptions via :func:`describe_model`."""
        for cls in self.entities:
            if self.entities[cls] is None:
                self.entities[cls] = describe_model(cls)


def describe_model(model_cls: type) -> dict:
    """Introspect a SQLAlchemy ORM model and return a serialisable description.

    Returns a dict with ``model``, ``table``, ``columns``, and
    ``relationships`` keys suitable for embedding in an LLM system prompt.
    """
    mapper = sa_inspect(model_cls)

    columns = [
        {
            "name": col.key,
            "type": str(col.type),
            "nullable": col.nullable,
            "primary_key": col.primary_key,
            "foreign_keys": [
                {"target": fk.target_fullname, "column": str(fk.column)}
                for fk in col.foreign_keys
            ],
        }
        for col in mapper.columns
    ]

    relationships = [
        {
            "name": rel.key,
            "target_class": rel.mapper.class_.__name__,
            "direction": rel.direction.name,
            "uselist": rel.uselist,
            "local_columns": [c.key for c in rel.local_columns],
            "remote_side": [c.key for c in rel.remote_side],
            "foreign_keys": [c.key for c in rel._calculated_foreign_keys],
        }
        for rel in mapper.relationships
    ]

    return {
        "model": model_cls.__name__,
        "table": str(mapper.persist_selectable),
        "columns": columns,
        "relationships": relationships,
    }
