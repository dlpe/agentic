"""SQLAlchemy database integration for agents."""

import logging
from datetime import date, datetime
from typing import Any

from sqlalchemy import inspect as sa_inspect, select, types as sa_types
from sqlalchemy.orm import Session

from .core import Agent, Function, active_scope

__all__ = ["SqlAlchemyAgent", "describe_model"]

logger = logging.getLogger("pygentix")

QUERY_REFERENCE = """\
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
        self.writable: dict[str, type] = {}
        self.entity_scope: dict[str, dict] = {}
        self.entity_scope_chain: dict[str, list] = {}

    # -- decorators --------------------------------------------------------

    def reads(
        self,
        cls: type,
        *,
        scope: dict[str, str] | None = None,
        scope_chain: list[tuple] | None = None,
    ) -> type:
        """Register a SQLAlchemy model for read access (``run_query``).

        Parameters
        ----------
        scope:
            Direct column-to-scope mapping, e.g.
            ``{"user_id": "current_user"}`` means *WHERE user_id = scope["current_user"]*.
        scope_chain:
            Multi-level ownership chain for tables without a direct scope
            column. Each element is ``(fk_column, TargetModel)`` except
            the last which is ``(scope_column, scope_key)``.
        """
        self.entities[cls] = None
        self.entities_by_name[cls.__name__] = cls
        self.store_scope(cls.__name__, scope, scope_chain)
        self.functions["run_query"] = Function(self.run_query)
        return cls

    def writes(
        self,
        cls: type,
        *,
        scope: dict[str, str] | None = None,
        scope_chain: list[tuple] | None = None,
    ) -> type:
        """Register a SQLAlchemy model for write access (insert/update/delete).

        Accepts the same *scope* / *scope_chain* parameters as :meth:`reads`.
        """
        self.entities[cls] = None
        self.writable[cls.__name__] = cls
        self.store_scope(cls.__name__, scope, scope_chain)
        for name in ("run_insert", "run_update", "run_delete"):
            self.functions[name] = Function(getattr(self, name))
        return cls

    def store_scope(
        self,
        name: str,
        scope: dict[str, str] | None,
        scope_chain: list[tuple] | None,
    ) -> None:
        if scope:
            self.entity_scope[name] = scope
        if scope_chain:
            self.entity_scope_chain[name] = scope_chain

    # -- scope helpers -----------------------------------------------------

    def get_scope(self) -> dict:
        """Return the active conversation scope, or empty dict."""
        return active_scope.get() or {}

    def resolve_scope_filters(self, entity_name: str) -> dict[str, Any]:
        """Resolve direct-scope column values from the active scope.

        Returns a dict of ``{column: value}`` to inject into queries,
        or an empty dict if no direct scope is defined for *entity_name*.
        """
        mapping = self.entity_scope.get(entity_name)
        if not mapping:
            return {}
        scope = self.get_scope()
        if not scope:
            return {}
        filters: dict[str, Any] = {}
        for col, scope_key in mapping.items():
            if scope_key in scope:
                filters[col] = scope[scope_key]
        return filters

    def apply_scope_to_query(
        self, query: Any, entity_cls: type, entity_name: str,
    ) -> Any:
        """Apply direct scope (A) or scope chain (B) filters to a SELECT."""
        direct = self.resolve_scope_filters(entity_name)
        if direct:
            for col_name, val in direct.items():
                query = query.filter(getattr(entity_cls, col_name) == val)
            return query

        chain = self.entity_scope_chain.get(entity_name)
        if chain:
            query = self.apply_chain_to_query(query, entity_cls, chain)
        return query

    def apply_chain_to_query(
        self, query: Any, root_cls: type, chain: list[tuple],
    ) -> Any:
        """Walk a scope_chain and add JOINs + a final WHERE to *query*."""
        scope = self.get_scope()
        if not scope:
            return query

        current_cls = root_cls
        for i, link in enumerate(chain):
            is_last = i == len(chain) - 1
            if is_last:
                col_name, scope_key = link
                if scope_key in scope:
                    query = query.filter(
                        getattr(current_cls, col_name) == scope[scope_key],
                    )
            else:
                fk_col_name, target_cls = link[0], link[1]
                pk_col = sa_inspect(target_cls).primary_key[0]
                query = query.join(
                    target_cls,
                    getattr(current_cls, fk_col_name) == getattr(target_cls, pk_col.key),
                )
                current_cls = target_cls
        return query

    def validate_scope_insert(self, entity_name: str, record: dict) -> dict:
        """Enforce scope on an INSERT: auto-set or validate direct columns,
        check FK ownership for chains.  Returns the (possibly augmented) record.
        """
        scope = self.get_scope()
        if not scope:
            return record

        direct = self.entity_scope.get(entity_name)
        if direct:
            for col, scope_key in direct.items():
                if scope_key not in scope:
                    continue
                expected = scope[scope_key]
                if col in record and record[col] != expected:
                    raise PermissionError(
                        f"Cannot set {col}={record[col]!r}; "
                        f"scope requires {col}={expected!r}"
                    )
                record[col] = expected
            return record

        chain = self.entity_scope_chain.get(entity_name)
        if chain:
            self.validate_chain_insert(entity_name, record, chain, scope)
        return record

    def validate_chain_insert(
        self,
        entity_name: str,
        record: dict,
        chain: list[tuple],
        scope: dict,
    ) -> None:
        """Verify that referenced FK in *record* belongs to the scoped user."""
        if not chain:
            return
        fk_col_name = chain[0][0]
        fk_value = record.get(fk_col_name)
        if fk_value is None:
            return

        target_cls = chain[0][1]
        remaining = chain[1:]

        with Session(self.engine) as session:
            q = select(target_cls)
            pk_col = sa_inspect(target_cls).primary_key[0]
            q = q.filter(getattr(target_cls, pk_col.key) == fk_value)

            current_cls = target_cls
            for i, link in enumerate(remaining):
                is_last = i == len(remaining) - 1
                if is_last:
                    col_name, scope_key = link
                    if scope_key in scope:
                        q = q.filter(
                            getattr(current_cls, col_name) == scope[scope_key],
                        )
                else:
                    fk_name, next_cls = link[0], link[1]
                    next_pk = sa_inspect(next_cls).primary_key[0]
                    q = q.join(
                        next_cls,
                        getattr(current_cls, fk_name) == getattr(next_cls, next_pk.key),
                    )
                    current_cls = next_cls

            if not session.execute(q).first():
                raise PermissionError(
                    f"Cannot insert into {entity_name}: "
                    f"{fk_col_name}={fk_value!r} is not owned by the current scope"
                )

    def scope_filters_for_mutation(self, entity_name: str, filters: dict) -> dict:
        """Augment *filters* with direct scope columns for update/delete."""
        direct = self.resolve_scope_filters(entity_name)
        if direct:
            for col, val in direct.items():
                if col in filters and filters[col] != val:
                    raise PermissionError(
                        f"Cannot target {col}={filters[col]!r}; "
                        f"scope restricts to {col}={val!r}"
                    )
                filters[col] = val
        return filters

    def validate_chain_mutation(
        self, entity_name: str, cls: type, filters: dict,
    ) -> None:
        """For scope_chain entities, verify that matched rows belong to scope."""
        chain = self.entity_scope_chain.get(entity_name)
        scope = self.get_scope()
        if not chain or not scope:
            return
        with Session(self.engine) as session:
            q = select(cls).filter_by(**filters)
            q = self.apply_chain_to_query(q, cls, chain)
            direct_count = session.query(cls).filter_by(**filters).count()
            scoped_count = len(session.execute(q).fetchall())
            if direct_count > 0 and scoped_count == 0:
                raise PermissionError(
                    f"Cannot modify {entity_name}: "
                    f"matched rows are not owned by the current scope"
                )

    # -- CRUD tools --------------------------------------------------------

    def run_query(self, query_steps: list[dict]) -> list[dict]:
        """Query an entity using a list of filter/modifier steps."""
        entity_name = query_steps[0]["entity"]
        entity_cls = self.entities_by_name[entity_name]
        query = select(entity_cls)

        query = self.apply_scope_to_query(query, entity_cls, entity_name)

        for step in query_steps:
            if not step.get("op"):
                continue
            query = self.apply_step(query, step.get("field"), step["op"], step.get("value"))

        columns = sa_inspect(entity_cls).columns
        with Session(self.engine) as session:
            rows = session.execute(query).fetchall()
            return [self.row_to_dict(row, entity_cls, columns) for row in rows]

    def run_insert(self, entity: str, values: dict | list[dict]) -> str:
        """Insert one or many rows.  Accepts a dict or a list of dicts."""
        cls = self.writable[entity]
        records = values if isinstance(values, list) else [values]
        columns = sa_inspect(cls).columns
        inserted = []

        with Session(self.engine) as session:
            for record in records:
                record = self.validate_scope_insert(entity, dict(record))
                obj = cls(**self.coerce_values(cls, record))
                session.add(obj)
                session.flush()
                inserted.append({c.key: getattr(obj, c.key) for c in columns})
            session.commit()

        return f"Inserted {len(inserted)} row(s) into {entity}: {inserted}"

    def run_update(self, entity: str, filters: dict, values: dict) -> str:
        """Update rows matching *filters* with new *values*."""
        cls = self.writable[entity]
        filters = self.coerce_values(cls, dict(filters))
        values = self.coerce_values(cls, values)

        filters = self.scope_filters_for_mutation(entity, filters)
        self.validate_chain_mutation(entity, cls, filters)

        with Session(self.engine) as session:
            count = session.query(cls).filter_by(**filters).update(values)
            session.commit()

        return f"Updated {count} row(s) in {entity} where {filters}, set {values}"

    def run_delete(self, entity: str, filters: dict) -> str:
        """Delete rows matching *filters*."""
        cls = self.writable[entity]
        filters = self.coerce_values(cls, dict(filters))

        filters = self.scope_filters_for_mutation(entity, filters)
        self.validate_chain_mutation(entity, cls, filters)

        with Session(self.engine) as session:
            count = session.query(cls).filter_by(**filters).delete()
            session.commit()

        return f"Deleted {count} row(s) from {entity} where {filters}"

    # -- conversation setup ------------------------------------------------

    def start_conversation(self, **kwargs: Any):
        """Build a system prompt with entity descriptions and query reference.

        Accepts the same *scope* and *policy* keyword arguments as
        :meth:`Agent.start_conversation`.
        """
        self.resolve_entities()

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
            f"{QUERY_REFERENCE}"
        )
        return super().start_conversation(system=system, **kwargs)

    # -- internals ---------------------------------------------------------

    def apply_step(self, query: Any, field: str | None, op: str, value: Any) -> Any:
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
    def row_to_dict(row: Any, entity_cls: type, columns: Any) -> dict:
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

    def coerce_values(self, cls: type, values: dict) -> dict:
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

    def resolve_entities(self) -> None:
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
