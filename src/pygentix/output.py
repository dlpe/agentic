"""Structured output enforcement for agent responses."""

import json
from typing import Any, get_args, get_origin, get_type_hints

from .core import Agent

__all__ = ["OutputAgent"]


class OutputAgent(Agent):
    """Mixin that constrains agent responses to a predefined JSON schema.

    Use the :meth:`output` decorator to register a schema — either a raw
    JSON-schema ``dict`` or a plain Python class whose type annotations
    are converted automatically::

        @agent.output
        class Answer:
            text: str
            confidence: float = 0.0

    When a schema is set, :meth:`start_conversation` injects a hint into
    the system prompt and the conversation pipeline applies an Ollama
    ``format`` constraint on the final response.
    """

    PYTHON_TO_JSON: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.output_schema: dict | None = None
        self.output_cls: type | None = None

    # -- public API --------------------------------------------------------

    def output(self, schema: dict | type) -> dict | type:
        """Set the output schema.  Works as a decorator or a direct call."""
        if isinstance(schema, dict):
            self.output_schema = schema
        else:
            self.output_cls = schema
            self.output_schema = self.schema_from_class(schema)
        return schema

    def parse_output(self, response: Any) -> Any:
        """Parse a model response into a ``dict`` or a class instance.

        Falls back to the raw content string if the response isn't valid JSON.
        """
        try:
            data = json.loads(response.message.content)
        except (json.JSONDecodeError, TypeError):
            return response.message.content

        if not self.output_cls:
            return data

        try:
            return self.output_cls(**data)
        except TypeError:
            obj = object.__new__(self.output_cls)
            for key, value in data.items():
                setattr(obj, key, value)
            return obj

    def start_conversation(self, **kwargs: Any):
        """Inject a schema-enforcement hint into the system prompt."""
        if self.output_schema:
            hint = (
                "\n\nYour final responses MUST be valid JSON matching this schema: "
                f"{json.dumps(self.output_schema)}\n"
                "ALWAYS include the complete actual data returned by tools. "
                "Never give vague summaries — put the real values in your response."
            )
            kwargs["system"] = kwargs.get("system", "") + hint
        return super().start_conversation(**kwargs)

    # -- schema generation -------------------------------------------------

    @classmethod
    def type_to_json(cls, tp: type) -> dict:
        """Convert a Python type annotation to a JSON-schema fragment."""
        origin = get_origin(tp)
        if origin is list:
            args = get_args(tp)
            return {"type": "array", "items": cls.type_to_json(args[0]) if args else {}}
        if origin is dict:
            return {"type": "object"}
        return {"type": cls.PYTHON_TO_JSON.get(tp, "string")}

    @classmethod
    def schema_from_class(cls, klass: type) -> dict:
        """Derive a JSON schema from a class's type annotations."""
        hints = get_type_hints(klass)
        properties = {name: cls.type_to_json(tp) for name, tp in hints.items()}
        class_vars = vars(klass)
        required = [name for name in hints if name not in class_vars]
        return {"type": "object", "properties": properties, "required": required}
