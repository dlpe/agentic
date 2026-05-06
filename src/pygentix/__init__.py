"""pygentix — Composable AI agent framework with tool-calling, structured output, and SQLAlchemy integration."""

from .core import Agent, ChatResponse, Conversation, Function, Usage
from .ollama import Ollama
from .chatgpt import ChatGPT
from .gemini import Gemini
from .copilot import Copilot
from .claude import Claude
from .output import OutputAgent
from .sqlalchemy import SqlAlchemyAgent, describe_model
from .scheduler import SchedulerAgent
from .tabular import TabularStore, reduce_tabular_rows, register_tabular_tools

__all__ = [
    "Agent",
    "ChatResponse",
    "Conversation",
    "Function",
    "Usage",
    "ChatGPT",
    "Claude",
    "Copilot",
    "Gemini",
    "Ollama",
    "OutputAgent",
    "SchedulerAgent",
    "SqlAlchemyAgent",
    "describe_model",
    "TabularStore",
    "reduce_tabular_rows",
    "register_tabular_tools",
]
