# pygentix

A composable Python framework for building AI agents with **tool-calling**, **structured output**, and **SQLAlchemy integration** — across any LLM provider.

```
pip install pygentix                    # core only
pip install pygentix[ollama]            # + Ollama backend
pip install pygentix[openai]            # + OpenAI (ChatGPT) backend
pip install pygentix[gemini]            # + Google Gemini backend
pip install pygentix[all]               # every backend
```

> **Azure OpenAI / Copilot** uses the `openai` package — install `pygentix[openai]`.

---

## Quick Start

Pick a backend, register tools, and start a conversation:

```python
from pygentix import Ollama

agent = Ollama(model="qwen2.5:7b")            # runs locally — no API key needed

@agent.uses
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 22 °C in {city}"

conv = agent.start_conversation()
response = conv.ask("What's the weather in Paris?")
print(response.message.content)
# → "It's sunny and 22 °C in Paris right now."
```

Every backend returns the same `ChatResponse` object, so switching providers is a one-line change:

```python
from pygentix import ChatGPT, Gemini, Copilot

agent = ChatGPT(model="gpt-4o-mini")          # OpenAI
agent = Gemini(model="gemini-2.5-flash")       # Google
agent = Copilot(model="gpt-4o")               # Azure OpenAI
```

---

## Backends

| Class | Provider | Default model | Install extra |
|---|---|---|---|
| `Ollama` | [Ollama](https://ollama.com) (local) | `qwen2.5:7b` | `ollama` |
| `ChatGPT` | [OpenAI](https://platform.openai.com) | `gpt-4o-mini` | `openai` |
| `Gemini` | [Google AI](https://ai.google.dev) | `gemini-2.5-flash` | `gemini` |
| `Copilot` | [Azure OpenAI](https://azure.microsoft.com/products/ai-services/openai-service) | `gpt-4o` | `openai` |

### API keys

Cloud backends read their key from the environment (or accept it in the constructor). Ollama runs locally and needs no key.

| Backend | Environment variable | Constructor kwarg |
|---|---|---|
| `Ollama` | *(none — runs locally)* | — |
| `ChatGPT` | `OPENAI_API_KEY` | `api_key` |
| `Gemini` | `GEMINI_API_KEY` | `api_key` |
| `Copilot` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | `api_key`, `endpoint` |

```python
from pygentix import ChatGPT

agent = ChatGPT(api_key="sk-...")              # explicit
agent = ChatGPT()                              # reads OPENAI_API_KEY
```

---

## Tool Calling

Decorate any Python function with `@agent.uses` to expose it as a tool the LLM can invoke:

```python
from pygentix import Ollama

agent = Ollama()

@agent.uses
def search_docs(query: str) -> str:
    """Search the documentation for relevant articles."""
    return run_search(query)

@agent.uses
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified address."""
    return mailer.send(to, subject, body)

conv = agent.start_conversation()
response = conv.ask("Find docs about authentication and email them to alice@co.com")
```

The framework introspects the function's signature and docstring to build the tool definition automatically. When the model decides to call a tool, the framework executes it and feeds the result back — looping until the model produces a final answer.

---

## Vision / Image Understanding

Pass images alongside your question to any vision-capable model:

```python
from pygentix import Ollama

agent = Ollama(model="llama3.2-vision")        # local vision model
conv = agent.start_conversation()

response = conv.ask("How many cats are in this photo?", images=["photo.jpeg"])
print(response.message.content)
# → "There are 3 cats in the photo."
```

The `images` parameter accepts a list of file paths and works across all backends:

| Backend | Vision model examples |
|---|---|
| `Ollama` | `llama3.2-vision`, `moondream` |
| `ChatGPT` | `gpt-4o`, `gpt-4o-mini` |
| `Gemini` | `gemini-2.5-flash`, `gemini-2.5-pro` |
| `Copilot` | `gpt-4o` (via Azure) |

---

## PDF Document Parsing

Combine a vision model with PyMuPDF to extract structured information from PDF documents:

```python
import fitz  # PyMuPDF — pip install pymupdf
from pygentix import Ollama, OutputAgent

class PDFAgent(Ollama, OutputAgent):
    pass

agent = PDFAgent(model="llama3.2-vision")

@agent.output
class InvoiceData:
    company: str
    invoice_number: str
    total: str
    client: str

# Render the first page to an image
doc = fitz.open("invoice.pdf")
page = doc[0]
pix = page.get_pixmap(dpi=200)
pix.save("invoice_page.png")
doc.close()

conv = agent.start_conversation()
response = conv.ask(
    "Extract the company name, invoice number, total amount, "
    "and client name from this invoice.",
    images=["invoice_page.png"],
)

parsed = agent.parse_output(response)
print(parsed.company)         # "TechCorp Solutions"
print(parsed.invoice_number)  # "INV-2026-001"
print(parsed.total)           # "$5,454.00"
print(parsed.client)          # "Acme Industries"
```

This pattern works for receipts, contracts, reports — any PDF you can render to an image.

---

## Populating a Database with Generated Data

Let the LLM generate realistic data and write it directly to your database:

```python
from sqlalchemy import Column, Integer, String, Float, Date, create_engine
from sqlalchemy.orm import declarative_base

from pygentix import Ollama, SqlAlchemyAgent

Base = declarative_base()

class Sale(Base):
    __tablename__ = "sales"
    id = Column(Integer, primary_key=True)
    product = Column(String)
    amount = Column(Float)
    date = Column(Date)

engine = create_engine("sqlite:///sales.db")
Base.metadata.create_all(engine)

class SalesAgent(Ollama, SqlAlchemyAgent):
    pass

agent = SalesAgent(engine=engine)
agent.writes(Sale)   # grants the model insert access

conv = agent.start_conversation()
conv.ask("Create 10 sales records with realistic product names, amounts between $10 and $500, and dates in 2026.")

# Verify the rows were inserted
from sqlalchemy.orm import Session
with Session(engine) as s:
    count = s.query(Sale).count()
    print(f"{count} sales created")  # 10 sales created
    for sale in s.query(Sale).all():
        print(f"  {sale.product}: ${sale.amount} on {sale.date}")
```

The agent introspects the ORM model's columns and generates `run_insert` calls
automatically — no manual SQL or fixture files needed.

---

## Structured Output

Use `OutputAgent` to guarantee responses follow a JSON schema:

```python
from pygentix import Ollama, OutputAgent

class MyAgent(Ollama, OutputAgent):
    pass

agent = MyAgent()

@agent.output
class Answer:
    answer: str
    confidence: float = 0.0
    sources: list = []

conv = agent.start_conversation()
response = conv.ask("What is the capital of France?")

parsed = agent.parse_output(response)
print(parsed.answer)       # "Paris"
print(parsed.confidence)   # 0.95
```

The schema can also be a raw dict — pass any valid JSON Schema to `agent.output({"type": "object", ...})`.

---

## SQLAlchemy Integration

`SqlAlchemyAgent` gives the LLM read/write access to your database through auto-generated tools:

```python
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base

from pygentix import Ollama, OutputAgent, SqlAlchemyAgent

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    price = Column(Integer)

engine = create_engine("sqlite:///shop.db")
Base.metadata.create_all(engine)

class ShopAgent(Ollama, SqlAlchemyAgent, OutputAgent):
    pass

agent = ShopAgent(engine=engine)
agent.reads(Product)                  # enables run_query
agent.writes(Product)                 # enables run_insert, run_update, run_delete

@agent.output
class Response:
    answer: str
    data: list = []

conv = agent.start_conversation()
conv.ask("Add a product called 'Widget' priced at 9.99")
response = conv.ask("List all products under $20")

parsed = agent.parse_output(response)
for item in parsed.data:
    print(item)
```

The agent automatically generates `run_query`, `run_insert`, `run_update`, and `run_delete` tools, handles type coercion (strings → ints, dates, etc.), and serialises results back to the model.

---

## Row-Level Security

Ensure users can only access their own data — even when the LLM generates the queries. pygentix supports three complementary layers.

### Direct Scope (automatic WHERE injection)

Map a column on the table to a key in the conversation's scope. All CRUD operations are automatically constrained:

```python
from pygentix import Ollama, SqlAlchemyAgent

class MyAgent(Ollama, SqlAlchemyAgent):
    pass

agent = MyAgent(engine=engine)
agent.reads(User, scope={"id": "current_user"})
agent.writes(User, scope={"id": "current_user"})

# Alice's session — she can only see and modify her own row
conv = agent.start_conversation(scope={"current_user": 5})
conv.ask("Update my name to Alice Smith")
# → UPDATE users SET name='Alice Smith' WHERE id=5
# Attempts to access other users' rows are silently filtered out
```

Inserts auto-set the scoped column; updates and deletes inject it into the filter. If the LLM tries to target a different user, it gets a `PermissionError`.

### Scope Chains (multi-level relationships)

When ownership is inferred through foreign keys (e.g. User → Sale → SaleItem), declare the chain:

```python
agent.reads(SaleItem, scope_chain=[
    ("sale_id", Sale),              # SaleItem.sale_id → JOIN Sale
    ("user_id", "current_user"),    # WHERE Sale.user_id = scope["current_user"]
])
agent.writes(SaleItem, scope_chain=[
    ("sale_id", Sale),
    ("user_id", "current_user"),
])

conv = agent.start_conversation(scope={"current_user": 5})
conv.ask("List my sale items")
# → SELECT ... FROM sale_items JOIN sales ON ... WHERE sales.user_id = 5
```

Chains can be arbitrarily deep — each tuple is `(fk_column, TargetModel)` except the last which is `(scope_column, scope_key)`:

```python
agent.reads(LineDetail, scope_chain=[
    ("item_id", SaleItem),          # JOIN SaleItem
    ("sale_id", Sale),              # JOIN Sale
    ("user_id", "current_user"),    # WHERE Sale.user_id = ...
])
```

### Policy Callbacks (general-purpose gate)

For authorization logic beyond SQL — API calls, custom rules, role checks — register a policy callback. It runs before **every** tool execution on any agent type:

```python
def my_policy(tool_name: str, arguments: dict, scope: dict) -> bool:
    """Return False to block the tool call."""
    if tool_name == "run_delete" and scope.get("role") != "admin":
        return False  # only admins can delete
    return True

conv = agent.start_conversation(
    scope={"current_user": 5, "role": "viewer"},
    policy=my_policy,
)
conv.ask("Delete all records")
# → LLM receives "Permission denied: run_delete blocked by policy"
```

### Combining Scope and Policy

When both are defined, both run. The policy gate executes first — if it denies, the tool never reaches the database. If it allows, the scope filters are applied as usual:

```python
conv = agent.start_conversation(
    scope={"current_user": 5, "role": "editor"},
    policy=my_policy,       # checked first
)
# 1. Policy: is this tool allowed for this role? ✓
# 2. Scope:  constrain query to current_user's data ✓
```

### No scope = unrestricted (backward compatible)

If you don't pass `scope` or `policy`, everything works exactly as before — no filters, no restrictions.

---

## Task Scheduling

`SchedulerAgent` lets the LLM schedule tool calls and conversations for future execution — using natural language like "send this email tomorrow" or "check sales every Monday at 9am".

```
pip install pygentix[scheduler]   # adds croniter for cron expressions
```

### Basic setup

```python
from pygentix import Ollama, SchedulerAgent

class MyAgent(Ollama, SchedulerAgent):
    pass

agent = MyAgent(schedule_file="tasks.json", poll_interval=10)

@agent.uses
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

agent.start_scheduler()

conv = agent.start_conversation()
conv.ask("Send an email to alice@co.com saying hello tomorrow at 9am")
# LLM calls schedule_task("send_email", {"to": "alice@co.com", ...}, run_at="2026-04-04T09:00:00")
```

The scheduler auto-registers four LLM tools: `schedule_task`, `schedule_conversation`, `list_scheduled_tasks`, and `cancel_scheduled_task`. No manual tool wiring needed.

### One-shot vs. recurring

```python
# One-shot — direct tool call at a specific time
conv.ask("Send the report email at 5pm today")
# LLM calls: schedule_task("send_email", {...}, run_at="2026-04-03T17:00:00")

# Recurring — cron expression
conv.ask("Every Monday at 9am, send a summary email to the team")
# LLM calls: schedule_task("send_email", {...}, cron="0 9 * * 1")
```

### Conversation replay (deferred reasoning)

When the LLM doesn't have all the data yet, it can schedule a future conversation instead of a direct call. At execution time, a fresh conversation runs so the LLM can reason about current data:

```python
conv.ask("Every Friday at 6pm, check the latest sales numbers and email a report to bob@co.com")
# LLM calls: schedule_conversation("check latest sales and email report to bob@co.com", cron="0 18 * * 5")
```

### Manual tick

For cron-based setups or testing, call `tick()` directly:

```python
results = agent.tick()  # executes all due tasks right now
```

### Task persistence

All scheduled tasks are persisted to a JSON file (default: `scheduled_tasks.json`). The file is human-readable and can be inspected or edited manually:

```json
[
  {
    "id": "a1b2c3",
    "type": "tool_call",
    "function_name": "send_email",
    "arguments": {"to": "alice@co.com", "subject": "Hello"},
    "run_at": "2026-04-04T09:00:00",
    "cron": null,
    "status": "pending"
  }
]
```

### Missed tasks

One-shot tasks whose `run_at` has passed while the process was down are marked as `"missed"` — they are **not** retried on startup. Recurring cron tasks simply wait for their next occurrence.

### Lifecycle

```python
agent.start_scheduler()   # start background polling thread
# ... application runs ...
agent.stop_scheduler()    # stop polling, join thread
```

---

## Mixing Backends

Every agent is a composable mixin — swap the backend class and everything else stays the same:

```python
from pygentix import Ollama, ChatGPT, Gemini, Copilot, SqlAlchemyAgent, OutputAgent

class LocalAgent(Ollama, SqlAlchemyAgent, OutputAgent):
    """Runs entirely on your machine via Ollama."""

class CloudAgent(ChatGPT, SqlAlchemyAgent, OutputAgent):
    """Uses OpenAI for inference."""

class GoogleAgent(Gemini, SqlAlchemyAgent, OutputAgent):
    """Uses Google Gemini for inference."""

class EnterpriseAgent(Copilot, SqlAlchemyAgent, OutputAgent):
    """Routes through your Azure OpenAI deployment."""
```

---

## Multi-turn Conversations

A `Conversation` maintains the full message history, so follow-up questions have context:

```python
from pygentix import Ollama, SqlAlchemyAgent

# ... define models, engine, etc.

agent = Ollama(engine=engine)
conv = agent.start_conversation()
conv.ask("Create a user named Alice with email alice@example.com")
conv.ask("Now create one for Bob at bob@example.com")
response = conv.ask("List all users")
```

---

## Streaming Responses

Stream tokens as they arrive instead of waiting for the full response:

```python
from pygentix import Ollama

agent = Ollama()
conv = agent.start_conversation()

for chunk in conv.ask_stream("Tell me a story about a robot"):
    print(chunk, end="", flush=True)
```

When tools are registered, the tool-call loop runs normally and the final answer is streamed. Every backend supports streaming natively (Ollama, OpenAI, Gemini, Azure).

---

## Async Support

Use `ask_async` in async frameworks like FastAPI, Starlette, or Django:

```python
import asyncio
from pygentix import Ollama

agent = Ollama()
conv = agent.start_conversation()

async def main():
    response = await conv.ask_async("What is the capital of France?")
    print(response.message.content)

asyncio.run(main())
```

By default, `chat_async` runs the sync method in a thread pool via `asyncio.to_thread`. Backends can override with native async clients for lower overhead.

---

## MockAgent for Testing

Unit-test your application without hitting a real LLM:

```python
from pygentix.testing import MockAgent

agent = MockAgent(responses=["Hello!", "Goodbye!"])
conv = agent.start_conversation()

r1 = conv.ask("Hi")     # → "Hello!"
r2 = conv.ask("Bye")    # → "Goodbye!"
```

MockAgent also supports tool-call simulation and usage metadata:

```python
agent = MockAgent(responses=[
    {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
    "It's sunny in Paris!",
])

@agent.uses
def get_weather(city: str) -> str:
    """Get weather."""
    return f"22°C in {city}"

conv = agent.start_conversation()
resp = conv.ask("Weather?")  # tool executes, then returns "It's sunny in Paris!"
```

---

## Event Hooks

Register callbacks to observe tool calls, results, and LLM responses in real-time:

```python
from pygentix import Ollama

agent = Ollama()

agent.on("tool_call", lambda name, args: print(f"→ Calling {name}({args})"))
agent.on("tool_result", lambda name, result: print(f"← {name} returned: {result}"))
agent.on("response", lambda resp: print(f"LLM: {resp.message.content[:80]}"))

@agent.uses
def search(query: str) -> str:
    """Search the web."""
    return f"Results for '{query}'..."

conv = agent.start_conversation()
conv.ask("Search for pygentix")
# → Calling search({'query': 'pygentix'})
# ← search returned: Results for 'pygentix'...
# LLM: Here are the search results for pygentix...
```

Hooks are ideal for logging, metrics, audit trails, and content filtering.

---

## Conversation Save & Load

Persist conversations to JSON and restore them later:

```python
from pygentix import Ollama
from pygentix.core import Conversation

agent = Ollama()
conv = agent.start_conversation()
conv.ask("My name is Alice")

# Save
json_str = conv.to_json()
# or: data = conv.to_dict()

# ... later, in another process ...
restored = Conversation.from_json(agent, json_str)
resp = restored.ask("What's my name?")
# → "Your name is Alice."
```

---

## Token Usage Tracking

Every `ChatResponse` includes token counts when the backend reports them:

```python
from pygentix import Ollama

agent = Ollama()
conv = agent.start_conversation()
response = conv.ask("Explain quantum computing in one sentence")

print(response.usage.prompt_tokens)      # e.g. 42
print(response.usage.completion_tokens)  # e.g. 28
print(response.usage.total_tokens)       # e.g. 70
```

All four backends (Ollama, ChatGPT, Gemini, Copilot) populate usage automatically.

---

## Retry with Exponential Backoff

Transient API errors (rate-limits, timeouts, 500s) are retried automatically with exponential backoff:

```python
from pygentix import ChatGPT

agent = ChatGPT(max_retries=5, retry_delay=2.0)
```

Retries apply to connection errors, timeouts, and HTTP status codes 429, 500, 502, 503, 504. Non-retriable errors (auth failures, validation) are raised immediately. The delay doubles after each attempt (2s → 4s → 8s → ...).

---

## Context Window Management

Prevent conversations from exceeding the model's context window by setting `max_history`:

```python
from pygentix import Ollama

agent = Ollama()
conv = agent.start_conversation(max_history=20)

# After many turns, only the system prompt + last 20 messages are kept
for i in range(100):
    conv.ask(f"Question {i}")

len(conv.messages)  # ≤ 22 (system + 20 + current response)
```

The system prompt is always preserved. Oldest messages are dropped first.

---

## Structured Logging

pygentix uses Python's standard `logging` module under the `"pygentix"` logger. Enable it to see every message, tool call, and response:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("pygentix").setLevel(logging.DEBUG)

# Now every conv.ask() logs:
# INFO  pygentix: User: What's the weather?
# DEBUG pygentix: Calling tool get_weather({'city': 'Paris'})
# DEBUG pygentix: Tool get_weather → Sunny, 22°C
# INFO  pygentix: Assistant: It's sunny and 22°C in Paris.
```

Set to `WARNING` in production to silence informational logs.

---

## API Reference

### Core

| Symbol | Description |
|---|---|
| `Agent` | Abstract base class — subclass to create a backend |
| `ChatResponse` | Normalized response every backend returns |
| `Conversation` | Multi-turn conversation with save/load, streaming, async |
| `Function` | Introspectable wrapper around a tool callable |
| `Usage` | Token usage statistics (prompt, completion, total) |

### Backends

| Symbol | Description |
|---|---|
| `Ollama` | Local inference via Ollama |
| `ChatGPT` | OpenAI Chat Completions |
| `Gemini` | Google Gemini (via `google-genai`) |
| `Copilot` | Azure OpenAI |

### Mixins & Utilities

| Symbol | Description |
|---|---|
| `OutputAgent` | JSON schema enforcement for responses |
| `SchedulerAgent` | Schedule tool calls and conversations for future execution |
| `SqlAlchemyAgent` | Database CRUD tools from ORM models |
| `MockAgent` | Fake backend for unit testing (`pygentix.testing`) |

---

## Development

```bash
git clone https://github.com/andreperussi/pygentix.git
cd pygentix
pip install -e ".[dev]"
pytest
```

---

## License

MIT
