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

## API Reference

### Core

| Symbol | Description |
|---|---|
| `Agent` | Abstract base class — subclass to create a backend |
| `ChatResponse` | Normalized response every backend returns |
| `Conversation` | Multi-turn conversation manager |
| `Function` | Introspectable wrapper around a tool callable |

### Backends

| Symbol | Description |
|---|---|
| `Ollama` | Local inference via Ollama |
| `ChatGPT` | OpenAI Chat Completions |
| `Gemini` | Google Gemini (via `google-genai`) |
| `Copilot` | Azure OpenAI |

### Mixins

| Symbol | Description |
|---|---|
| `OutputAgent` | JSON schema enforcement for responses |
| `SqlAlchemyAgent` | Database CRUD tools from ORM models |

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
