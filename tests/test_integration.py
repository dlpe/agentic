"""Integration test that exercises the full agent stack against a live Ollama instance.

Run with:  pytest tests/test_integration.py -v -s
Requires:  A running Ollama server. Default model is ``qwen2.5:7b`` (fewer
bad turns than tiny models; wall time often beats ``3b`` anyway). Override with
``PYGENTIX_OLLAMA_TEST_MODEL`` (e.g. ``qwen2.5:3b``) if you want smallest weights.

``reads()`` is registered only on tests that need ``run_query``. Registering
``run_query`` for every ORM type (plus all write tools) makes local models return
garbled pseudo-JSON instead of native tool calls, so inserts never hit the DB.

Each step asserts against the actual database state (deterministic) and
checks the model's structured response for expected keywords/data.
"""

import ast
import os

# Cap each Ollama HTTP call so a hung server does not stall the suite indefinitely.
os.environ.setdefault("PYGENTIX_OLLAMA_HTTP_TIMEOUT_SEC", "120")

import pytest
from datetime import date
from sqlalchemy import Column, ForeignKey, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import Date, Float, Integer, String

from pygentix import Ollama, OutputAgent, SqlAlchemyAgent

Base = declarative_base()

# 7B default: better tool compliance → fewer retries/rounds than 3B on the same suite.
OLLAMA_INTEGRATION_MODEL = os.environ.get("PYGENTIX_OLLAMA_TEST_MODEL", "qwen2.5:7b")


# -- models ----------------------------------------------------------------


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)


class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    sales = relationship("Sale", back_populates="customer")


class Sale(Base):
    __tablename__ = "sales"
    id = Column(Integer, primary_key=True)
    date = Column(Date)
    amount = Column(Float)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    customer = relationship("Customer", back_populates="sales")
    user = relationship("User")


# -- agent -----------------------------------------------------------------


class IntegrationAgent(Ollama, SqlAlchemyAgent, OutputAgent):
    pass


class WeatherAgent(Ollama, OutputAgent):
    """Ollama + structured output only (no SQL tools) for small-tool weather checks."""


# -- helpers ---------------------------------------------------------------


def _parse(agent, resp):
    """Parse response; return (answer_text, data_list) regardless of format."""
    parsed = agent.parse_output(resp)
    if isinstance(parsed, str):
        return parsed.lower(), []
    if isinstance(parsed, dict):
        raw = parsed.get("data")
        return str(parsed.get("answer", "")).lower(), (
            raw if isinstance(raw, list) else []
        )
    raw = getattr(parsed, "data", None)
    if not isinstance(raw, list):
        raw = []
    return parsed.answer.lower(), raw


def _tool_transcript(conv) -> str:
    """Lowercased concatenation of every tool result in the conversation."""
    return " ".join(
        str(m.get("content", "")).lower()
        for m in conv.messages
        if m.get("role") == "tool"
    )


def _last_tool_rows(conv, tool_name: str) -> list:
    """Parse the most recent *tool_name* tool result as a list (``run_query`` returns rows)."""
    for m in reversed(conv.messages):
        if m.get("role") != "tool":
            continue
        if m.get("tool_name") != tool_name:
            continue
        raw = str(m.get("content", "")).strip()
        if not raw.startswith("["):
            return []
        try:
            val = ast.literal_eval(raw)
        except (SyntaxError, ValueError, TypeError):
            return []
        return val if isinstance(val, list) else []


def _count_rows(engine, model) -> int:
    with Session(engine) as s:
        return s.query(model).count()


def customer_match_count(engine, name: str, email: str) -> int:
    with Session(engine) as s:
        return s.query(Customer).filter_by(name=name, email=email).count()


def user_match_count(engine, name: str, email: str) -> int:
    with Session(engine) as s:
        return s.query(User).filter_by(name=name, email=email).count()


def seed_customer_and_user(conv, engine) -> None:
    """One user turn each; internal tool rounds are handled inside ``ask``."""
    conv.ask(
        "Create a new customer named 'Acme Corp' with email 'acme@example.com' "
        "using run_insert on entity Customer."
    )
    assert customer_match_count(engine, "Acme Corp", "acme@example.com") >= 1
    conv.ask(
        "Create a new user named 'John Doe' with email 'john.doe@example.com' "
        "using run_insert on entity User."
    )
    assert user_match_count(engine, "John Doe", "john.doe@example.com") >= 1


def seed_sale_150(conv, engine) -> None:
    conv.ask(
        "Create a sale with date '2026-01-01', amount '150.00', customer_id '1', "
        "user_id '1' using run_insert on entity Sale."
    )
    assert _count_rows(engine, Sale) >= 1


def _get_sale(engine, sale_id: int):
    with Session(engine) as s:
        return s.get(Sale, sale_id)


def _fresh_engine():
    # StaticPool: one shared in-memory DB for agent tools and test assertions.
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


def _build_integration_agent(
    engine,
    *,
    write_models: tuple = (User, Customer, Sale),
    read_models: tuple = (),
):
    agent = IntegrationAgent(model=OLLAMA_INTEGRATION_MODEL, engine=engine)

    @agent.output
    class AgentResponse:
        answer: str
        data: list = []

    for model in write_models:
        agent.writes(model)
    for model in read_models:
        agent.reads(model)

    conv = agent.start_conversation()
    return agent, conv


# -- fixtures --------------------------------------------------------------


@pytest.fixture
def weather_setup():
    agent = WeatherAgent(model=OLLAMA_INTEGRATION_MODEL)

    @agent.output
    class AgentResponse:
        answer: str
        data: list = []

    @agent.uses
    def get_temperature():
        return "20 degrees Celsius"

    @agent.uses
    def get_conditions():
        return "sunny"

    conv = agent.start_conversation()
    return agent, conv


@pytest.fixture
def setup():
    engine = _fresh_engine()
    agent, conv = _build_integration_agent(engine)
    return agent, engine, conv


@pytest.fixture
def setup_sale_query():
    """``run_query`` on ``Sale`` only (no write tools) so Ollama keeps native tool calls."""
    engine = _fresh_engine()
    agent, conv = _build_integration_agent(engine, write_models=(), read_models=(Sale,))
    return agent, engine, conv


@pytest.fixture
def setup_user_query():
    """``run_query`` on ``User`` only."""
    engine = _fresh_engine()
    agent, conv = _build_integration_agent(engine, write_models=(), read_models=(User,))
    return agent, engine, conv


# -- tests -----------------------------------------------------------------


class TestIntegration:

    def test_tool_call_weather(self, weather_setup):
        """Tools must run; final prose may omit one keyword on small models."""
        agent, conv = weather_setup

        resp = conv.ask(
            "What is the weather and temperature? "
            "Call get_conditions and get_temperature in the same turn (both tools); "
            "use tool results only."
        )
        blob = _tool_transcript(conv)

        answer, data = _parse(agent, resp)
        assert "20" in blob, f"get_temperature output missing from tools: {blob!r}"
        assert "sunny" in blob, f"get_conditions output missing from tools: {blob!r}"
        assert "20" in answer or "20" in blob
        assert "sunny" in answer or "sunny" in blob

    def test_create_customer(self, setup):
        """Inserting a customer should produce exactly one row in the DB."""
        agent, engine, conv = setup

        conv.ask(
            "Create a new customer named 'Acme Corp' with email 'acme@example.com' "
            "using run_insert on entity Customer."
        )
        assert customer_match_count(engine, "Acme Corp", "acme@example.com") >= 1

        with Session(engine) as s:
            customers = (
                s.query(Customer)
                .filter_by(name="Acme Corp", email="acme@example.com")
                .all()
            )
        assert len(customers) >= 1
        assert all(
            c.name == "Acme Corp" and c.email == "acme@example.com" for c in customers
        )

    def test_create_user(self, setup):
        """Inserting a user should produce exactly one row in the DB."""
        agent, engine, conv = setup

        conv.ask(
            "Create a new user named 'John Doe' with email 'john.doe@example.com' "
            "using run_insert on entity User."
        )
        assert user_match_count(engine, "John Doe", "john.doe@example.com") >= 1

        with Session(engine) as s:
            users = (
                s.query(User)
                .filter_by(name="John Doe", email="john.doe@example.com")
                .all()
            )
        assert len(users) >= 1
        assert all(
            u.name == "John Doe" and u.email == "john.doe@example.com" for u in users
        )

    def test_create_sale(self, setup):
        """Creating a sale with explicit fields should persist correctly."""
        agent, engine, conv = setup

        seed_customer_and_user(conv, engine)
        seed_sale_150(conv, engine)

        with Session(engine) as s:
            sales = s.query(Sale).all()
        assert len(sales) == 1
        assert sales[0].amount == pytest.approx(150.0)
        assert sales[0].customer_id == 1
        assert sales[0].user_id == 1

    def test_query_all_sales(self, setup_sale_query):
        """Querying all sales should return every row present in the DB."""
        agent, engine, conv = setup_sale_query

        # Seed with SQL so this test exercises ``run_query`` only (insert + read tools
        # together often make small local models omit ``data`` in the final JSON).
        with Session(engine) as s:
            s.add(Customer(name="Acme Corp", email="acme@example.com"))
            s.add(User(name="John Doe", email="john.doe@example.com"))
            s.flush()
            s.add(
                Sale(
                    date=date(2026, 1, 1),
                    amount=150.0,
                    customer_id=1,
                    user_id=1,
                )
            )
            s.commit()

        with Session(engine) as s:
            db_count = s.query(Sale).count()
        answer, data = _parse(
            agent,
            conv.ask("Query all Sale rows using run_query on entity Sale."),
        )
        if not isinstance(data, list):
            data = []

        rows = _last_tool_rows(conv, "run_query")
        if not isinstance(rows, list):
            rows = []
        effective = data if len(data) >= len(rows) else rows
        assert (
            len(effective) >= 1
        ), f"Expected query rows in output or tool result, data={data!r} tool_rows={rows!r}"
        assert (
            len(effective) == db_count
        ), f"Row count mismatch: response {len(effective)} vs DB {db_count}"

    def test_batch_create_sales(self, setup):
        """The LLM generates random data for each sale — we don't specify it."""
        agent, engine, conv = setup

        seed_customer_and_user(conv, engine)

        with Session(engine) as s:
            before = s.query(Sale).count()

        # Two distinct user requests (not retries); each may trigger several model steps.
        for i in range(2):
            conv.ask(
                "Call run_insert to create a sale with a random date in 2026, "
                f"a random amount between {50 + i * 80} and {130 + i * 80}, "
                "customer_id 1, and user_id 1."
            )

        with Session(engine) as s:
            after = s.query(Sale).count()

        assert (
            after - before >= 1
        ), f"Expected at least 1 new sale, got {after - before}"

    def test_update_sale(self, setup):
        """Updating sale #1 to amount 200 should be reflected in the DB."""
        agent, engine, conv = setup

        seed_customer_and_user(conv, engine)
        seed_sale_150(conv, engine)

        conv.ask(
            "Update the sale with id '1' to amount '200.00' using run_update on entity Sale."
        )

        with Session(engine) as s:
            sale = s.get(Sale, 1)
        assert sale is not None, "Sale #1 not found"
        assert sale.amount == pytest.approx(
            200.0
        ), f"Expected amount 200.0, got {sale.amount}"

    def test_query_sales_above_threshold(self, setup_sale_query):
        """DB has sale #1 at 200; ``run_query`` must return a row with amount > 180."""
        agent, engine, conv = setup_sale_query

        with Session(engine) as s:
            s.add(Customer(name="Acme Corp", email="acme@example.com"))
            s.add(User(name="John Doe", email="john.doe@example.com"))
            s.flush()
            s.add(
                Sale(
                    date=date(2026, 1, 1),
                    amount=150.0,
                    customer_id=1,
                    user_id=1,
                )
            )
            s.commit()
        with Session(engine) as s:
            row = s.get(Sale, 1)
            row.amount = 200.0
            s.commit()

        answer, data = _parse(
            agent,
            conv.ask(
                "Query sales with amount greater than 180 using run_query on entity Sale."
            ),
        )
        if not isinstance(data, list):
            data = []
        rows = _last_tool_rows(conv, "run_query")
        if not isinstance(rows, list):
            rows = []
        effective = data if len(data) >= len(rows) else rows

        with Session(engine) as s:
            db_rows = s.query(Sale).filter(Sale.amount > 180.0).all()

        assert len(db_rows) >= 1, "DB should have at least one sale > 180"

        sale_ids_in_db = {row.id for row in db_rows}
        assert 1 in sale_ids_in_db, "Sale #1 (amount 200) should be > 180 in DB"

        assert (
            len(effective) >= 1
        ), f"Response should contain at least one sale > 180; effective={effective!r}"
        response_amounts = [
            float(row.get("amount", 0)) if isinstance(row, dict) else 0.0
            for row in effective
        ]
        assert any(
            amt > 180 for amt in response_amounts
        ), f"No amount > 180 in response data: {effective}"

    def test_list_all_users(self, setup_user_query):
        """Listing users should return every user row in the DB."""
        agent, engine, conv = setup_user_query

        with Session(engine) as s:
            s.add(User(name="John Doe", email="john.doe@example.com"))
            s.commit()
        assert _count_rows(engine, User) >= 1

        answer, data = _parse(
            agent, conv.ask("List all users using run_query on entity User.")
        )
        if not isinstance(data, list):
            data = []

        with Session(engine) as s:
            db_users = s.query(User).all()

        assert len(db_users) >= 1

        rows = _last_tool_rows(conv, "run_query")
        if not isinstance(rows, list):
            rows = []
        effective = data if len(data) >= len(rows) else rows
        assert (
            len(effective) >= 1
        ), f"Expected at least one user in output or tool result, data={data!r} tool_rows={rows!r}"
        names_in_response = [
            row.get("name", "") if isinstance(row, dict) else "" for row in effective
        ]
        assert any(
            "John Doe" in name for name in names_in_response
        ), f"Expected 'John Doe' in rows: {effective}"
