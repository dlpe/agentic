"""Integration test that exercises the full agent stack against a live Ollama instance.

Run with:  pytest tests/test_integration.py -v -s
Requires:  A running Ollama server with the qwen2.5:7b model available.

Each step asserts against the actual database state (deterministic) and
checks the model's structured response for expected keywords/data.
"""

import pytest
from sqlalchemy import Column, ForeignKey, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship
from sqlalchemy.types import Date, Float, Integer, String

from pygentix import Ollama, OutputAgent, SqlAlchemyAgent

Base = declarative_base()


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


# -- helpers ---------------------------------------------------------------


def _is_ollama_available() -> bool:
    try:
        from ollama import list as ollama_list
        ollama_list()
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _is_ollama_available(),
    reason="Ollama server not available",
)


def _parse(agent, resp):
    """Parse response; return (answer_text, data_list) regardless of format."""
    parsed = agent.parse_output(resp)
    if isinstance(parsed, str):
        return parsed.lower(), []
    return parsed.answer.lower(), parsed.data or []


# -- fixtures --------------------------------------------------------------


@pytest.fixture
def setup():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    agent = IntegrationAgent(engine=engine)

    @agent.output
    class AgentResponse:
        answer: str
        data: list = []

    for model in (User, Customer, Sale):
        agent.reads(model)
        agent.writes(model)

    @agent.uses
    def get_temperature():
        return "20 degrees Celsius"

    @agent.uses
    def get_conditions():
        return "sunny"

    conv = agent.start_conversation()
    return agent, engine, conv


# -- tests -----------------------------------------------------------------


@requires_ollama
class TestIntegration:

    def test_tool_call_weather(self, setup):
        """The agent should call get_temperature and get_conditions, then
        mention both 'sunny' and '20' in its answer."""
        agent, engine, conv = setup

        answer, data = _parse(agent, conv.ask("What is the weather and temperature?"))

        assert "sunny" in answer, f"Expected 'sunny' in answer: {answer}"
        assert "20" in answer, f"Expected '20' in answer: {answer}"

    def test_create_customer(self, setup):
        """Inserting a customer should produce exactly one row in the DB."""
        agent, engine, conv = setup

        answer, data = _parse(agent, conv.ask(
            "Create a new customer named 'Acme Corp' with email 'acme@example.com'"
        ))

        with Session(engine) as s:
            customers = s.query(Customer).all()
        assert len(customers) == 1
        assert customers[0].name == "Acme Corp"
        assert customers[0].email == "acme@example.com"

    def test_create_user(self, setup):
        """Inserting a user should produce exactly one row in the DB."""
        agent, engine, conv = setup

        answer, data = _parse(agent, conv.ask(
            "Create a new user named 'John Doe' with email 'john.doe@example.com'"
        ))

        with Session(engine) as s:
            users = s.query(User).all()
        assert len(users) == 1
        assert users[0].name == "John Doe"
        assert users[0].email == "john.doe@example.com"

    def test_create_sale(self, setup):
        """Creating a sale with explicit fields should persist correctly."""
        agent, engine, conv = setup

        conv.ask("Create a new customer named 'Acme Corp' with email 'acme@example.com'")
        conv.ask("Create a new user named 'John Doe' with email 'john.doe@example.com'")
        answer, data = _parse(agent, conv.ask(
            "Create a sale: date '2026-01-01', amount '150.00', customer_id '1', user_id '1'"
        ))

        with Session(engine) as s:
            sales = s.query(Sale).all()
        assert len(sales) == 1
        assert sales[0].amount == pytest.approx(150.0)
        assert sales[0].customer_id == 1
        assert sales[0].user_id == 1

    def test_query_all_sales(self, setup):
        """Querying all sales should return every row present in the DB."""
        agent, engine, conv = setup

        conv.ask("Create a new customer named 'Acme Corp' with email 'acme@example.com'")
        conv.ask("Create a new user named 'John Doe' with email 'john.doe@example.com'")
        conv.ask("Create a sale: date '2026-01-01', amount '150.00', customer_id '1', user_id '1'")

        answer, data = _parse(agent, conv.ask("Query all sales"))

        with Session(engine) as s:
            db_count = s.query(Sale).count()

        assert len(data) >= 1, "Expected at least one sale in the response data"
        assert len(data) == db_count, (
            f"Response data ({len(data)} rows) doesn't match DB ({db_count} rows)"
        )

    def test_batch_create_sales(self, setup):
        """The LLM generates random data for each sale — we don't specify it."""
        agent, engine, conv = setup

        conv.ask("Create a new customer named 'Acme Corp' with email 'acme@example.com'")
        conv.ask("Create a new user named 'John Doe' with email 'john.doe@example.com'")

        with Session(engine) as s:
            before = s.query(Sale).count()

        for i in range(5):
            fresh = agent.start_conversation()
            fresh.ask(
                "Call run_insert to create a sale with a random date in 2026, "
                f"a random amount between {50 + i * 80} and {130 + i * 80}, "
                "customer_id 1, and user_id 1."
            )

        with Session(engine) as s:
            after = s.query(Sale).count()

        assert after - before >= 3, (
            f"Expected at least 3 new sales, got {after - before}"
        )

    def test_update_sale(self, setup):
        """Updating sale #1 to amount 200 should be reflected in the DB."""
        agent, engine, conv = setup

        conv.ask("Create a new customer named 'Acme Corp' with email 'acme@example.com'")
        conv.ask("Create a new user named 'John Doe' with email 'john.doe@example.com'")
        conv.ask("Create a sale: date '2026-01-01', amount '150.00', customer_id '1', user_id '1'")

        conv.ask("Update the sale with id '1' to amount '200.00'")

        with Session(engine) as s:
            sale = s.get(Sale, 1)
        assert sale is not None, "Sale #1 not found"
        assert sale.amount == pytest.approx(200.0), (
            f"Expected amount 200.0, got {sale.amount}"
        )

    def test_query_sales_above_threshold(self, setup):
        """After updating sale #1 to 200, querying >180 must include it."""
        agent, engine, conv = setup

        conv.ask("Create a new customer named 'Acme Corp' with email 'acme@example.com'")
        conv.ask("Create a new user named 'John Doe' with email 'john.doe@example.com'")
        conv.ask("Create a sale: date '2026-01-01', amount '150.00', customer_id '1', user_id '1'")
        conv.ask("Update the sale with id '1' to amount '200.00'")

        answer, data = _parse(agent, conv.ask(
            "Query all sales with amount greater than 180.00"
        ))

        with Session(engine) as s:
            db_rows = s.query(Sale).filter(Sale.amount > 180.0).all()

        assert len(db_rows) >= 1, "DB should have at least one sale > 180"

        sale_ids_in_db = {row.id for row in db_rows}
        assert 1 in sale_ids_in_db, "Sale #1 (amount 200) should be > 180 in DB"

        assert len(data) >= 1, "Response should contain at least one sale > 180"
        response_amounts = [
            row.get("amount", 0) if isinstance(row, dict) else 0
            for row in data
        ]
        assert any(
            amt > 180 for amt in response_amounts
        ), f"No amount > 180 in response data: {data}"

    def test_list_all_users(self, setup):
        """Listing users should return every user row in the DB."""
        agent, engine, conv = setup

        conv.ask("Create a new user named 'John Doe' with email 'john.doe@example.com'")

        answer, data = _parse(agent, conv.ask("List all users"))

        with Session(engine) as s:
            db_users = s.query(User).all()

        assert len(db_users) >= 1

        assert len(data) >= 1, "Response should contain at least one user"
        names_in_response = [
            row.get("name", "") if isinstance(row, dict) else ""
            for row in data
        ]
        assert any(
            "John Doe" in name for name in names_in_response
        ), f"Expected 'John Doe' in response data: {data}"
