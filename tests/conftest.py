"""Shared fixtures and test models for the agentic test suite."""

import pytest
from sqlalchemy import Column, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import Boolean, Date, DateTime, Float, Integer, String

from pygentix.sqlalchemy import SqlAlchemyAgent

Base = declarative_base()


# -- test models -----------------------------------------------------------


class Author(Base):
    __tablename__ = "authors"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    books = relationship("Book", back_populates="author")


class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    price = Column(Float)
    author_id = Column(Integer, ForeignKey("authors.id"))
    author = relationship("Author", back_populates="books")


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    event_date = Column(Date)
    created_at = Column(DateTime)
    active = Column(Boolean)
    score = Column(Float)


# -- stubs -----------------------------------------------------------------


class StubSqlAgent(SqlAlchemyAgent):
    """Minimal concrete agent for unit-testing SqlAlchemyAgent."""

    def chat(self, messages: list[dict], **kwargs):
        raise NotImplementedError


# -- fixtures --------------------------------------------------------------


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def agent(engine):
    a = StubSqlAgent(engine=engine)
    a.reads(Author)
    a.reads(Book)
    a.writes(Author)
    a.writes(Book)
    return a
