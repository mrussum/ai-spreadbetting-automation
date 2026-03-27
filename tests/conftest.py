"""Shared pytest fixtures for the trading system test suite."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base


@pytest.fixture(scope="function")
def db_session():
    """Provide an in-memory SQLite session for a single test.

    Creates all tables fresh for each test function, then tears down
    completely so tests are fully isolated.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def in_memory_engine():
    """Provide a bare in-memory SQLAlchemy engine with all tables created.

    Useful for tests that need to patch the module-level engine directly.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()
