"""
Test Configuration & Fixtures
==============================
Shared fixtures for all tests. Uses:
- In-memory SQLite for fast DB tests (no PostgreSQL needed)
- httpx.AsyncClient for async API testing
- Mocked Redis for cache/rate-limiter tests
"""

import asyncio
import os
import pickle
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Override settings before importing the app
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["APP_ENV"] = "development"
os.environ["MODEL_STORAGE_PATH"] = "./test_model_storage"
os.environ["CACHE_ENABLED"] = "false"
os.environ["LOG_FORMAT"] = "console"

from app.database.connection import Base, get_db_session
from app.api.dependencies import get_redis
from app.main import app


# ── Event Loop ───────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for all tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Database Fixtures ────────────────────────────────────────

@pytest_asyncio.fixture(scope="function")
async def db_engine():
    """Create a test database engine with SQLite."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///./test.db",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

    # Clean up test database file
    if os.path.exists("./test.db"):
        os.remove("./test.db")


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    session_factory = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


# ── Mock Redis ───────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.incr = AsyncMock(return_value=1)
    redis_mock.ttl = AsyncMock(return_value=60)
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.dbsize = AsyncMock(return_value=0)
    redis_mock.info = AsyncMock(return_value={"used_memory_human": "1M"})

    # Pipeline mock
    pipe_mock = AsyncMock()
    pipe_mock.incr = MagicMock(return_value=pipe_mock)
    pipe_mock.ttl = MagicMock(return_value=pipe_mock)
    pipe_mock.execute = AsyncMock(return_value=[1, 60])
    redis_mock.pipeline = MagicMock(return_value=pipe_mock)

    return redis_mock


# ── API Client ───────────────────────────────────────────────

@pytest_asyncio.fixture(scope="function")
async def client(db_session, mock_redis) -> AsyncGenerator[AsyncClient, None]:
    """
    Create an async test client with overridden dependencies.
    Replaces real DB and Redis with test fixtures.
    """

    async def override_db_session():
        yield db_session

    async def override_redis():
        return mock_redis

    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_redis] = override_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Model Fixtures ───────────────────────────────────────────

@pytest.fixture
def sample_sklearn_model(tmp_path) -> Path:
    """Create a sample sklearn model file for testing."""
    from sklearn.linear_model import LogisticRegression

    # Train a simple model
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)

    # Save to pickle
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path


@pytest.fixture
def model_storage_path(tmp_path) -> Path:
    """Create a temporary model storage directory."""
    storage = tmp_path / "model_storage"
    storage.mkdir()
    return storage
