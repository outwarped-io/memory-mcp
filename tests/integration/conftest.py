"""Shared fixtures for real-Postgres integration tests."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

AsyncSessionFactory = async_sessionmaker[AsyncSession]
SessionPairFactory = Callable[[], tuple[AsyncSessionFactory, AsyncSessionFactory]]

REPO_ROOT = Path(__file__).resolve().parents[2]
_current_session_factory: ContextVar[AsyncSessionFactory | None] = ContextVar(
    "memory_mcp_integration_session_factory",
    default=None,
)


class Barrier:
    """Wait for N parties to arrive, then release all simultaneously."""

    def __init__(self, parties: int):
        self._parties = parties
        self._waiting = 0
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            self._waiting += 1
            if self._waiting >= self._parties:
                self._event.set()
        await self._event.wait()


@asynccontextmanager
async def routed_session_scope() -> AsyncIterator[AsyncSession]:
    """Session-scope replacement routed by a context-local session factory."""

    factory = _current_session_factory.get()
    if factory is None:
        raise RuntimeError("integration session factory was not set")
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def use_session_factory(factory: AsyncSessionFactory) -> Token[AsyncSessionFactory | None]:
    """Route production session_scope calls in this task to ``factory``."""

    return _current_session_factory.set(factory)


def reset_session_factory(token: Token[AsyncSessionFactory | None]) -> None:
    _current_session_factory.reset(token)


@pytest.fixture(scope="session")
def postgres_session_factories() -> Iterator[SessionPairFactory]:
    """Start one Postgres testcontainer, migrate to head, and yield pair factories."""

    container = PostgresContainer(
        "postgres:16-alpine",
        username="memory",
        password="memory",
        dbname="memory",
    )
    try:
        container.start()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres testcontainer unavailable; Docker is required for integration tests: {exc!r}")

    engines: list[AsyncEngine] = []
    try:
        sync_url = container.get_connection_url(driver="psycopg2")
        async_url = container.get_connection_url(driver="asyncpg")
        config = Config(str(REPO_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
        prior_url = os.environ.get("POSTGRES_URL")
        os.environ["POSTGRES_URL"] = sync_url
        try:
            command.upgrade(config, "head")
        finally:
            if prior_url is None:
                os.environ.pop("POSTGRES_URL", None)
            else:
                os.environ["POSTGRES_URL"] = prior_url

        def make_session_pair() -> tuple[AsyncSessionFactory, AsyncSessionFactory]:
            pair: list[AsyncSessionFactory] = []
            for _ in range(2):
                engine = create_async_engine(async_url, pool_pre_ping=True, pool_size=2, max_overflow=0)
                engines.append(engine)
                pair.append(async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession))
            return pair[0], pair[1]

        yield make_session_pair
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres integration setup failed before tests could run: {exc!r}")
    finally:
        for engine in engines:
            asyncio.run(engine.dispose())
        try:
            container.stop()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture
async def clean_db(postgres_session_factories: SessionPairFactory) -> AsyncIterator[None]:
    """Clean test-owned tables between integration tests."""

    factory, _ = postgres_session_factories()

    async def truncate() -> None:
        async with factory() as session:
            await session.execute(
                text("TRUNCATE tasks, memories, graph_nodes, relations, dream_proposals CASCADE")
            )
            await session.commit()

    await truncate()
    yield
    await truncate()
