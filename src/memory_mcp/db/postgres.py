"""Postgres connection / session factory.

The application acquires sessions through ``session_scope`` (an async context
manager) so that every unit of work is bracketed by a transaction. The
projection-worker uses raw connections directly for ``FOR UPDATE SKIP LOCKED``
patterns — see ``projection_worker.runner``.

Lifespan: ``init_engine(settings)`` is called once on app startup; teardown
disposes the engine.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memory_mcp.config import Settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> AsyncEngine:
    """Initialize the global engine + session factory.

    Idempotent — a second call with the same URL returns the existing engine.
    Calling with a different URL raises ``RuntimeError`` (use ``dispose_engine``
    first if you really need to swap).

    URL comparison renders the existing engine's URL with ``hide_password=False``
    so it compares cleanly to ``settings.postgres_url`` (SQLAlchemy's default
    ``str(URL)`` masks the password as ``***``).
    """
    global _engine, _session_factory

    if _engine is not None:
        existing_url = _engine.url.render_as_string(hide_password=False)
        if existing_url != settings.postgres_url:
            raise RuntimeError(
                "init_engine called twice with different URLs; "
                "call dispose_engine() first"
            )
        return _engine

    _engine = create_async_engine(
        settings.postgres_url,
        pool_size=settings.postgres_pool_size,
        max_overflow=settings.postgres_max_overflow,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "statement_timeout": str(settings.postgres_statement_timeout_ms),
                "application_name": "memory-mcp",
            },
        },
    )
    _session_factory = async_sessionmaker(
        _engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    return _engine


async def dispose_engine() -> None:
    """Tear down the engine; safe to call when no engine exists.

    Always clears the globals, even if ``engine.dispose()`` raises — leaking
    a half-disposed engine pinned in module state would deadlock subsequent
    ``init_engine`` calls (URL-mismatch error path).
    """
    import contextlib

    global _engine, _session_factory
    engine = _engine
    _engine = None
    _session_factory = None
    if engine is not None:
        with contextlib.suppress(Exception):
            await engine.dispose()


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Postgres engine not initialized; call init_engine() first")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Session factory not initialized; call init_engine() first")
    return _session_factory


@contextlib.asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Async context manager yielding a session bound to a transaction.

    Commits on clean exit, rolls back on exception. Caller must not call
    ``commit()`` / ``rollback()`` themselves.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
