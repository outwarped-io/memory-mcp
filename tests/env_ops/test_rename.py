from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.env_ops import EnvRenameRequest
from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memory_mcp.db.models import Agent, Environment
from memory_mcp.env_ops import rename as renamer
from memory_mcp.env_ops.rename import rename_env
from memory_mcp.errors import InvalidInputError, NotFoundError
from memory_mcp.identity import AgentContext
from tests.env_ops.test_roundtrip import _truncate, postgres_factory  # noqa: F401


@pytest.fixture
async def rename_db(
    postgres_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncSession, AgentContext]]:
    ctx = AgentContext(agent_id=uuid4(), agent_name="rename-agent")

    @asynccontextmanager
    async def routed_session_scope() -> AsyncIterator[AsyncSession]:
        async with postgres_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr(renamer, "session_scope", routed_session_scope)

    async with postgres_factory() as session:
        await _truncate(session)
        session.add(Agent(id=ctx.agent_id, name="rename-agent"))
        await session.commit()

    async with postgres_factory() as session:
        yield session, ctx

    async with postgres_factory() as session:
        await _truncate(session)


@pytest.mark.asyncio
async def test_rename_changes_name(rename_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = rename_db
    env_id = await _create_env(session, "old")

    out = await rename_env(EnvRenameRequest(env_id=env_id, new_name="new"), ctx=ctx)

    assert out.name == "new"
    assert out.changed_fields == ["name"]
    session.expire_all()
    env = await session.get(Environment, env_id)
    assert env is not None
    assert env.name == "new"


@pytest.mark.asyncio
async def test_rename_name_conflict(rename_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = rename_db
    await _create_env(session, "a")
    env_b = await _create_env(session, "b")

    with pytest.raises(renamer.ConflictError) as exc:
        await rename_env(EnvRenameRequest(env_id=env_b, new_name="a"), ctx=ctx)
    assert exc.value.code == "ENV_NAME_TAKEN"


@pytest.mark.asyncio
async def test_rename_case_insensitive_conflict(rename_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = rename_db
    await _create_env(session, "Foo")
    other = await _create_env(session, "bar")

    with pytest.raises(renamer.ConflictError) as exc:
        await rename_env(EnvRenameRequest(env_id=other, new_name="FOO"), ctx=ctx)
    assert exc.value.code == "ENV_NAME_TAKEN"


@pytest.mark.asyncio
async def test_rename_changes_embedding_model_warns(rename_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = rename_db
    env_id = await _create_env(session, "model-env", model="model-a")

    out = await rename_env(
        EnvRenameRequest(env_id=env_id, new_default_embedding_model_id="model-b"),
        ctx=ctx,
    )

    assert out.default_embedding_model_id == "model-b"
    assert out.changed_fields == ["default_embedding_model_id"]
    assert out.warning is not None
    assert "does not re-embed existing memories" in out.warning


@pytest.mark.asyncio
async def test_rename_changes_retention_policy(rename_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = rename_db
    env_id = await _create_env(session, "retention-env", retention={"days": 30})

    out = await rename_env(
        EnvRenameRequest(env_id=env_id, new_retention_policy={"days": 7, "mode": "strict"}),
        ctx=ctx,
    )

    assert out.retention_policy == {"days": 7, "mode": "strict"}
    assert out.changed_fields == ["retention_policy"]


@pytest.mark.asyncio
async def test_rename_multiple_fields_at_once(rename_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = rename_db
    env_id = await _create_env(session, "multi", retention={"days": 30})

    out = await rename_env(
        EnvRenameRequest(env_id=env_id, new_name="multi-new", new_retention_policy={"days": 14}),
        ctx=ctx,
    )

    assert out.name == "multi-new"
    assert out.retention_policy == {"days": 14}
    assert out.changed_fields == ["name", "retention_policy"]


@pytest.mark.asyncio
async def test_rename_no_fields_set(rename_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = rename_db
    env_id = await _create_env(session, "noop")

    with pytest.raises(InvalidInputError) as exc:
        await rename_env(EnvRenameRequest(env_id=env_id), ctx=ctx)
    assert exc.value.code == "NOTHING_TO_RENAME"


@pytest.mark.asyncio
async def test_rename_rejects_deleted_env(rename_db: tuple[AsyncSession, AgentContext]) -> None:
    session, ctx = rename_db
    env_id = await _create_env(session, "deleted")
    await session.execute(
        update(Environment).where(Environment.id == env_id).values(status="deleted", deleted_at=func.now())
    )
    await session.commit()

    with pytest.raises(NotFoundError) as exc:
        await rename_env(EnvRenameRequest(env_id=env_id, new_name="after-delete"), ctx=ctx)
    assert exc.value.code == "ENV_DELETED"


@pytest.mark.asyncio
async def test_rename_with_same_value_excluded_from_changed_fields(
    rename_db: tuple[AsyncSession, AgentContext],
) -> None:
    session, ctx = rename_db
    env_id = await _create_env(session, "same")

    out = await rename_env(EnvRenameRequest(env_id=env_id, new_name="same"), ctx=ctx)

    assert "name" not in out.changed_fields
    assert out.changed_fields == []


@pytest.mark.asyncio
async def test_rename_emits_outbox_event(
    rename_db: tuple[AsyncSession, AgentContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, ctx = rename_db
    env_id = await _create_env(session, "event-old")
    calls: list[dict[str, object]] = []

    async def fake_enqueue_event(session, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)

    monkeypatch.setattr(renamer, "enqueue_event", fake_enqueue_event)

    await rename_env(EnvRenameRequest(env_id=env_id, new_name="event-new"), ctx=ctx)

    assert len(calls) == 1
    assert calls[0]["payload"]["event"] == "EnvRenamed"
    assert calls[0]["payload"]["old_name"] == "event-old"
    assert calls[0]["payload"]["new_name"] == "event-new"
    assert calls[0]["payload"]["changed_fields"] == ["name"]


async def _create_env(
    session: AsyncSession,
    name: str,
    *,
    retention: dict[str, object] | None = None,
    model: str = "test-model",
) -> UUID:
    env_id = uuid4()
    session.add(
        Environment(
            id=env_id,
            name=name,
            retention_policy=retention or {},
            default_embedding_model_id=model,
        )
    )
    await session.commit()
    return env_id
