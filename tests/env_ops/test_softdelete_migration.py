from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from memory_mcp_schemas.envs import EnvResponse

from memory_mcp.db.models import Environment
from memory_mcp.envs import list_envs


def test_environment_has_status_and_deleted_at_columns() -> None:
    assert {"status", "deleted_at"} <= set(Environment.__table__.columns.keys())


def test_env_response_carries_new_fields() -> None:
    now = dt.datetime.now(dt.UTC)
    base = {
        "id": uuid4(),
        "name": "work",
        "kind": None,
        "retention_policy": {},
        "default_embedding_model_id": "openai/text-embedding-3-small",
        "created_at": now,
    }

    active = EnvResponse(**base)
    assert active.status == "active"
    assert active.deleted_at is None

    deleted = EnvResponse(**base, status="deleted", deleted_at=now)
    round_tripped = EnvResponse.model_validate_json(deleted.model_dump_json())
    assert round_tripped.status == "deleted"
    assert round_tripped.deleted_at == now


@pytest.mark.asyncio
async def test_list_envs_excludes_deleted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    active = Environment(
        id=uuid4(),
        name="active-env",
        kind=None,
        retention_policy={},
        default_embedding_model_id="openai/text-embedding-3-small",
        created_at=dt.datetime.now(dt.UTC),
        status="active",
    )
    deleted = Environment(
        id=uuid4(),
        name="deleted-env",
        kind=None,
        retention_policy={},
        default_embedding_model_id="openai/text-embedding-3-small",
        created_at=dt.datetime.now(dt.UTC),
        status="deleted",
        deleted_at=dt.datetime.now(dt.UTC),
    )

    class _ScalarResult:
        def __init__(self, rows: list[Environment]) -> None:
            self._rows = rows

        def all(self) -> list[Environment]:
            return self._rows

    class _ExecuteResult:
        def __init__(self, rows: list[Environment]) -> None:
            self._rows = rows

        def scalars(self) -> _ScalarResult:
            return _ScalarResult(self._rows)

    class _Session:
        async def execute(self, stmt: object) -> _ExecuteResult:
            sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))  # type: ignore[attr-defined]
            rows = [active, deleted]
            if "environments.status = 'active'" in sql:
                rows = [row for row in rows if row.status == "active"]
            return _ExecuteResult(rows)

    @asynccontextmanager
    async def fake_session_scope() -> AsyncIterator[_Session]:
        yield _Session()

    monkeypatch.setattr("memory_mcp.envs.session_scope", fake_session_scope)

    assert [row.name for row in await list_envs()] == ["active-env"]
    assert {row.name for row in await list_envs(include_deleted=True)} == {
        "active-env",
        "deleted-env",
    }
