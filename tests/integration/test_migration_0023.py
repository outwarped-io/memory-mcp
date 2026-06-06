"""Migration 0023 integration coverage: env_acls table + principal_id audit columns.

Verifies the substrate shape described in ``docs/adr/0001-auth-phase-2a.md`` §7:

1. ``env_acls`` table exists with PK ``(env_id, principal_id)`` and FK CASCADE on env.
2. ``env_acls.role`` CHECK constraint rejects values outside the closed set.
3. ``agents.principal_id`` partial unique index allows multiple NULLs (synthetic
   ``agent-<uuid>`` rows) but rejects duplicate non-NULL subjects.
4. ``memories.created_by_principal_id`` is nullable + writable.
5. ``relations.created_by_principal_id`` is nullable + writable.
6. ``memory_tombstones.created_by_principal_id`` is nullable + writable.

The conftest migrates the testcontainer DB to head, which now ends at
``0023_auth_phase2a``. Tests own their environment / agent rows and clean up
behind themselves (env_acls is not in the ``clean_db`` truncate list).
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from .conftest import SessionPairFactory

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Setup helpers — keep parallel-test-safe by giving each row a unique uuid.
# ---------------------------------------------------------------------------


async def _mk_env(session) -> UUID:
    env_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO environments (id, name, kind, default_embedding_model_id) "
            "VALUES (:id, :name, 'test', 'test-embedding')"
        ),
        {"id": env_id, "name": f"mig23-{env_id.hex[:8]}"},
    )
    return env_id


async def _mk_agent(session, *, principal_id: str | None = None) -> UUID:
    agent_id = uuid4()
    await session.execute(
        text("INSERT INTO agents (id, name, principal_id) VALUES (:id, :name, :p)"),
        {"id": agent_id, "name": f"mig23-{agent_id.hex[:8]}", "p": principal_id},
    )
    return agent_id


# ---------------------------------------------------------------------------
# 1. env_acls table shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_acls_pk_and_fk_cascade(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """PK is (env_id, principal_id); deleting the env cascades the ACL rows."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        await session.execute(
            text(
                "INSERT INTO env_acls (env_id, principal_id, role, granted_by) "
                "VALUES (:e, :p, 'admin', 'bootstrap')"
            ),
            {"e": env_id, "p": "user-alice@example.com"},
        )
        await session.commit()

        # PK conflict on duplicate (env_id, principal_id)
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await session.execute(
                    text(
                        "INSERT INTO env_acls (env_id, principal_id, role, granted_by) "
                        "VALUES (:e, :p, 'writer', 'someone-else')"
                    ),
                    {"e": env_id, "p": "user-alice@example.com"},
                )
        await session.rollback()

        # Different principal on same env is allowed.
        await session.execute(
            text(
                "INSERT INTO env_acls (env_id, principal_id, role, granted_by) "
                "VALUES (:e, :p, 'reader', 'user-alice@example.com')"
            ),
            {"e": env_id, "p": "user-bob@example.com"},
        )
        await session.commit()

        rows = (
            await session.execute(
                text("SELECT principal_id, role FROM env_acls WHERE env_id = :e ORDER BY principal_id"),
                {"e": env_id},
            )
        ).all()
        assert [(r[0], r[1]) for r in rows] == [
            ("user-alice@example.com", "admin"),
            ("user-bob@example.com", "reader"),
        ]

        # CASCADE delete
        await session.execute(text("DELETE FROM environments WHERE id = :e"), {"e": env_id})
        await session.commit()
        leftover = (
            await session.execute(text("SELECT count(*) FROM env_acls WHERE env_id = :e"), {"e": env_id})
        ).scalar_one()
        assert leftover == 0


# ---------------------------------------------------------------------------
# 2. env_acls.role CHECK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_acls_role_check_rejects_unknown(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """role must be one of admin/writer/reader."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        await session.commit()

        with pytest.raises(IntegrityError):
            async with session.begin():
                await session.execute(
                    text(
                        "INSERT INTO env_acls (env_id, principal_id, role, granted_by) "
                        "VALUES (:e, 'p', 'owner', 'bootstrap')"
                    ),
                    {"e": env_id},
                )

        # Cleanup
        async with session.begin():
            await session.execute(text("DELETE FROM environments WHERE id = :e"), {"e": env_id})


# ---------------------------------------------------------------------------
# 3. agents.principal_id partial unique index
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agents_principal_id_partial_unique(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """Multiple NULL principal_ids coexist; duplicate non-NULL is rejected."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        # Two synthetic rows, both NULL principal — must coexist.
        a1 = await _mk_agent(session, principal_id=None)
        a2 = await _mk_agent(session, principal_id=None)
        assert a1 != a2

        # First non-NULL principal succeeds.
        a3 = await _mk_agent(session, principal_id="user-shared@example.com")
        await session.commit()

        # Second non-NULL row with the *same* principal must be rejected.
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await _mk_agent(session, principal_id="user-shared@example.com")
        await session.rollback()

        # A different principal succeeds.
        a4 = await _mk_agent(session, principal_id="user-other@example.com")
        await session.commit()

        # Cleanup
        async with session.begin():
            await session.execute(
                text("DELETE FROM agents WHERE id = ANY(:ids)"),
                {"ids": [a1, a2, a3, a4]},
            )


# ---------------------------------------------------------------------------
# 4. memories.created_by_principal_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memories_created_by_principal_id_writable(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """The new audit column is nullable and round-trips a TEXT principal."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        legacy_mem_id = uuid4()
        attested_mem_id = uuid4()
        await session.execute(
            text(
                "INSERT INTO memories (id, env_id, kind, status, body) "
                "VALUES (:id, :env, 'fact', 'active', 'legacy')"
            ),
            {"id": legacy_mem_id, "env": env_id},
        )
        await session.execute(
            text(
                "INSERT INTO memories (id, env_id, kind, status, body, created_by_principal_id) "
                "VALUES (:id, :env, 'fact', 'active', 'attested', :p)"
            ),
            {"id": attested_mem_id, "env": env_id, "p": "user-eve@example.com"},
        )
        await session.commit()

        rows = (
            await session.execute(
                text("SELECT id, created_by_principal_id FROM memories WHERE env_id = :env ORDER BY body"),
                {"env": env_id},
            )
        ).all()
        result = {r[0]: r[1] for r in rows}
        assert result[attested_mem_id] == "user-eve@example.com"
        assert result[legacy_mem_id] is None


# ---------------------------------------------------------------------------
# 5. relations.created_by_principal_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relations_created_by_principal_id_writable(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """``relations`` gets the same nullable audit column."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        src_mem = uuid4()
        dst_mem = uuid4()
        src_node = uuid4()
        dst_node = uuid4()
        for mem_id, node_id in ((src_mem, src_node), (dst_mem, dst_node)):
            await session.execute(
                text(
                    "INSERT INTO memories (id, env_id, kind, status, body) "
                    "VALUES (:id, :env, 'fact', 'active', 'rel-test')"
                ),
                {"id": mem_id, "env": env_id},
            )
            await session.execute(
                text(
                    "INSERT INTO graph_nodes (id, env_id, node_type, memory_id) "
                    "VALUES (:id, :env, 'memory', :mid)"
                ),
                {"id": node_id, "env": env_id, "mid": mem_id},
            )
        rel_id = uuid4()
        await session.execute(
            text(
                "INSERT INTO relations (id, env_id, src_node_id, dst_node_id, type, created_by_principal_id) "
                "VALUES (:id, :env, :src, :dst, 'mentions', :p)"
            ),
            {
                "id": rel_id,
                "env": env_id,
                "src": src_node,
                "dst": dst_node,
                "p": "user-rel@example.com",
            },
        )
        await session.commit()

        principal = (
            await session.execute(
                text("SELECT created_by_principal_id FROM relations WHERE id = :id"),
                {"id": rel_id},
            )
        ).scalar_one()
        assert principal == "user-rel@example.com"


# ---------------------------------------------------------------------------
# 6. memory_tombstones.created_by_principal_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_tombstones_created_by_principal_id_writable(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """``memory_tombstones`` gets the same nullable audit column."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        tomb_id = uuid4()
        await session.execute(
            text(
                "INSERT INTO memory_tombstones (id, env_id, reason, created_by_principal_id) "
                "VALUES (:id, :env, 'leak recovery', :p)"
            ),
            {"id": tomb_id, "env": env_id, "p": "user-deleter@example.com"},
        )
        await session.commit()

        principal = (
            await session.execute(
                text("SELECT created_by_principal_id FROM memory_tombstones WHERE id = :id"),
                {"id": tomb_id},
            )
        ).scalar_one()
        assert principal == "user-deleter@example.com"
