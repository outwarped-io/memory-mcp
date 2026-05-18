"""Postgres-trigger correctness for migration 0017 popularity counters.

These tests exercise the live triggers directly via raw SQL — they don't go
through the high-level ``mem_write`` / ``rel_link`` APIs because we want
deterministic, narrow assertions on counter state.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from .conftest import SessionPairFactory

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Raw-SQL helpers
# ---------------------------------------------------------------------------

async def _mk_env(session) -> UUID:
    env_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO environments (id, name, kind, default_embedding_model_id) "
            "VALUES (:id, :name, 'test', 'test-embedding')"
        ),
        {"id": env_id, "name": f"refcount-{env_id}"},
    )
    return env_id


async def _mk_mem(session, env_id: UUID, *, kind: str = "fact", status: str = "active") -> UUID:
    mem_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO memories (id, env_id, kind, status, body) "
            "VALUES (:id, :env_id, :kind, :status, 'b')"
        ),
        {"id": mem_id, "env_id": env_id, "kind": kind, "status": status},
    )
    return mem_id


async def _mk_task(session, env_id: UUID) -> UUID:
    task_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO tasks (id, env_id, status, title, description) "
            "VALUES (:id, :env_id, 'pending', 't', 'task')"
        ),
        {"id": task_id, "env_id": env_id},
    )
    return task_id


async def _mk_gn_for_memory(session, env_id: UUID, memory_id: UUID) -> UUID:
    gn_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO graph_nodes (id, env_id, node_type, memory_id) "
            "VALUES (:id, :env_id, 'memory', :memory_id)"
        ),
        {"id": gn_id, "env_id": env_id, "memory_id": memory_id},
    )
    return gn_id


async def _mk_gn_for_task(session, env_id: UUID, task_id: UUID) -> UUID:
    gn_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO graph_nodes (id, env_id, node_type, task_id) "
            "VALUES (:id, :env_id, 'task', :task_id)"
        ),
        {"id": gn_id, "env_id": env_id, "task_id": task_id},
    )
    return gn_id


async def _mk_relation(
    session,
    env_id: UUID,
    src_gn: UUID,
    dst_gn: UUID,
    *,
    rel_type: str = "mentions",
) -> UUID:
    rel_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO relations (id, env_id, src_node_id, dst_node_id, type) "
            "VALUES (:id, :env_id, :src, :dst, :type)"
        ),
        {"id": rel_id, "env_id": env_id, "src": src_gn, "dst": dst_gn, "type": rel_type},
    )
    return rel_id


async def _read_counts(session, mem_id: UUID) -> dict[str, int]:
    row = (
        await session.execute(
            text(
                "SELECT reference_count_rel_link, reference_count_lineage, "
                "       reference_count_task, reference_count_playbook, reference_count "
                "FROM memories WHERE id = :id"
            ),
            {"id": mem_id},
        )
    ).one()
    return {
        "rel_link": row[0],
        "lineage": row[1],
        "task": row[2],
        "playbook": row[3],
        "total": row[4],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rel_link_insert_increments(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_for_memory(session, env_id, m_src)
        gn_dst = await _mk_gn_for_memory(session, env_id, m_dst)
        await _mk_relation(session, env_id, gn_src, gn_dst, rel_type="mentions")
        await session.commit()

        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 1
        assert counts["lineage"] == 0
        assert counts["task"] == 0
        assert counts["playbook"] == 0
        assert counts["total"] == 1


@pytest.mark.asyncio
async def test_rel_link_delete_decrements(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_for_memory(session, env_id, m_src)
        gn_dst = await _mk_gn_for_memory(session, env_id, m_dst)
        rel = await _mk_relation(session, env_id, gn_src, gn_dst)
        await session.commit()

        await session.execute(text("DELETE FROM relations WHERE id = :id"), {"id": rel})
        await session.commit()

        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 0
        assert counts["total"] == 0


@pytest.mark.asyncio
async def test_task_edge_uses_task_counter(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_dst = await _mk_mem(session, env_id)
        task = await _mk_task(session, env_id)
        gn_src = await _mk_gn_for_task(session, env_id, task)
        gn_dst = await _mk_gn_for_memory(session, env_id, m_dst)
        await _mk_relation(session, env_id, gn_src, gn_dst, rel_type="references")
        await session.commit()

        counts = await _read_counts(session, m_dst)
        assert counts["task"] == 1
        assert counts["rel_link"] == 0
        assert counts["total"] == 1


@pytest.mark.asyncio
async def test_related_to_popular_skipped(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """Phase 4's auto-wire predicate must not feed back into popularity counts."""
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_for_memory(session, env_id, m_src)
        gn_dst = await _mk_gn_for_memory(session, env_id, m_dst)
        await _mk_relation(session, env_id, gn_src, gn_dst, rel_type="related_to_popular")
        await session.commit()

        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 0
        assert counts["total"] == 0


@pytest.mark.asyncio
async def test_lineage_summarized_increments_parent(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        parent = await _mk_mem(session, env_id)
        child = await _mk_mem(session, env_id)
        await session.execute(
            text(
                "INSERT INTO memory_lineage (parent_memory_id, child_memory_id, relation) "
                "VALUES (:p, :c, 'summarized_from')"
            ),
            {"p": parent, "c": child},
        )
        await session.commit()

        counts = await _read_counts(session, parent)
        assert counts["lineage"] == 1
        # Child should be unaffected
        child_counts = await _read_counts(session, child)
        assert child_counts["lineage"] == 0


@pytest.mark.asyncio
async def test_lineage_supersedes_excluded(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """``supersedes`` is not load-bearing — should not bump lineage counter."""
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        parent = await _mk_mem(session, env_id)
        child = await _mk_mem(session, env_id)
        await session.execute(
            text(
                "INSERT INTO memory_lineage (parent_memory_id, child_memory_id, relation) "
                "VALUES (:p, :c, 'supersedes')"
            ),
            {"p": parent, "c": child},
        )
        await session.commit()

        counts = await _read_counts(session, parent)
        assert counts["lineage"] == 0


@pytest.mark.asyncio
async def test_status_flip_decrements_and_re_increments(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """active→retired decrements targets; retired→active re-increments."""
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_for_memory(session, env_id, m_src)
        gn_dst = await _mk_gn_for_memory(session, env_id, m_dst)
        # Two parallel rel_link edges → expect count of 2
        await _mk_relation(session, env_id, gn_src, gn_dst, rel_type="mentions")
        await _mk_relation(session, env_id, gn_src, gn_dst, rel_type="derives_from")
        await session.commit()

        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 2

        # Flip src to retired → dst counts must decrement by 2 (aggregate path)
        await session.execute(
            text("UPDATE memories SET status = 'retired' WHERE id = :id"),
            {"id": m_src},
        )
        await session.commit()
        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 0

        # Un-retire → re-increment by 2
        await session.execute(
            text("UPDATE memories SET status = 'active' WHERE id = :id"),
            {"id": m_src},
        )
        await session.commit()
        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 2


@pytest.mark.asyncio
async def test_status_flip_lineage_child_retire(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """Retiring a child memory should decrement its parent's lineage counter."""
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        parent = await _mk_mem(session, env_id)
        child = await _mk_mem(session, env_id)
        await session.execute(
            text(
                "INSERT INTO memory_lineage (parent_memory_id, child_memory_id, relation) "
                "VALUES (:p, :c, 'summarized_from')"
            ),
            {"p": parent, "c": child},
        )
        await session.commit()

        counts = await _read_counts(session, parent)
        assert counts["lineage"] == 1

        await session.execute(
            text("UPDATE memories SET status = 'retired' WHERE id = :id"),
            {"id": child},
        )
        await session.commit()
        counts = await _read_counts(session, parent)
        assert counts["lineage"] == 0

        await session.execute(
            text("UPDATE memories SET status = 'active' WHERE id = :id"),
            {"id": child},
        )
        await session.commit()
        counts = await _read_counts(session, parent)
        assert counts["lineage"] == 1


@pytest.mark.asyncio
async def test_status_flip_ignores_other_status_changes(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """active→stale shouldn't decrement targets — only retired/superseded does."""
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_for_memory(session, env_id, m_src)
        gn_dst = await _mk_gn_for_memory(session, env_id, m_dst)
        await _mk_relation(session, env_id, gn_src, gn_dst)
        await session.commit()

        await session.execute(
            text("UPDATE memories SET status = 'stale' WHERE id = :id"),
            {"id": m_src},
        )
        await session.commit()

        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 1


@pytest.mark.asyncio
async def test_computed_column_sums(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """``reference_count`` is GENERATED ALWAYS AS STORED — must equal the sum."""
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_dst = await _mk_mem(session, env_id)
        m_src = await _mk_mem(session, env_id)
        task = await _mk_task(session, env_id)

        gn_src_mem = await _mk_gn_for_memory(session, env_id, m_src)
        gn_src_task = await _mk_gn_for_task(session, env_id, task)
        gn_dst = await _mk_gn_for_memory(session, env_id, m_dst)

        await _mk_relation(session, env_id, gn_src_mem, gn_dst, rel_type="mentions")
        await _mk_relation(session, env_id, gn_src_task, gn_dst, rel_type="references")

        await session.execute(
            text(
                "INSERT INTO memory_lineage (parent_memory_id, child_memory_id, relation) "
                "VALUES (:p, :c, 'promoted_from')"
            ),
            {"p": m_dst, "c": m_src},
        )

        # Manually bump the playbook counter (no trigger writes it; recount pass owns it)
        await session.execute(
            text("UPDATE memories SET reference_count_playbook = 3 WHERE id = :id"),
            {"id": m_dst},
        )
        await session.commit()

        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 1
        assert counts["task"] == 1
        assert counts["lineage"] == 1
        assert counts["playbook"] == 3
        assert counts["total"] == 6  # GENERATED column


@pytest.mark.asyncio
async def test_counter_only_update_does_not_retrigger(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """Direct UPDATEs to reference_count_* must not re-fire the status-flip trigger.

    Guards against a regression where the trigger condition becomes too broad
    (e.g., ``AFTER UPDATE`` without ``OF status`` or ``DISTINCT FROM`` guard).
    """
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_for_memory(session, env_id, m_src)
        gn_dst = await _mk_gn_for_memory(session, env_id, m_dst)
        await _mk_relation(session, env_id, gn_src, gn_dst)
        await session.commit()

        # Direct counter bump should not trigger anything else
        await session.execute(
            text("UPDATE memories SET reference_count_playbook = 5 WHERE id = :id"),
            {"id": m_dst},
        )
        await session.commit()

        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 1
        assert counts["playbook"] == 5
        assert counts["total"] == 6
