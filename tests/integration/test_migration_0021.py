"""Migration 0021 integration coverage: lineage CHECK widening, decompose_operations table, popularity trigger whitelist update.

These tests run against a Postgres testcontainer that has been migrated to
``head`` — which after this migration ships is ``0021_decompose_operations``.
The assertions cover the three substrate concerns Migration 0021 introduces:

1. **CHECK widening.** ``memory_lineage.relation`` now admits ``'split_from'``
   and ``'derived_from'`` (in addition to the five pre-0021 values).
2. **decompose_operations table.** Mode CHECK, ``(env_id, dedupe_key)``
   uniqueness, ``source_id`` FK RESTRICT, ``child_ids`` ARRAY round-trip.
3. **Popularity whitelist update.** ``'split_from'`` rows must NOT bump
   ``reference_count_lineage``; ``'derived_from'`` rows MUST. Status-flip
   trigger respects the same whitelist.

A recount-parity smoke test ensures the dream pass agrees with the live
triggers on the new whitelist (no drift on split-only / derive-only envs).
"""

from __future__ import annotations

import datetime as dt
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from memory_mcp.config import Settings
from memory_mcp.identity import AgentContext

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Raw-SQL helpers (same shape as test_reference_counts.py to keep the
# integration suite uniform).
# ---------------------------------------------------------------------------


async def _mk_env(session) -> UUID:
    env_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO environments (id, name, kind, default_embedding_model_id) "
            "VALUES (:id, :name, 'test', 'test-embedding')"
        ),
        {"id": env_id, "name": f"mig21-{env_id}"},
    )
    return env_id


async def _mk_mem(
    session,
    env_id: UUID,
    *,
    kind: str = "fact",
    status: str = "active",
) -> UUID:
    mem_id = uuid4()
    await session.execute(
        text("INSERT INTO memories (id, env_id, kind, status, body) VALUES (:id, :env_id, :kind, :status, 'b')"),
        {"id": mem_id, "env_id": env_id, "kind": kind, "status": status},
    )
    return mem_id


async def _mk_agent(session) -> UUID:
    agent_id = uuid4()
    await session.execute(
        text("INSERT INTO agents (id, name) VALUES (:id, :name) ON CONFLICT DO NOTHING"),
        {"id": agent_id, "name": f"mig21-{agent_id.hex[:8]}"},
    )
    return agent_id


async def _mk_lineage(
    session,
    parent_id: UUID,
    child_id: UUID,
    relation: str,
) -> None:
    await session.execute(
        text("INSERT INTO memory_lineage (parent_memory_id, child_memory_id, relation) VALUES (:p, :c, :r)"),
        {"p": parent_id, "c": child_id, "r": relation},
    )


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
# 1. CHECK widening
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_widens_lineage_check(postgres_session_factories: SessionPairFactory, clean_db: None) -> None:
    """Both ``split_from`` and ``derived_from`` are accepted by the CHECK after 0021."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        parent = await _mk_mem(session, env_id)
        child_a = await _mk_mem(session, env_id)
        child_b = await _mk_mem(session, env_id)

        await _mk_lineage(session, parent, child_a, "split_from")
        await _mk_lineage(session, parent, child_b, "derived_from")
        await session.commit()

        rows = (
            await session.execute(
                text(
                    "SELECT child_memory_id, relation FROM memory_lineage WHERE parent_memory_id = :p ORDER BY relation"
                ),
                {"p": parent},
            )
        ).all()
        relations = {r[1] for r in rows}
        assert relations == {"derived_from", "split_from"}


# ---------------------------------------------------------------------------
# 2. Popularity whitelist: split_from must NOT bump
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_from_lineage_does_not_bump_reference_count_lineage(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """C1.5 redirect E.11 regression: ``split_from`` is no longer load-bearing.

    Inserting a ``split_from`` lineage row must leave the parent's
    ``reference_count_lineage`` at zero. The pre-0021 trigger function would
    have bumped it.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        parent = await _mk_mem(session, env_id)
        child = await _mk_mem(session, env_id)

        await _mk_lineage(session, parent, child, "split_from")
        await session.commit()

        counts = await _read_counts(session, parent)
        assert counts["lineage"] == 0
        assert counts["total"] == 0


# ---------------------------------------------------------------------------
# 3. Popularity whitelist: derived_from MUST bump
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derived_from_lineage_bumps_reference_count_lineage(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """``derived_from`` connects an active parent to its derive children
    and contributes to the parent's popularity signal."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        parent = await _mk_mem(session, env_id)
        child_a = await _mk_mem(session, env_id)
        child_b = await _mk_mem(session, env_id)

        await _mk_lineage(session, parent, child_a, "derived_from")
        await _mk_lineage(session, parent, child_b, "derived_from")
        await session.commit()

        counts = await _read_counts(session, parent)
        assert counts["lineage"] == 2

        # Removing one row decrements.
        await session.execute(
            text(
                "DELETE FROM memory_lineage "
                "WHERE parent_memory_id = :p AND child_memory_id = :c "
                "  AND relation = 'derived_from'"
            ),
            {"p": parent, "c": child_a},
        )
        await session.commit()
        counts = await _read_counts(session, parent)
        assert counts["lineage"] == 1


# ---------------------------------------------------------------------------
# 4. Status-flip trigger respects the new whitelist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_flip_skips_split_from_outgoing(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """Flipping a child's status must not adjust a parent whose lineage edge is ``split_from``.

    Two parallel parents are wired to one child — one via ``split_from``
    (not load-bearing after 0021), one via ``derived_from`` (load-bearing).
    Bumping the child from active → retired walks outgoing lineage edges
    and decrements only the ``derived_from`` parent. Un-retiring re-increments
    only that one.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        parent_split = await _mk_mem(session, env_id)
        parent_derive = await _mk_mem(session, env_id)
        child = await _mk_mem(session, env_id)

        await _mk_lineage(session, parent_split, child, "split_from")
        await _mk_lineage(session, parent_derive, child, "derived_from")
        await session.commit()

        # Initial state: split parent at 0 (whitelist excludes), derive parent at 1.
        assert (await _read_counts(session, parent_split))["lineage"] == 0
        assert (await _read_counts(session, parent_derive))["lineage"] == 1

        # active → retired on child: only derive parent decrements.
        await session.execute(
            text("UPDATE memories SET status = 'retired' WHERE id = :id"),
            {"id": child},
        )
        await session.commit()
        assert (await _read_counts(session, parent_split))["lineage"] == 0
        assert (await _read_counts(session, parent_derive))["lineage"] == 0

        # retired → active: derive parent re-increments.
        await session.execute(
            text("UPDATE memories SET status = 'active' WHERE id = :id"),
            {"id": child},
        )
        await session.commit()
        assert (await _read_counts(session, parent_split))["lineage"] == 0
        assert (await _read_counts(session, parent_derive))["lineage"] == 1


# ---------------------------------------------------------------------------
# 5-7. decompose_operations table contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_operations_mode_check_rejects_unknown(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """``mode`` CHECK admits only 'split' or 'derive'."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        source = await _mk_mem(session, env_id)
        agent_id = await _mk_agent(session)
        await session.commit()

    async with factory() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text(
                    "INSERT INTO decompose_operations "
                    "(env_id, source_id, mode, dedupe_key, request_fingerprint, "
                    " child_ids, created_by_agent_id) "
                    "VALUES (:env, :src, 'upsert', 'k1', 'fp1', ARRAY[]::uuid[], :agent)"
                ),
                {"env": env_id, "src": source, "agent": agent_id},
            )
            await session.commit()


@pytest.mark.asyncio
async def test_decompose_operations_dedupe_unique(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """``(env_id, dedupe_key)`` collisions raise ``UniqueViolation``.

    Two attempts to write the same ``(env_id, dedupe_key)`` — even with
    different ``source_id``, ``mode``, and ``child_ids`` — collide on the
    unique index. The second INSERT must raise.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        source_a = await _mk_mem(session, env_id)
        source_b = await _mk_mem(session, env_id)
        agent_id = await _mk_agent(session)
        await session.execute(
            text(
                "INSERT INTO decompose_operations "
                "(env_id, source_id, mode, dedupe_key, request_fingerprint, "
                " child_ids, created_by_agent_id) "
                "VALUES (:env, :src, 'derive', 'shared_key', 'fp_a', "
                "        ARRAY[gen_random_uuid()]::uuid[], :agent)"
            ),
            {"env": env_id, "src": source_a, "agent": agent_id},
        )
        await session.commit()

    async with factory() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text(
                    "INSERT INTO decompose_operations "
                    "(env_id, source_id, mode, dedupe_key, request_fingerprint, "
                    " child_ids, created_by_agent_id) "
                    "VALUES (:env, :src, 'split', 'shared_key', 'fp_b', "
                    "        ARRAY[gen_random_uuid()]::uuid[], :agent)"
                ),
                {"env": env_id, "src": source_b, "agent": agent_id},
            )
            await session.commit()


@pytest.mark.asyncio
async def test_decompose_operations_source_restrict(
    postgres_session_factories: SessionPairFactory, clean_db: None
) -> None:
    """``source_id`` FK is ON DELETE RESTRICT — cannot delete a cited memory."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        source = await _mk_mem(session, env_id)
        agent_id = await _mk_agent(session)
        await session.execute(
            text(
                "INSERT INTO decompose_operations "
                "(env_id, source_id, mode, dedupe_key, request_fingerprint, "
                " child_ids, created_by_agent_id) "
                "VALUES (:env, :src, 'derive', 'k', 'fp', ARRAY[]::uuid[], :agent)"
            ),
            {"env": env_id, "src": source, "agent": agent_id},
        )
        await session.commit()

    async with factory() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text("DELETE FROM memories WHERE id = :id"),
                {"id": source},
            )
            await session.commit()


# ---------------------------------------------------------------------------
# 8. Recount-pass parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recount_pass_parity_after_0021(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Recount pass agrees with the live triggers on the post-0021 whitelist.

    Wire one ``split_from`` parent (excluded by both triggers and recount)
    and one ``derived_from`` parent (included by both). The live trigger
    leaves the canonical counters at the right values; the recount pass
    should report zero drift on the same data.
    """

    from memory_mcp import memories as memories_mod
    from memory_mcp.dream.passes import recount as recount_mod

    monkeypatch.setattr(recount_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        parent_split = await _mk_mem(session, env_id)
        parent_derive = await _mk_mem(session, env_id)
        child = await _mk_mem(session, env_id)
        await _mk_lineage(session, parent_split, child, "split_from")
        await _mk_lineage(session, parent_derive, child, "derived_from")
        await session.commit()

        # Confirm trigger-side baseline.
        assert (await _read_counts(session, parent_split))["lineage"] == 0
        assert (await _read_counts(session, parent_derive))["lineage"] == 1

    # Seed the agents row for the recount pass's audit-log writes.
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[])
    async with factory() as session:
        await session.execute(
            text("INSERT INTO agents (id, name) VALUES (:id, :name) ON CONFLICT DO NOTHING"),
            {"id": ctx.agent_id, "name": f"recount-{ctx.agent_id.hex[:8]}"},
        )
        await session.commit()

    token = use_session_factory(factory)
    try:
        result = await recount_mod.run_recount(
            env_id,
            actor_ctx=ctx,
            settings=Settings(),
            now=dt.datetime.now(dt.UTC),
        )
    finally:
        reset_session_factory(token)

    assert result.memories_adjusted == 0
    assert result.drift_lineage == 0
    assert result.drift_rel_link == 0
    assert result.drift_task == 0
    assert result.drift_playbook == 0
