"""Integration coverage for the dream recount pass (Phase 1, v0.14.0).

The recount pass is the **canonical writer** of the four
``reference_count_*`` columns. Trigger correctness lives in
``test_reference_counts.py``; this file exercises the recount-specific
responsibilities:

* drift reconciliation (manual mis-set of a counter → restored),
* playbook macro text-scan (no edge row backing it),
* supersede-chain ancestry exclusion (a guarantee the triggers
  intentionally do not provide),
* ``supersedes`` lineage exclusion (mirrors the trigger filter),
* cross-env macro suppression,
* idempotency (second run reports zero drift),
* dispatch wiring (``run_dream_pass`` with ``DreamMode.recount`` lands
  a ``dream_runs`` row).
"""

from __future__ import annotations

import datetime as dt
import os
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

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
# Raw-SQL helpers (mirror the conventions in test_reference_counts.py)
# ---------------------------------------------------------------------------


async def _mk_env(session) -> UUID:
    env_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO environments (id, name, kind, default_embedding_model_id) "
            "VALUES (:id, :name, 'test', 'test-embedding')"
        ),
        {"id": env_id, "name": f"recount-{env_id}"},
    )
    return env_id


async def _mk_mem(
    session,
    env_id: UUID,
    *,
    kind: str = "fact",
    status: str = "active",
    body: str = "b",
    steps: list[str] | None = None,
) -> UUID:
    mem_id = uuid4()
    if steps is None:
        await session.execute(
            text(
                "INSERT INTO memories (id, env_id, kind, status, body) "
                "VALUES (:id, :env_id, :kind, :status, :body)"
            ),
            {
                "id": mem_id,
                "env_id": env_id,
                "kind": kind,
                "status": status,
                "body": body,
            },
        )
    else:
        # ``steps`` is a Postgres text[] column on memories.
        await session.execute(
            text(
                "INSERT INTO memories (id, env_id, kind, status, body, steps) "
                "VALUES (:id, :env_id, :kind, :status, :body, :steps)"
            ),
            {
                "id": mem_id,
                "env_id": env_id,
                "kind": kind,
                "status": status,
                "body": body,
                "steps": steps,
            },
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


async def _mk_gn_mem(session, env_id: UUID, memory_id: UUID) -> UUID:
    gn_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO graph_nodes (id, env_id, node_type, memory_id) "
            "VALUES (:id, :env_id, 'memory', :memory_id)"
        ),
        {"id": gn_id, "env_id": env_id, "memory_id": memory_id},
    )
    return gn_id


async def _mk_gn_task(session, env_id: UUID, task_id: UUID) -> UUID:
    gn_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO graph_nodes (id, env_id, node_type, task_id) "
            "VALUES (:id, :env_id, 'task', :task_id)"
        ),
        {"id": gn_id, "env_id": env_id, "task_id": task_id},
    )
    return gn_id


async def _mk_rel(
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
        {
            "id": rel_id,
            "env_id": env_id,
            "src": src_gn,
            "dst": dst_gn,
            "type": rel_type,
        },
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


async def _set_counter_raw(
    session, mem_id: UUID, *, rl: int = 0, ln: int = 0, tk: int = 0, pb: int = 0
) -> None:
    """Force counters to specific (possibly wrong) values to seed drift."""

    await session.execute(
        text(
            "UPDATE memories SET "
            "  reference_count_rel_link = :rl, "
            "  reference_count_lineage  = :ln, "
            "  reference_count_task     = :tk, "
            "  reference_count_playbook = :pb "
            "WHERE id = :mid"
        ),
        {"rl": rl, "ln": ln, "tk": tk, "pb": pb, "mid": mem_id},
    )


def _settings() -> Settings:
    # Recount itself does not consult tunables today, but Settings()
    # validates required env vars (POSTGRES_URL etc.) so we make sure
    # the conftest's testcontainer URL is honored.
    return Settings()


def _ctx() -> AgentContext:
    return AgentContext(agent_id=uuid4(), attached_env_ids=[])


# ---------------------------------------------------------------------------
# Helpers to run the pass under the integration session factory
# ---------------------------------------------------------------------------


async def _run(env_id: UUID, monkeypatch, factory):
    """Invoke ``run_recount`` with session_scope routed to ``factory``."""

    from memory_mcp.dream.passes import recount as recount_mod

    monkeypatch.setattr(recount_mod, "session_scope", routed_session_scope)
    token = use_session_factory(factory)
    try:
        return await recount_mod.run_recount(
            env_id,
            actor_ctx=_ctx(),
            settings=_settings(),
            now=dt.datetime.now(dt.UTC),
        )
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# Tests — drift reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_recount_matches_triggers(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Triggers keep counters at canonical truth — recount sees zero drift."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_mem(session, env_id, m_src)
        gn_dst = await _mk_gn_mem(session, env_id, m_dst)
        await _mk_rel(session, env_id, gn_src, gn_dst, rel_type="mentions")
        await session.commit()

    result = await _run(env_id, monkeypatch, factory)
    assert result.memories_adjusted == 0
    assert result.drift_rel_link == 0
    assert result.drift_lineage == 0
    assert result.drift_task == 0
    assert result.drift_playbook == 0


@pytest.mark.asyncio
async def test_recount_recovers_drift(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Manually mis-set counter → recount restores canonical value."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_mem(session, env_id, m_src)
        gn_dst = await _mk_gn_mem(session, env_id, m_dst)
        await _mk_rel(session, env_id, gn_src, gn_dst)
        await session.commit()

        # Counter is now 1 (trigger). Force it to 99 → +98 drift.
        await _set_counter_raw(session, m_dst, rl=99)
        await session.commit()
        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 99

    result = await _run(env_id, monkeypatch, factory)
    assert result.memories_adjusted == 1
    # Net adjustment is canonical (1) - current (99) = -98.
    assert result.drift_rel_link == -98

    async with factory() as session:
        counts = await _read_counts(session, m_dst)
        assert counts["rel_link"] == 1
        assert counts["total"] == 1


@pytest.mark.asyncio
async def test_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Two consecutive recount runs — second reports zero adjustments."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_mem(session, env_id, m_src)
        gn_dst = await _mk_gn_mem(session, env_id, m_dst)
        await _mk_rel(session, env_id, gn_src, gn_dst)
        await session.commit()
        # Seed drift so the first run does work.
        await _set_counter_raw(session, m_dst, rl=42)
        await session.commit()

    first = await _run(env_id, monkeypatch, factory)
    assert first.memories_adjusted >= 1

    second = await _run(env_id, monkeypatch, factory)
    assert second.memories_adjusted == 0
    assert second.drift_rel_link == 0
    assert second.drift_lineage == 0
    assert second.drift_task == 0
    assert second.drift_playbook == 0


# ---------------------------------------------------------------------------
# Tests — playbook macro scan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_playbook_macro_scan(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``{{memory:<uuid>}}`` occurrences across steps accumulate per-target."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        target_a = await _mk_mem(session, env_id, body="target a")
        target_b = await _mk_mem(session, env_id, body="target b")
        # Playbook with 3 steps:
        # - step 1 cites A
        # - step 2 cites A again (duplicates count)
        # - step 3 cites B once
        await _mk_mem(
            session,
            env_id,
            kind="playbook",
            body="pb",
            steps=[
                f"Recall {{{{memory:{target_a}}}}} before starting.",
                f"Compare with {{{{memory:{target_a}}}}} once more.",
                f"Finalize using {{{{memory:{target_b}}}}}.",
            ],
        )
        await session.commit()

    result = await _run(env_id, monkeypatch, factory)
    assert result.playbooks_scanned == 1
    # 2 macros for A + 1 for B = 3 net new playbook citations
    assert result.drift_playbook == 3

    async with factory() as session:
        ca = await _read_counts(session, target_a)
        cb = await _read_counts(session, target_b)
    assert ca["playbook"] == 2
    assert cb["playbook"] == 1


@pytest.mark.asyncio
async def test_playbook_macro_inactive_skipped(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Retired playbooks must not contribute citations."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        target = await _mk_mem(session, env_id)
        # Playbook is retired → recount excludes it.
        await _mk_mem(
            session,
            env_id,
            kind="playbook",
            status="retired",
            steps=[f"Cite {{{{memory:{target}}}}}"],
        )
        await session.commit()

    result = await _run(env_id, monkeypatch, factory)
    assert result.playbooks_scanned == 0

    async with factory() as session:
        c = await _read_counts(session, target)
    assert c["playbook"] == 0


@pytest.mark.asyncio
async def test_cross_env_playbook_macros_dropped(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """A playbook citing a memory in another env must not bump that memory."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_a = await _mk_env(session)
        env_b = await _mk_env(session)
        target_b = await _mk_mem(session, env_b)
        # Playbook lives in env_a; cites target_b in env_b.
        await _mk_mem(
            session,
            env_a,
            kind="playbook",
            steps=[f"Cross-env cite {{{{memory:{target_b}}}}}"],
        )
        await session.commit()

    # Recount env_a: scan playbook but drop cross-env target.
    result_a = await _run(env_a, monkeypatch, factory)
    assert result_a.playbooks_scanned == 1
    assert result_a.drift_playbook == 0  # nothing eligible
    # Recount env_b: no playbook in env_b → nothing to scan.
    result_b = await _run(env_b, monkeypatch, factory)
    assert result_b.playbooks_scanned == 0

    async with factory() as session:
        c = await _read_counts(session, target_b)
    assert c["playbook"] == 0


# ---------------------------------------------------------------------------
# Tests — supersede-chain ancestry exclusion (S6 — recount only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersede_chain_ancestry_excluded(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``a → a'`` rel_link inside the same supersede chain doesn't count.

    The trigger bumps ``a'``'s rel_link to 1 (it can't afford the
    chain check at write time). Recount is canonical and must
    decrement that back to 0.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        a = await _mk_mem(session, env_id)
        a_prime = await _mk_mem(session, env_id)
        # a' supersedes a — set both the structural pointer AND the
        # status to satisfy ``memories_superseded_status_chk`` (status
        # must be 'superseded' iff superseded_by is set). NB: since a
        # is being marked superseded BEFORE the rel_link is created,
        # the rel_link trigger sees a retired-style src and won't bump
        # a_prime's counter — so we then re-activate a temporarily
        # to seed the trigger increment, then re-mark superseded with
        # the structural pointer intact.
        gn_a = await _mk_gn_mem(session, env_id, a)
        gn_ap = await _mk_gn_mem(session, env_id, a_prime)
        # rel_link a → a' WHILE a is still active → trigger bumps to 1.
        await _mk_rel(session, env_id, gn_a, gn_ap, rel_type="mentions")
        await session.commit()

        pre = await _read_counts(session, a_prime)
        assert pre["rel_link"] == 1

        # Now set the supersede pointer + flip status atomically.
        # The status-flip trigger then decrements a_prime by 1 (since
        # a is the src memory) → a_prime.rel_link == 0 at trigger
        # truth. We then manually force a_prime.rel_link back to 1
        # to simulate "trigger missed ancestry but counter is high"
        # — the case recount specifically exists to fix.
        await session.execute(
            text(
                "UPDATE memories SET status = 'superseded', "
                "  superseded_by = :p WHERE id = :a"
            ),
            {"p": a_prime, "a": a},
        )
        await session.commit()
        # Force the counter back to 1 to simulate ancestry-via-edge drift.
        await _set_counter_raw(session, a_prime, rl=1)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory)
    # Both ancestry exclusion AND retired-src exclusion drive
    # canonical to 0; current is 1; drift is -1.
    assert result.drift_rel_link == -1

    async with factory() as session:
        post = await _read_counts(session, a_prime)
    assert post["rel_link"] == 0


# ---------------------------------------------------------------------------
# Tests — lineage / status mirroring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersedes_lineage_excluded(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``relation='supersedes'`` is not in the recount lineage whitelist."""

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
        # Force a bogus lineage count to prove recount actively zeroes it.
        await _set_counter_raw(session, parent, ln=5)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory)
    assert result.drift_lineage == -5

    async with factory() as session:
        c = await _read_counts(session, parent)
    assert c["lineage"] == 0


@pytest.mark.asyncio
async def test_retired_src_memory_edge_excluded(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Edges sourced from a retired memory are not counted in canonical truth.

    The status-flip trigger decrements them on retire; recount must
    produce the same answer when invoked from a cold start (e.g.,
    fresh deployment that skipped the fast-path backfill).
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id, status="retired")
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_mem(session, env_id, m_src)
        gn_dst = await _mk_gn_mem(session, env_id, m_dst)
        await _mk_rel(session, env_id, gn_src, gn_dst, rel_type="mentions")
        # Force a phony count to prove recount zeroes the retired-src edge.
        await _set_counter_raw(session, m_dst, rl=7)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory)
    assert result.drift_rel_link == -7

    async with factory() as session:
        c = await _read_counts(session, m_dst)
    assert c["rel_link"] == 0


@pytest.mark.asyncio
async def test_related_to_popular_excluded(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Phase 4 auto-wire predicate must not feed popularity back to itself."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_mem(session, env_id, m_src)
        gn_dst = await _mk_gn_mem(session, env_id, m_dst)
        await _mk_rel(
            session, env_id, gn_src, gn_dst, rel_type="related_to_popular",
        )
        # Phony preset so we can see recount zeroing it.
        await _set_counter_raw(session, m_dst, rl=4)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory)
    assert result.drift_rel_link == -4

    async with factory() as session:
        c = await _read_counts(session, m_dst)
    assert c["rel_link"] == 0


@pytest.mark.asyncio
async def test_task_edges_counted_separately(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Task → memory edges land in the ``task`` counter, not ``rel_link``."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_dst = await _mk_mem(session, env_id)
        task = await _mk_task(session, env_id)
        gn_src = await _mk_gn_task(session, env_id, task)
        gn_dst = await _mk_gn_mem(session, env_id, m_dst)
        await _mk_rel(session, env_id, gn_src, gn_dst, rel_type="references")
        # Drift: swap counters so recount has work to do.
        await _set_counter_raw(session, m_dst, rl=1, tk=0)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory)
    # rel_link went from 1 → 0 (-1); task went from 0 → 1 (+1).
    assert result.drift_rel_link == -1
    assert result.drift_task == 1

    async with factory() as session:
        c = await _read_counts(session, m_dst)
    assert c["rel_link"] == 0
    assert c["task"] == 1
    assert c["total"] == 1


# ---------------------------------------------------------------------------
# Tests — dispatcher wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_via_run_dream_pass(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``run_dream_pass(..., DreamMode.recount)`` lands a ``dream_runs`` row."""

    from unittest.mock import MagicMock

    from dream_worker.jobs import run_dream_pass
    from memory_mcp.dream.passes import recount as recount_mod
    from memory_mcp_schemas.dream import DreamMode
    import dream_worker.jobs as jobs_mod
    from memory_mcp.db import postgres as pg_mod

    # Route both the recount-pass session_scope and the jobs-module
    # session_scope (used to write dream_runs).
    monkeypatch.setattr(recount_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(jobs_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(pg_mod, "session_scope", routed_session_scope)

    summarizer = MagicMock()
    summarizer.kind = MagicMock(value="template")

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        gn_src = await _mk_gn_mem(session, env_id, m_src)
        gn_dst = await _mk_gn_mem(session, env_id, m_dst)
        await _mk_rel(session, env_id, gn_src, gn_dst)
        await session.commit()

    token = use_session_factory(factory)
    try:
        # Ensure the agents row exists — run_dream_pass writes
        # dream_runs.agent_id with a FK to agents.
        agent_id = uuid4()
        async with factory() as session:
            await session.execute(
                text(
                    "INSERT INTO agents (id, name) "
                    "VALUES (:id, :name) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                {"id": agent_id, "name": f"recount-test-{agent_id}"},
            )
            await session.commit()

        report = await run_dream_pass(
            env_id=env_id,
            mode=DreamMode.recount,
            actor_ctx=AgentContext(
                agent_id=agent_id, attached_env_ids=[env_id]
            ),
            summarizer=summarizer,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert report.mode == DreamMode.recount
    # dream_runs row landed.
    async with factory() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT mode FROM dream_runs "
                    "WHERE env_id = :env_id AND mode = 'recount'"
                ),
                {"env_id": env_id},
            )
        ).all()
    assert len(rows) >= 1
