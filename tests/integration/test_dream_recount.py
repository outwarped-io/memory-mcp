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
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from memory_mcp.config import Settings
from memory_mcp.dream.passes.recount import _q
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
    macro: str | None = None,
) -> UUID:
    mem_id = uuid4()
    # Phase 1e-d test-fixture safety net: playbooks created via _mk_mem
    # are now reachable from _recompute_salience_for via the
    # formula-version mismatch backfill (every active row in a fresh
    # test env starts at salience_formula_version=0 < target=1).
    # ``memory_update`` requires a non-empty macro on playbook kind, so
    # synthesize one if the caller didn't supply one.
    if kind == "playbook" and not macro:
        macro = f"test-macro-{mem_id}"
    cols = ["id", "env_id", "kind", "status", "body"]
    params: dict[str, object] = {
        "id": mem_id,
        "env_id": env_id,
        "kind": kind,
        "status": status,
        "body": body,
    }
    if steps is not None:
        cols.append("steps")
        params["steps"] = steps
    if macro is not None:
        cols.append("macro")
        params["macro"] = macro
    col_sql = ", ".join(cols)
    val_sql = ", ".join(f":{c}" for c in cols)
    await session.execute(
        text(f"INSERT INTO memories ({col_sql}) VALUES ({val_sql})"),
        params,
    )
    return mem_id


async def _mk_task(session, env_id: UUID) -> UUID:
    task_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO tasks (id, env_id, status, title, description) VALUES (:id, :env_id, 'pending', 't', 'task')"
        ),
        {"id": task_id, "env_id": env_id},
    )
    return task_id


async def _mk_gn_mem(session, env_id: UUID, memory_id: UUID) -> UUID:
    gn_id = uuid4()
    await session.execute(
        text("INSERT INTO graph_nodes (id, env_id, node_type, memory_id) VALUES (:id, :env_id, 'memory', :memory_id)"),
        {"id": gn_id, "env_id": env_id, "memory_id": memory_id},
    )
    return gn_id


async def _mk_gn_task(session, env_id: UUID, task_id: UUID) -> UUID:
    gn_id = uuid4()
    await session.execute(
        text("INSERT INTO graph_nodes (id, env_id, node_type, task_id) VALUES (:id, :env_id, 'task', :task_id)"),
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


async def _set_counter_raw(session, mem_id: UUID, *, rl: int = 0, ln: int = 0, tk: int = 0, pb: int = 0) -> None:
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


async def _run(env_id: UUID, monkeypatch, factory, *, settings: Settings | None = None):
    """Invoke ``run_recount`` with session_scope routed to ``factory``.

    Also patches ``memory_mcp.memories.session_scope`` so the R-B3
    salience-recompute path (which calls ``memory_update``) lands in
    the test container, and pre-inserts the actor's ``agents`` row so
    the audit_log FK doesn't fire.
    """

    from memory_mcp import memories as memories_mod
    from memory_mcp.dream.passes import recount as recount_mod

    monkeypatch.setattr(recount_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    ctx = _ctx()
    # ``memory_update`` writes audit_log rows with by_agent_id pointing
    # at ``agents.id`` — pre-seed so the R-B3 path's API call doesn't
    # trip the FK constraint.
    async with factory() as _agent_session:
        await _agent_session.execute(
            text("INSERT INTO agents (id, name) VALUES (:id, :name) ON CONFLICT DO NOTHING"),
            {"id": ctx.agent_id, "name": f"test-{ctx.agent_id.hex[:8]}"},
        )
        await _agent_session.commit()
    token = use_session_factory(factory)
    try:
        return await recount_mod.run_recount(
            env_id,
            actor_ctx=ctx,
            settings=settings or _settings(),
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
            text("UPDATE memories SET status = 'superseded',   superseded_by = :p WHERE id = :a"),
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
                "INSERT INTO memory_lineage (parent_memory_id, child_memory_id, relation) VALUES (:p, :c, 'supersedes')"
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
            session,
            env_id,
            gn_src,
            gn_dst,
            rel_type="related_to_popular",
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

    from memory_mcp_schemas.dream import DreamMode

    import dream_worker.jobs as jobs_mod
    from dream_worker.jobs import run_dream_pass
    from memory_mcp.db import postgres as pg_mod
    from memory_mcp.dream.passes import recount as recount_mod

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
                text("INSERT INTO agents (id, name) VALUES (:id, :name) ON CONFLICT (id) DO NOTHING"),
                {"id": agent_id, "name": f"recount-test-{agent_id}"},
            )
            await session.commit()

        report = await run_dream_pass(
            env_id=env_id,
            mode=DreamMode.recount,
            actor_ctx=AgentContext(agent_id=agent_id, attached_env_ids=[env_id]),
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
                text("SELECT mode FROM dream_runs WHERE env_id = :env_id AND mode = 'recount'"),
                {"env_id": env_id},
            )
        ).all()
    assert len(rows) >= 1


# ===========================================================================
# Phase 1e — authority leg coverage (slice 1e-c, v0.14.1)
# ===========================================================================
#
# 13 new tests covering the gated authority recount + R-B3 salience
# recompute. Layout mirrors the existing sections: helpers first, then
# tests grouped by concern (knob gating, initial recount, drift,
# idempotency, exclusion rules, R-B3 recompute, eventual correction).


async def _set_salience(session, mem_id: UUID, value: float) -> None:
    """Force a memory's salience to a known value to seed citer-salience scenarios.

    Phase 1e-d: also stamp ``salience_formula_version=1`` so the recount
    pass's formula-version backfill leg does NOT pull the row into
    ``mismatched_ids`` and recompute the pinned salience away. Manual
    fixtures bypass the formula by design — that's the whole point of
    ``_set_salience``.
    """

    await session.execute(
        text("UPDATE memories SET salience = :v, salience_formula_version = 1 WHERE id = :id"),
        {"v": value, "id": mem_id},
    )


async def _read_authority(session, mem_id: UUID) -> dict[str, object]:
    """Read the four ``ref_authority_*`` columns + total + recount stamp."""

    row = (
        await session.execute(
            text(
                "SELECT ref_authority_rel_link, ref_authority_lineage, "
                "       ref_authority_task, ref_authority_playbook, "
                "       reference_authority, authority_last_recount_at "
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
        "stamp": row[5],
    }


async def _set_authority_raw(
    session,
    mem_id: UUID,
    *,
    rl: float = 0.0,
    ln: float = 0.0,
    tk: float = 0.0,
    pb: float = 0.0,
) -> None:
    """Force ``ref_authority_*`` to specific values to seed drift."""

    await session.execute(
        text(
            "UPDATE memories SET "
            "  ref_authority_rel_link = :rl, "
            "  ref_authority_lineage  = :ln, "
            "  ref_authority_task     = :tk, "
            "  ref_authority_playbook = :pb "
            "WHERE id = :mid"
        ),
        {"rl": rl, "ln": ln, "tk": tk, "pb": pb, "mid": mem_id},
    )


def _settings_authority_on(**overrides) -> Settings:
    """Settings with ``dream_popularity_authority_weighted=True`` (knob ON)."""

    return Settings(dream_popularity_authority_weighted=True, **overrides)


# ---------------------------------------------------------------------------
# Test 13: knob off skips authority leg entirely
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authority_off_skips_authority_leg(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Default Settings (knob off) → no authority columns touched, no stamp set."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m_src = await _mk_mem(session, env_id)
        m_dst = await _mk_mem(session, env_id)
        await _set_salience(session, m_src, 0.5)
        gn_src = await _mk_gn_mem(session, env_id, m_src)
        gn_dst = await _mk_gn_mem(session, env_id, m_dst)
        await _mk_rel(session, env_id, gn_src, gn_dst)
        await session.commit()

    # Run with default settings (knob OFF).
    result = await _run(env_id, monkeypatch, factory)
    assert result.memories_authority_adjusted == 0
    assert result.drift_authority_rel_link == Decimal("0")
    assert result.drift_authority_lineage == Decimal("0")
    assert result.drift_authority_task == Decimal("0")
    assert result.drift_authority_playbook == Decimal("0")

    async with factory() as session:
        auth = await _read_authority(session, m_dst)
    # Defaults persist.
    assert auth["rel_link"] == Decimal("0.000000")
    assert auth["total"] == Decimal("0.000000")
    assert auth["stamp"] is None


# ---------------------------------------------------------------------------
# Test 14: initial recount sums citer salience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authority_initial_recount_sums_citer_salience(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Two citers with salience 0.4 and 0.7 → target's rel_link authority = 1.1."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        a = await _mk_mem(session, env_id)
        b = await _mk_mem(session, env_id)
        dst = await _mk_mem(session, env_id)
        await _set_salience(session, a, 0.4)
        await _set_salience(session, b, 0.7)
        gn_a = await _mk_gn_mem(session, env_id, a)
        gn_b = await _mk_gn_mem(session, env_id, b)
        gn_dst = await _mk_gn_mem(session, env_id, dst)
        await _mk_rel(session, env_id, gn_a, gn_dst)
        await _mk_rel(session, env_id, gn_b, gn_dst)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings_authority_on())
    assert result.memories_authority_adjusted == 1
    assert _q(result.drift_authority_rel_link) == _q(Decimal("1.1"))

    async with factory() as session:
        auth = await _read_authority(session, dst)
    assert auth["rel_link"] == Decimal("1.100000")
    assert auth["total"] == Decimal("1.100000")
    assert auth["stamp"] is not None


# ---------------------------------------------------------------------------
# Test 15: drift correction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authority_drift_corrected_by_recount(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Manually mis-set ``ref_authority_rel_link = 99`` → recount restores canonical."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        src = await _mk_mem(session, env_id)
        dst = await _mk_mem(session, env_id)
        await _set_salience(session, src, 0.5)
        gn_src = await _mk_gn_mem(session, env_id, src)
        gn_dst = await _mk_gn_mem(session, env_id, dst)
        await _mk_rel(session, env_id, gn_src, gn_dst)
        await _set_authority_raw(session, dst, rl=99.0)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings_authority_on())
    assert result.memories_authority_adjusted == 1
    # canonical 0.5 - current 99 = -98.5
    assert _q(result.drift_authority_rel_link) == _q(Decimal("-98.5"))

    async with factory() as session:
        auth = await _read_authority(session, dst)
    assert auth["rel_link"] == Decimal("0.500000")


# ---------------------------------------------------------------------------
# Test 16: idempotency at α=1.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authority_idempotent_at_alpha_1_0(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Two consecutive recount runs with α=1.0 — second reports zero authority drift.

    Idempotency holds only at α=1.0 (default). At α<1.0 the blend is
    monotonic-convergent but not idempotent — covered by separate
    test if/when damping is exercised.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        src = await _mk_mem(session, env_id)
        dst = await _mk_mem(session, env_id)
        await _set_salience(session, src, 0.6)
        gn_src = await _mk_gn_mem(session, env_id, src)
        gn_dst = await _mk_gn_mem(session, env_id, dst)
        await _mk_rel(session, env_id, gn_src, gn_dst)
        await session.commit()

    settings_on = _settings_authority_on()
    first = await _run(env_id, monkeypatch, factory, settings=settings_on)
    assert first.memories_authority_adjusted >= 1

    second = await _run(env_id, monkeypatch, factory, settings=settings_on)
    assert second.memories_authority_adjusted == 0
    assert second.drift_authority_rel_link == Decimal("0")
    assert second.drift_authority_lineage == Decimal("0")
    assert second.drift_authority_task == Decimal("0")
    assert second.drift_authority_playbook == Decimal("0")


# ---------------------------------------------------------------------------
# Test 17: retired citer excluded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retired_citer_excluded_from_authority(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Retired src memory's salience does not contribute to dst authority."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        active_src = await _mk_mem(session, env_id, status="active")
        retired_src = await _mk_mem(session, env_id, status="retired")
        dst = await _mk_mem(session, env_id)
        await _set_salience(session, active_src, 0.5)
        await _set_salience(session, retired_src, 0.8)
        gn_active = await _mk_gn_mem(session, env_id, active_src)
        gn_retired = await _mk_gn_mem(session, env_id, retired_src)
        gn_dst = await _mk_gn_mem(session, env_id, dst)
        await _mk_rel(session, env_id, gn_active, gn_dst)
        await _mk_rel(session, env_id, gn_retired, gn_dst)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings_authority_on())
    # Only the active citer contributes: 0.5
    assert _q(result.drift_authority_rel_link) == _q(Decimal("0.5"))

    async with factory() as session:
        auth = await _read_authority(session, dst)
    assert auth["rel_link"] == Decimal("0.500000")


# ---------------------------------------------------------------------------
# Test 18: task source contributes zero authority (D1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_source_contributes_zero_authority(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Task edges increment integer counter but contribute 0 to authority (D1)."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        dst = await _mk_mem(session, env_id)
        task = await _mk_task(session, env_id)
        gn_src = await _mk_gn_task(session, env_id, task)
        gn_dst = await _mk_gn_mem(session, env_id, dst)
        await _mk_rel(session, env_id, gn_src, gn_dst, rel_type="references")
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings_authority_on())
    # Integer counter bumps but authority stays at 0.
    assert result.drift_authority_task == Decimal("0")
    assert result.drift_authority_rel_link == Decimal("0")

    async with factory() as session:
        counts = await _read_counts(session, dst)
        auth = await _read_authority(session, dst)
    assert counts["task"] == 1
    assert auth["task"] == Decimal("0.000000")
    assert auth["total"] == Decimal("0.000000")


# ---------------------------------------------------------------------------
# Test 19: salience recomputed after integer-counter drift (R-B3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_salience_recomputed_after_integer_counter_drift(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Counter drift + wrong salience → recount fixes both via outbox path (R-B3).

    Verifies (a) integer counter restored, (b) salience recomputed
    consistent with canonical inputs, (c) outbox event enqueued (the
    memory_update API path that keeps the Qdrant payload aligned).
    """

    from memory_mcp.dream.salience import (
        SalienceInputs,
        compute_salience,
        salience_weights_from_settings,
    )

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        src = await _mk_mem(session, env_id)
        dst = await _mk_mem(session, env_id)
        await _set_salience(session, src, 0.5)
        gn_src = await _mk_gn_mem(session, env_id, src)
        gn_dst = await _mk_gn_mem(session, env_id, dst)
        await _mk_rel(session, env_id, gn_src, gn_dst)
        # Drift: counter forced too-high, salience stale.
        await _set_counter_raw(session, dst, rl=99)
        await _set_salience(session, dst, 0.5)
        await session.commit()

    # Run with knob OFF — R-B3 must still recompute salience for
    # the counter-drifted row (D6).
    result = await _run(env_id, monkeypatch, factory)
    assert result.memories_adjusted >= 1
    assert result.memories_salience_recomputed >= 1
    assert result.salience_version_conflicts == 0

    async with factory() as session:
        # (a) integer counter restored
        counts = await _read_counts(session, dst)
        assert counts["rel_link"] == 1
        # (b) salience recomputed consistent with current canonical inputs
        row = (
            await session.execute(
                text(
                    "SELECT salience, access_count, last_accessed_at, "
                    "       confidence, pinned, negative_feedback_count, "
                    "       verified_at, created_at, "
                    "       reference_count_rel_link, reference_count_lineage, "
                    "       reference_count_task, reference_count_playbook "
                    "FROM memories WHERE id = :id"
                ),
                {"id": dst},
            )
        ).one()
        # (c) outbox enqueued — at least one event for dst memory
        outbox_rows = (
            await session.execute(
                text("SELECT count(*) FROM outbox WHERE aggregate_id = :id AND aggregate_type = 'memory'"),
                {"id": dst},
            )
        ).scalar_one()

    inputs = SalienceInputs(
        access_count=int(row[1] or 0),
        last_accessed_at=row[2],
        confidence=float(row[3] or 0.0),
        pinned=bool(row[4]),
        negative_feedback_count=int(row[5] or 0),
        verified_at=row[6],
        created_at=row[7],
        reference_count_rel_link=int(row[8] or 0),
        reference_count_lineage=int(row[9] or 0),
        reference_count_task=int(row[10] or 0),
        reference_count_playbook=int(row[11] or 0),
    )
    expected = compute_salience(
        inputs,
        now=dt.datetime.now(dt.UTC),
        weights=salience_weights_from_settings(_settings()),
    )
    # Tight tolerance — recency term has a few seconds of drift from
    # test elapsed time, but salience math is deterministic enough.
    assert abs(float(row[0]) - expected) < 0.01, f"stored salience {row[0]} differs from canonical {expected}"
    assert outbox_rows >= 1


# ---------------------------------------------------------------------------
# Test 20: next pass heals new edge after recount (eventual correction)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_next_pass_heals_new_edge_after_recount(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """New edge inserted between two recount passes is picked up on pass 2.

    Models the eventual-correction concurrency contract (R-S8) — NOT
    a mid-pass race. A true race test would need an injected barrier
    inside ``run_recount`` and is deferred. Asserts no NULL / negative
    values land in any authority column at any point.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        src_a = await _mk_mem(session, env_id)
        src_b = await _mk_mem(session, env_id)
        dst = await _mk_mem(session, env_id)
        await _set_salience(session, src_a, 0.5)
        await _set_salience(session, src_b, 0.5)
        gn_a = await _mk_gn_mem(session, env_id, src_a)
        gn_dst = await _mk_gn_mem(session, env_id, dst)
        await _mk_gn_mem(session, env_id, src_b)  # GN exists, edge does not yet
        await _mk_rel(session, env_id, gn_a, gn_dst)
        await session.commit()

    settings_on = _settings_authority_on()
    # Pass 1: only src_a contributes
    await _run(env_id, monkeypatch, factory, settings=settings_on)
    async with factory() as session:
        auth1 = await _read_authority(session, dst)
    assert auth1["rel_link"] == Decimal("0.500000")
    assert auth1["rel_link"] >= Decimal("0")  # no negative

    # Insert second edge after pass 1 lands
    async with factory() as session:
        # Refetch gn ids for src_b
        gn_b_row = (
            await session.execute(
                text("SELECT id FROM graph_nodes WHERE memory_id = :m"),
                {"m": src_b},
            )
        ).one()
        gn_dst_row = (
            await session.execute(
                text("SELECT id FROM graph_nodes WHERE memory_id = :m"),
                {"m": dst},
            )
        ).one()
        await _mk_rel(session, env_id, gn_b_row[0], gn_dst_row[0])
        await session.commit()

    # Pass 2: sums both
    await _run(env_id, monkeypatch, factory, settings=settings_on)
    async with factory() as session:
        auth2 = await _read_authority(session, dst)
    assert auth2["rel_link"] == Decimal("1.000000")
    assert auth2["total"] == Decimal("1.000000")
    assert auth2["rel_link"] >= Decimal("0")


# ---------------------------------------------------------------------------
# Test 21: supersede-chain ancestry excluded from authority (parity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersede_chain_ancestry_excluded_from_authority(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``a → a'`` rel_link inside supersede chain contributes 0 to authority."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        a = await _mk_mem(session, env_id)
        a_prime = await _mk_mem(session, env_id)
        gn_a = await _mk_gn_mem(session, env_id, a)
        gn_ap = await _mk_gn_mem(session, env_id, a_prime)
        # Pre-set salience while a is still active so citer lookup
        # would have a value to contribute (if not for chain exclusion).
        await _set_salience(session, a, 0.7)
        await _set_salience(session, a_prime, 0.5)
        await _mk_rel(session, env_id, gn_a, gn_ap, rel_type="mentions")
        await session.commit()
        # Now make a a member of a_prime's supersede chain.
        await session.execute(
            text("UPDATE memories SET status = 'superseded',   superseded_by = :p WHERE id = :a"),
            {"p": a_prime, "a": a},
        )
        # Seed authority drift to prove recount zeroes it.
        await _set_authority_raw(session, a_prime, rl=99.0)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings_authority_on())
    # Authority canonical = 0 (a is in chain AND retired; both rules
    # exclude). Drift = 0 - 99 = -99.
    assert _q(result.drift_authority_rel_link) == _q(Decimal("-99.0"))

    async with factory() as session:
        auth = await _read_authority(session, a_prime)
    assert auth["rel_link"] == Decimal("0.000000")


# ---------------------------------------------------------------------------
# Test 22: self-citation excluded from authority (D8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_citation_excluded_from_authority(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """A memory citing itself contributes 0 to its own authority (D8).

    Prevents fixed-point feedback once slice 1e-d wires authority into
    the salience formula. The integer counter may include self-cites
    (Phase 1 contract unchanged); only authority enforces D8.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m = await _mk_mem(session, env_id)
        await _set_salience(session, m, 0.5)
        gn = await _mk_gn_mem(session, env_id, m)
        # Self-citation: src == dst
        await _mk_rel(session, env_id, gn, gn, rel_type="mentions")
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings_authority_on())
    assert result.drift_authority_rel_link == Decimal("0")

    async with factory() as session:
        auth = await _read_authority(session, m)
    assert auth["rel_link"] == Decimal("0.000000")
    assert auth["total"] == Decimal("0.000000")


# ---------------------------------------------------------------------------
# Test 23: ``related_to_popular`` excluded from authority (parity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_related_to_popular_excluded_from_authority(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Phase 4 auto-wire predicate contributes 0 to authority."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        src = await _mk_mem(session, env_id)
        dst = await _mk_mem(session, env_id)
        await _set_salience(session, src, 0.6)
        gn_src = await _mk_gn_mem(session, env_id, src)
        gn_dst = await _mk_gn_mem(session, env_id, dst)
        await _mk_rel(
            session,
            env_id,
            gn_src,
            gn_dst,
            rel_type="related_to_popular",
        )
        # Seed authority drift to prove recount zeroes it.
        await _set_authority_raw(session, dst, rl=42.0)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings_authority_on())
    assert _q(result.drift_authority_rel_link) == _q(Decimal("-42.0"))

    async with factory() as session:
        auth = await _read_authority(session, dst)
    assert auth["rel_link"] == Decimal("0.000000")


# ---------------------------------------------------------------------------
# Test 24: cross-env playbook macro excluded from authority (parity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_env_playbook_macro_excluded_from_authority(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """A playbook in env A citing a memory in env B → no authority in B."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_a = await _mk_env(session)
        env_b = await _mk_env(session)
        target_b = await _mk_mem(session, env_b)
        # Playbook lives in env_a, cites target_b in env_b.
        pb = await _mk_mem(
            session,
            env_a,
            kind="playbook",
            steps=[f"Cite {{{{memory:{target_b}}}}}"],
        )
        await _set_salience(session, pb, 0.8)
        await session.commit()

    # Recount env A — cross-env macro is silently dropped.
    result_a = await _run(env_a, monkeypatch, factory, settings=_settings_authority_on())
    assert result_a.drift_authority_playbook == Decimal("0")

    # Recount env B — no playbook in env B → nothing to scan.
    result_b = await _run(env_b, monkeypatch, factory, settings=_settings_authority_on())
    assert result_b.drift_authority_playbook == Decimal("0")

    async with factory() as session:
        auth = await _read_authority(session, target_b)
    assert auth["playbook"] == Decimal("0.000000")
    assert auth["total"] == Decimal("0.000000")


# ---------------------------------------------------------------------------
# Test 25: lineage and playbook authority sum correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lineage_and_playbook_authority_sum_correctly(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Combined: child contributes via lineage (0.3); playbook cites target 3× (0.6·3=1.8)."""

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        parent = await _mk_mem(session, env_id)
        child = await _mk_mem(session, env_id)
        await _set_salience(session, child, 0.3)
        # Lineage edge: child summarized_from parent (child contributes)
        await session.execute(
            text(
                "INSERT INTO memory_lineage (parent_memory_id, child_memory_id, relation) "
                "VALUES (:p, :c, 'summarized_from')"
            ),
            {"p": parent, "c": child},
        )
        # Playbook cites parent 3 times across its steps.
        pb = await _mk_mem(
            session,
            env_id,
            kind="playbook",
            steps=[
                f"Step 1 {{{{memory:{parent}}}}}",
                f"Step 2 {{{{memory:{parent}}}}} again",
                f"Step 3 {{{{memory:{parent}}}}} third",
            ],
        )
        await _set_salience(session, pb, 0.6)
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings_authority_on())
    # parent.lineage = 0.3 (one child contributes)
    # parent.playbook = 0.6 * 3 = 1.8 (per-occurrence)
    assert _q(result.drift_authority_lineage) == _q(Decimal("0.3"))
    assert _q(result.drift_authority_playbook) == _q(Decimal("1.8"))

    async with factory() as session:
        auth = await _read_authority(session, parent)
    assert auth["lineage"] == Decimal("0.300000")
    assert auth["playbook"] == Decimal("1.800000")
    assert auth["total"] == Decimal("2.100000")


# ---------------------------------------------------------------------------
# Phase 1e-d (v0.14.1) — formula-version backfill + authority-in-formula
# ---------------------------------------------------------------------------


async def _read_salience_and_version(session, mem_id: UUID) -> tuple[float, int]:
    row = (
        await session.execute(
            text("SELECT salience, salience_formula_version FROM memories WHERE id = :id"),
            {"id": mem_id},
        )
    ).one()
    return float(row[0] or 0), int(row[1] or 0)


@pytest.mark.asyncio
async def test_formula_version_mismatch_triggers_recompute(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """A row whose stored ``salience_formula_version`` is behind the
    settings target is picked up by recount and re-stamped: both
    ``salience`` and ``salience_formula_version`` advance together.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        # Active row; default DB column is salience_formula_version=0
        # (per Migration 0019), settings target = 1, so the row is
        # "behind" and will be re-stamped.
        m = await _mk_mem(session, env_id, body="m")
        # Pin an unusual salience so we can detect that recount overwrote
        # it with compute_salience(...) output.
        await session.execute(
            text("UPDATE memories SET salience = 0.123 WHERE id = :id"),
            {"id": m},
        )
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings())

    async with factory() as session:
        sal, ver = await _read_salience_and_version(session, m)

    assert ver == 1
    assert sal != 0.123  # recount overwrote with compute_salience output
    assert result.memories_formula_version_restamped >= 1
    assert result.memories_formula_version_pending == 0


@pytest.mark.asyncio
async def test_formula_version_match_skips_recompute(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """When stored ``salience_formula_version`` already equals the
    settings target AND no counter / authority drift exists, no row is
    pulled into the recompute set: ``restamped=0`` and the seeded
    salience is untouched.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m = await _mk_mem(session, env_id, body="m")
        # Pre-stamp the row at the current formula version + seed a known salience.
        await session.execute(
            text("UPDATE memories SET salience = 0.456, salience_formula_version = 1 WHERE id = :id"),
            {"id": m},
        )
        await session.commit()

    result = await _run(env_id, monkeypatch, factory, settings=_settings())

    async with factory() as session:
        sal, ver = await _read_salience_and_version(session, m)

    # Row was already at target version; no recount work for it.
    assert ver == 1
    assert sal == pytest.approx(0.456, abs=1e-6)
    # Other tests in this file count restamped via the mismatch helper,
    # which only fires for rows behind target.
    assert result.memories_formula_version_restamped == 0
    assert result.memories_formula_version_pending == 0


@pytest.mark.asyncio
async def test_authority_value_lifts_salience_when_knob_on(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """With the authority knob ON, a target whose citers carry positive
    salience ends up with a salience strictly greater than an
    otherwise-identical isolated row that has no citers.

    Validates the authority term in ``compute_salience`` is reached
    through the recount's salience-recompute leg.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        # Citer with a substantial salience (pinned via _set_salience,
        # so the recompute leg leaves it alone — see _set_salience
        # docstring re: salience_formula_version stamping).
        citer = await _mk_mem(session, env_id, body="citer")
        await _set_salience(session, citer, 0.8)
        # Two targets — one cited, one isolated.
        cited = await _mk_mem(session, env_id, body="cited")
        isolated = await _mk_mem(session, env_id, body="isolated")
        gn_citer = await _mk_gn_mem(session, env_id, citer)
        gn_cited = await _mk_gn_mem(session, env_id, cited)
        await _mk_gn_mem(session, env_id, isolated)
        await _mk_rel(session, env_id, gn_citer, gn_cited)
        await session.commit()

    await _run(env_id, monkeypatch, factory, settings=_settings_authority_on())

    async with factory() as session:
        cited_sal, _ = await _read_salience_and_version(session, cited)
        iso_sal, _ = await _read_salience_and_version(session, isolated)
        auth = await _read_authority(session, cited)

    assert auth["rel_link"] == Decimal("0.800000")
    assert cited_sal > iso_sal, (
        f"cited salience ({cited_sal:.4f}) should exceed isolated salience ({iso_sal:.4f}) when authority knob is ON"
    )


@pytest.mark.asyncio
async def test_authority_value_no_effect_when_knob_off(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Even when ``reference_authority > 0`` is present on the row (e.g.
    residual from a prior on-cycle), the salience computed with the
    knob OFF matches an otherwise-identical row with zero authority.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        # Two identical-shaped rows; one carries residual authority,
        # the other does not. Both have salience_formula_version=0 so
        # recount will recompute under knob-OFF weights.
        with_auth = await _mk_mem(session, env_id, body="with-auth")
        no_auth = await _mk_mem(session, env_id, body="no-auth")
        await session.execute(
            text("UPDATE memories SET ref_authority_rel_link = 10.0 WHERE id = :id"),
            {"id": with_auth},
        )
        await session.commit()

    # Default settings — knob OFF.
    await _run(env_id, monkeypatch, factory, settings=_settings())

    async with factory() as session:
        sal_with, _ = await _read_salience_and_version(session, with_auth)
        sal_no, _ = await _read_salience_and_version(session, no_auth)

    # With knob OFF, authority residual must not move salience.
    assert sal_with == sal_no, f"knob-OFF salience must ignore authority: with={sal_with}, no={sal_no}"


@pytest.mark.asyncio
async def test_formula_version_backfill_chunked(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """When more rows are behind the target version than the
    per-cycle cap allows, recount restamps up to ``cap`` rows and
    reports the remainder via ``memories_formula_version_pending``.
    The next cycle picks up where the first left off.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        # 5 active rows, all at formula_version=0 by default.
        for i in range(5):
            await _mk_mem(session, env_id, body=f"m{i}")
        await session.commit()

    # Cap at 2 per cycle.
    settings = Settings(dream_recount_salience_recompute_cap=2)

    first = await _run(env_id, monkeypatch, factory, settings=settings)
    assert first.memories_formula_version_restamped == 2
    assert first.memories_formula_version_pending == 3

    second = await _run(env_id, monkeypatch, factory, settings=settings)
    assert second.memories_formula_version_restamped == 2
    assert second.memories_formula_version_pending == 1

    third = await _run(env_id, monkeypatch, factory, settings=settings)
    assert third.memories_formula_version_restamped == 1
    assert third.memories_formula_version_pending == 0

    fourth = await _run(env_id, monkeypatch, factory, settings=settings)
    assert fourth.memories_formula_version_restamped == 0
    assert fourth.memories_formula_version_pending == 0


@pytest.mark.asyncio
async def test_formula_version_backfill_unbounded_when_cap_zero(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``dream_recount_salience_recompute_cap=0`` means unbounded: one
    cycle drains the entire backlog.
    """

    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        for i in range(4):
            await _mk_mem(session, env_id, body=f"m{i}")
        await session.commit()

    settings = Settings(dream_recount_salience_recompute_cap=0)
    result = await _run(env_id, monkeypatch, factory, settings=settings)

    assert result.memories_formula_version_restamped == 4
    assert result.memories_formula_version_pending == 0
