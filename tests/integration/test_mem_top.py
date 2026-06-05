"""Integration coverage for ``memory_top`` (mem_top MCP tool, v0.14).

Exercises the ranking metrics end-to-end against a real Postgres:
* ``salience`` / ``access_count`` / ``reference_count`` — column-backed
* ``reference_velocity`` — derived from ``relations`` + ``memory_lineage``
  edge tables over a rolling time window

Also covers:
* tag_match=any (default) and tag_match=all (AND) semantics
* stable tie-breaker `(metric DESC, created_at DESC, id DESC)`
* status filter default (active-only)
* limit cap
"""

from __future__ import annotations

import datetime as dt
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from memory_mcp import top as top_mod
from memory_mcp.db.types import MemoryStatus
from memory_mcp.identity import AgentContext
from memory_mcp.top import MemTopRequest, memory_top

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers (raw SQL so we can control salience / access_count / counters / tags
# deterministically without going through the access-bump path)
# ---------------------------------------------------------------------------


async def _mk_env(session) -> UUID:
    env_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO environments (id, name, kind, default_embedding_model_id) "
            "VALUES (:id, :name, 'test', 'test-embedding')"
        ),
        {"id": env_id, "name": f"memtop-{env_id}"},
    )
    return env_id


async def _mk_mem(
    session,
    env_id: UUID,
    *,
    kind: str = "fact",
    status: str = "active",
    title: str = "m",
    body: str = "b",
    salience: float = 0.0,
    access_count: int = 0,
    created_at: dt.datetime | None = None,
) -> UUID:
    mem_id = uuid4()
    if created_at is None:
        await session.execute(
            text(
                "INSERT INTO memories "
                "(id, env_id, kind, status, title, body, salience, access_count) "
                "VALUES (:id, :env_id, :kind, :status, :title, :body, :salience, :access)"
            ),
            {
                "id": mem_id,
                "env_id": env_id,
                "kind": kind,
                "status": status,
                "title": title,
                "body": body,
                "salience": salience,
                "access": access_count,
            },
        )
    else:
        await session.execute(
            text(
                "INSERT INTO memories "
                "(id, env_id, kind, status, title, body, salience, access_count, created_at) "
                "VALUES (:id, :env_id, :kind, :status, :title, :body, :salience, :access, :created)"
            ),
            {
                "id": mem_id,
                "env_id": env_id,
                "kind": kind,
                "status": status,
                "title": title,
                "body": body,
                "salience": salience,
                "access": access_count,
                "created": created_at,
            },
        )
    return mem_id


async def _mk_gn_for_memory(session, env_id: UUID, memory_id: UUID) -> UUID:
    gn_id = uuid4()
    await session.execute(
        text("INSERT INTO graph_nodes (id, env_id, node_type, memory_id) VALUES (:id, :env_id, 'memory', :memory_id)"),
        {"id": gn_id, "env_id": env_id, "memory_id": memory_id},
    )
    return gn_id


async def _mk_relation(
    session,
    env_id: UUID,
    src_gn: UUID,
    dst_gn: UUID,
    *,
    rel_type: str = "mentions",
    created_at: dt.datetime | None = None,
) -> UUID:
    rel_id = uuid4()
    if created_at is None:
        await session.execute(
            text(
                "INSERT INTO relations (id, env_id, src_node_id, dst_node_id, type) "
                "VALUES (:id, :env_id, :src, :dst, :type)"
            ),
            {"id": rel_id, "env_id": env_id, "src": src_gn, "dst": dst_gn, "type": rel_type},
        )
    else:
        await session.execute(
            text(
                "INSERT INTO relations (id, env_id, src_node_id, dst_node_id, type, created_at) "
                "VALUES (:id, :env_id, :src, :dst, :type, :created)"
            ),
            {
                "id": rel_id,
                "env_id": env_id,
                "src": src_gn,
                "dst": dst_gn,
                "type": rel_type,
                "created": created_at,
            },
        )
    return rel_id


async def _attach_tag(session, env_id: UUID, mem_id: UUID, tag_name: str) -> None:
    tag_id = uuid4()
    await session.execute(
        text("INSERT INTO tags (id, env_id, name) VALUES (:id, :env_id, :name) ON CONFLICT (env_id, name) DO NOTHING"),
        {"id": tag_id, "env_id": env_id, "name": tag_name},
    )
    real_id = (
        await session.execute(
            text("SELECT id FROM tags WHERE env_id = :env_id AND name = :name"),
            {"env_id": env_id, "name": tag_name},
        )
    ).scalar_one()
    await session.execute(
        text(
            "INSERT INTO memory_tags (env_id, memory_id, tag_id) "
            "VALUES (:env_id, :memory_id, :tag_id) ON CONFLICT DO NOTHING"
        ),
        {"env_id": env_id, "memory_id": mem_id, "tag_id": real_id},
    )


async def _ctx_with_env(env_id: UUID) -> AgentContext:
    return AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mem_top_by_salience_ranks_descending(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        m_low = await _mk_mem(session, env_id, title="low", salience=0.10)
        m_mid = await _mk_mem(session, env_id, title="mid", salience=0.50)
        m_high = await _mk_mem(session, env_id, title="high", salience=0.95)
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp = await memory_top(MemTopRequest(env_ids=[env_id], by="salience", limit=10), ctx=ctx)
    finally:
        reset_session_factory(token)

    ids = [item.memory.id for item in resp.items]
    assert ids == [m_high, m_mid, m_low]
    metrics = [item.metric_value for item in resp.items]
    assert metrics == pytest.approx([0.95, 0.50, 0.10], abs=1e-5)
    assert resp.by == "salience"
    assert resp.total_examined == 3


@pytest.mark.asyncio
async def test_mem_top_by_reference_count_uses_computed_column(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        m_target = await _mk_mem(session, env_id, title="popular")
        m_other = await _mk_mem(session, env_id, title="not-popular")
        gn_target = await _mk_gn_for_memory(session, env_id, m_target)
        gn_other = await _mk_gn_for_memory(session, env_id, m_other)

        # Create 3 rel_link edges from m_other -> m_target via triggers.
        for _ in range(3):
            src_mem = await _mk_mem(session, env_id, title="src")
            gn_src = await _mk_gn_for_memory(session, env_id, src_mem)
            await _mk_relation(session, env_id, gn_src, gn_target, rel_type="mentions")

        # m_other gets only 1 incoming edge.
        src_mem2 = await _mk_mem(session, env_id, title="src2")
        gn_src2 = await _mk_gn_for_memory(session, env_id, src_mem2)
        await _mk_relation(session, env_id, gn_src2, gn_other, rel_type="mentions")
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp = await memory_top(
            MemTopRequest(env_ids=[env_id], by="reference_count", limit=5),
            ctx=ctx,
        )
    finally:
        reset_session_factory(token)

    ids = [item.memory.id for item in resp.items]
    # m_target ranks first with 3; m_other second with 1.
    assert ids[0] == m_target
    assert ids[1] == m_other
    assert resp.items[0].metric_value == 3.0
    assert resp.items[1].metric_value == 1.0
    # Breakdown surfaces in the response too.
    assert resp.items[0].memory.reference_breakdown == {
        "rel_link": 3,
        "lineage": 0,
        "task": 0,
        "playbook": 0,
    }


@pytest.mark.asyncio
async def test_mem_top_default_status_filter_excludes_non_active(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        active = await _mk_mem(session, env_id, status="active", salience=0.5, title="A")
        proposed = await _mk_mem(session, env_id, status="proposed", salience=0.9, title="P")
        stale = await _mk_mem(session, env_id, status="stale", salience=0.99, title="S")
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        # Default: active only.
        resp = await memory_top(MemTopRequest(env_ids=[env_id], by="salience", limit=10), ctx=ctx)
        active_only_ids = [item.memory.id for item in resp.items]

        # Explicit override: include stale.
        resp2 = await memory_top(
            MemTopRequest(
                env_ids=[env_id],
                by="salience",
                limit=10,
                statuses=[MemoryStatus.active, MemoryStatus.stale, MemoryStatus.proposed],
            ),
            ctx=ctx,
        )
        all_ids = [item.memory.id for item in resp2.items]
    finally:
        reset_session_factory(token)

    assert active_only_ids == [active]
    assert set(all_ids) == {active, proposed, stale}
    # When all three are visible, salience ordering wins (stale 0.99 > proposed 0.9 > active 0.5).
    assert all_ids == [stale, proposed, active]


@pytest.mark.asyncio
async def test_mem_top_tag_match_any_vs_all(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        m_both = await _mk_mem(session, env_id, salience=0.3, title="both")
        m_foo = await _mk_mem(session, env_id, salience=0.6, title="foo-only")
        m_bar = await _mk_mem(session, env_id, salience=0.5, title="bar-only")
        await _mk_mem(session, env_id, salience=0.9, title="no-tags")
        await _attach_tag(session, env_id, m_both, "task:foo")
        await _attach_tag(session, env_id, m_both, "task:bar")
        await _attach_tag(session, env_id, m_foo, "task:foo")
        await _attach_tag(session, env_id, m_bar, "task:bar")
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)

        any_resp = await memory_top(
            MemTopRequest(
                env_ids=[env_id],
                by="salience",
                limit=10,
                tags=["task:foo", "task:bar"],
                tag_match="any",
            ),
            ctx=ctx,
        )
        all_resp = await memory_top(
            MemTopRequest(
                env_ids=[env_id],
                by="salience",
                limit=10,
                tags=["task:foo", "task:bar"],
                tag_match="all",
            ),
            ctx=ctx,
        )
    finally:
        reset_session_factory(token)

    any_ids = {item.memory.id for item in any_resp.items}
    all_ids = {item.memory.id for item in all_resp.items}

    assert any_ids == {m_both, m_foo, m_bar}  # m_none excluded; rest match OR
    assert all_ids == {m_both}  # only m_both has both tags


@pytest.mark.asyncio
async def test_mem_top_stable_tiebreaker(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    now = dt.datetime.now(dt.UTC)

    async with factory() as session:
        env_id = await _mk_env(session)
        # Three memories with identical salience but different created_at.
        # Expected ordering: newest first (created_at DESC).
        m_old = await _mk_mem(
            session,
            env_id,
            salience=0.5,
            title="old",
            created_at=now - dt.timedelta(days=3),
        )
        m_mid = await _mk_mem(
            session,
            env_id,
            salience=0.5,
            title="mid",
            created_at=now - dt.timedelta(days=2),
        )
        m_new = await _mk_mem(
            session,
            env_id,
            salience=0.5,
            title="new",
            created_at=now - dt.timedelta(days=1),
        )
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp = await memory_top(MemTopRequest(env_ids=[env_id], by="salience", limit=10), ctx=ctx)
    finally:
        reset_session_factory(token)

    ids = [item.memory.id for item in resp.items]
    # Tie on salience → created_at DESC wins; all metric values are equal.
    assert ids == [m_new, m_mid, m_old]
    assert [item.metric_value for item in resp.items] == [0.5, 0.5, 0.5]


@pytest.mark.asyncio
async def test_mem_top_by_reference_velocity_window(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    now = dt.datetime.now(dt.UTC)

    async with factory() as session:
        env_id = await _mk_env(session)
        m_recent_pop = await _mk_mem(session, env_id, title="recent-popular")
        m_old_pop = await _mk_mem(session, env_id, title="old-popular")
        gn_recent = await _mk_gn_for_memory(session, env_id, m_recent_pop)
        gn_old = await _mk_gn_for_memory(session, env_id, m_old_pop)

        # m_recent_pop: 2 fresh edges (within 30d default)
        for _ in range(2):
            src_mem = await _mk_mem(session, env_id)
            gn_src = await _mk_gn_for_memory(session, env_id, src_mem)
            await _mk_relation(
                session,
                env_id,
                gn_src,
                gn_recent,
                created_at=now - dt.timedelta(days=5),
            )

        # m_old_pop: 5 STALE edges (older than 30d) — should not count toward
        # default velocity window.
        for _ in range(5):
            src_mem = await _mk_mem(session, env_id)
            gn_src = await _mk_gn_for_memory(session, env_id, src_mem)
            await _mk_relation(
                session,
                env_id,
                gn_src,
                gn_old,
                created_at=now - dt.timedelta(days=60),
            )
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp = await memory_top(
            MemTopRequest(env_ids=[env_id], by="reference_velocity", limit=10),
            ctx=ctx,
        )
    finally:
        reset_session_factory(token)

    ids = [item.memory.id for item in resp.items]
    # Only m_recent_pop has any edges within the default 30d window.
    assert ids == [m_recent_pop]
    assert resp.items[0].metric_value == 2.0
    # Velocity is also surfaced on the per-memory response field.
    assert resp.items[0].memory.reference_velocity == 2

    # Widening the window picks up the older popular memory too.
    token = use_session_factory(factory)
    try:
        wider = await memory_top(
            MemTopRequest(
                env_ids=[env_id],
                by="reference_velocity",
                limit=10,
                velocity_window_days=90,
            ),
            ctx=await _ctx_with_env(env_id),
        )
    finally:
        reset_session_factory(token)

    wider_ids = [item.memory.id for item in wider.items]
    assert set(wider_ids) == {m_recent_pop, m_old_pop}
    # Older has 5 edges, recent has 2 — old ranks first.
    assert wider_ids[0] == m_old_pop


@pytest.mark.asyncio
async def test_mem_top_limit_cap_honored(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        for i in range(5):
            await _mk_mem(session, env_id, salience=float(i) / 10.0, title=f"m{i}")
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp = await memory_top(
            MemTopRequest(env_ids=[env_id], by="salience", limit=2),
            ctx=ctx,
        )
    finally:
        reset_session_factory(token)

    assert len(resp.items) == 2
    assert resp.total_examined == 5


# ---------------------------------------------------------------------------
# Phase 1e-e (v0.14.1) — reference_authority metric + AUTHORITY_DISABLED
# ---------------------------------------------------------------------------


from memory_mcp.config import Settings
from memory_mcp.errors import AuthorityDisabledError


def _settings_authority_on(**overrides) -> Settings:
    """Settings preset with the popularity-authority knob ON."""
    return Settings(dream_popularity_authority_weighted=True, **overrides)


async def _set_authority(
    session,
    mem_id: UUID,
    *,
    rel_link: float = 0.0,
    lineage: float = 0.0,
    task: float = 0.0,
    playbook: float = 0.0,
) -> None:
    """Directly stamp the four per-kind ref_authority_* columns.

    The GENERATED total (``reference_authority``) updates automatically.
    Bypasses the recount pass — fixture-only path.
    """
    await session.execute(
        text(
            "UPDATE memories "
            "SET ref_authority_rel_link = :rl, ref_authority_lineage = :ln, "
            "    ref_authority_task = :tk, ref_authority_playbook = :pb "
            "WHERE id = :id"
        ),
        {"id": mem_id, "rl": rel_link, "ln": lineage, "tk": task, "pb": playbook},
    )


@pytest.mark.asyncio
async def test_mem_top_by_reference_authority_ranks_descending(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Knob ON. Two memories with different authority values are ranked
    descending by reference_authority. metric_value matches the GENERATED
    total. Response carries ``memory.reference_authority``.
    """
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        m_high = await _mk_mem(session, env_id, title="high")
        m_low = await _mk_mem(session, env_id, title="low")
        await _set_authority(session, m_high, rel_link=1.2)
        await _set_authority(session, m_low, rel_link=0.4)
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp = await memory_top(
            MemTopRequest(env_ids=[env_id], by="reference_authority", limit=5),
            ctx=ctx,
            settings=_settings_authority_on(),
        )
    finally:
        reset_session_factory(token)

    ids = [item.memory.id for item in resp.items]
    assert ids == [m_high, m_low]
    assert resp.items[0].metric_value == pytest.approx(1.2, abs=1e-6)
    assert resp.items[1].metric_value == pytest.approx(0.4, abs=1e-6)
    assert resp.items[0].memory.reference_authority == pytest.approx(1.2, abs=1e-6)
    assert resp.items[1].memory.reference_authority == pytest.approx(0.4, abs=1e-6)
    assert resp.by == "reference_authority"


@pytest.mark.asyncio
async def test_mem_top_by_reference_authority_knob_off_raises_zero_db(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Knob OFF (default). Tool raises AUTHORITY_DISABLED before any DB
    work — asserted by replacing ``session_scope`` with a tripwire that
    raises if entered.
    """
    factory, _ = postgres_session_factories()
    async with factory() as session:
        env_id = await _mk_env(session)
        m = await _mk_mem(session, env_id, title="m")
        await _set_authority(session, m, rel_link=1.0)
        await session.commit()

    def _tripwire(*_args, **_kwargs):
        raise AssertionError("session_scope must not be entered on AUTHORITY_DISABLED")

    monkeypatch.setattr(top_mod, "session_scope", _tripwire)

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        with pytest.raises(AuthorityDisabledError) as exc:
            await memory_top(
                MemTopRequest(env_ids=[env_id], by="reference_authority", limit=5),
                ctx=ctx,
                # No settings override — falls back to Settings() defaults
                # which carry dream_popularity_authority_weighted=False.
            )
        assert exc.value.code == "AUTHORITY_DISABLED"
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_mem_top_by_reference_authority_excludes_zero_rows(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Knob ON. Zero-authority rows are excluded from ``items`` but
    counted by ``total_examined`` (mirrors ``reference_velocity``
    semantics).
    """
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        m_a = await _mk_mem(session, env_id, title="A")
        m_b = await _mk_mem(session, env_id, title="B")
        m_c = await _mk_mem(session, env_id, title="C")  # zero-authority
        await _set_authority(session, m_a, rel_link=1.5)
        await _set_authority(session, m_b, rel_link=0.3)
        # m_c left at zero
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp = await memory_top(
            MemTopRequest(env_ids=[env_id], by="reference_authority", limit=10),
            ctx=ctx,
            settings=_settings_authority_on(),
        )
    finally:
        reset_session_factory(token)

    item_ids = {item.memory.id for item in resp.items}
    assert m_a in item_ids
    assert m_b in item_ids
    assert m_c not in item_ids
    assert len(resp.items) == 2
    # total_examined counts ALL eligible rows pre-metric-filter
    assert resp.total_examined == 3


@pytest.mark.asyncio
async def test_mem_top_by_reference_authority_default_status_excludes_non_active(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Knob ON. A stale memory with positive authority is filtered
    out by the default ``statuses=[active]`` filter; explicit opt-in
    surfaces it.
    """
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        m_active = await _mk_mem(session, env_id, title="active", status="active")
        m_stale = await _mk_mem(session, env_id, title="stale", status="stale")
        await _set_authority(session, m_active, rel_link=0.5)
        await _set_authority(session, m_stale, rel_link=2.0)  # higher
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        # Default statuses → only active row visible
        resp_default = await memory_top(
            MemTopRequest(env_ids=[env_id], by="reference_authority", limit=10),
            ctx=ctx,
            settings=_settings_authority_on(),
        )
        # Explicit opt-in → stale row included and ranks first
        resp_explicit = await memory_top(
            MemTopRequest(
                env_ids=[env_id],
                by="reference_authority",
                limit=10,
                statuses=[MemoryStatus.active, MemoryStatus.stale],
            ),
            ctx=ctx,
            settings=_settings_authority_on(),
        )
    finally:
        reset_session_factory(token)

    assert [item.memory.id for item in resp_default.items] == [m_active]
    assert [item.memory.id for item in resp_explicit.items] == [m_stale, m_active]


@pytest.mark.asyncio
async def test_mem_top_by_reference_authority_tag_match_all(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Knob ON. ``tag_match='all'`` requires every listed tag — a row
    tagged with only one of the two listed tags is excluded even when
    its authority would otherwise rank it first.
    """
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        m_both = await _mk_mem(session, env_id, title="both")
        m_one = await _mk_mem(session, env_id, title="one")
        await _attach_tag(session, env_id, m_both, "a")
        await _attach_tag(session, env_id, m_both, "b")
        await _attach_tag(session, env_id, m_one, "a")
        # Make the one-tag row carry HIGHER authority so the test
        # actually exercises the tag filter and not just a ranking sort.
        await _set_authority(session, m_both, rel_link=0.5)
        await _set_authority(session, m_one, rel_link=2.0)
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp = await memory_top(
            MemTopRequest(
                env_ids=[env_id],
                by="reference_authority",
                limit=10,
                tags=["a", "b"],
                tag_match="all",
            ),
            ctx=ctx,
            settings=_settings_authority_on(),
        )
    finally:
        reset_session_factory(token)

    assert [item.memory.id for item in resp.items] == [m_both]


@pytest.mark.asyncio
async def test_mem_top_by_reference_authority_limit_truncates(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Knob ON. ``limit=3`` returns top 3 of 5 cited memories.
    ``total_examined`` counts all 5.
    """
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    async with factory() as session:
        env_id = await _mk_env(session)
        for i in range(5):
            m = await _mk_mem(session, env_id, title=f"m{i}")
            await _set_authority(session, m, rel_link=float(i + 1) * 0.1)
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp = await memory_top(
            MemTopRequest(env_ids=[env_id], by="reference_authority", limit=3),
            ctx=ctx,
            settings=_settings_authority_on(),
        )
    finally:
        reset_session_factory(token)

    assert len(resp.items) == 3
    assert resp.total_examined == 5
    # Descending: 0.5, 0.4, 0.3
    values = [item.metric_value for item in resp.items]
    assert values == sorted(values, reverse=True)


@pytest.mark.asyncio
async def test_mem_top_by_reference_authority_stable_tiebreaker(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Knob ON. Two memories with identical authority — order is stable
    across requeries by ``(metric DESC, created_at DESC, id DESC)``.
    """
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()

    now = dt.datetime.now(dt.UTC)
    async with factory() as session:
        env_id = await _mk_env(session)
        # Stagger created_at so tie-breaker can fire deterministically.
        m_old = await _mk_mem(
            session,
            env_id,
            title="old",
            created_at=now - dt.timedelta(hours=1),
        )
        m_new = await _mk_mem(
            session,
            env_id,
            title="new",
            created_at=now,
        )
        await _set_authority(session, m_old, rel_link=0.5)
        await _set_authority(session, m_new, rel_link=0.5)
        await session.commit()

    token = use_session_factory(factory)
    try:
        ctx = await _ctx_with_env(env_id)
        resp_a = await memory_top(
            MemTopRequest(env_ids=[env_id], by="reference_authority", limit=10),
            ctx=ctx,
            settings=_settings_authority_on(),
        )
        resp_b = await memory_top(
            MemTopRequest(env_ids=[env_id], by="reference_authority", limit=10),
            ctx=ctx,
            settings=_settings_authority_on(),
        )
    finally:
        reset_session_factory(token)

    # created_at DESC → newer first
    assert [i.memory.id for i in resp_a.items] == [m_new, m_old]
    # Stable across re-requests
    assert [i.memory.id for i in resp_a.items] == [i.memory.id for i in resp_b.items]
