"""Real-Postgres smoke for ``mem_decompose`` transaction body (Phase 3 C7).

Eight smoke cases that the unit suite cannot exercise without a live
database:

* ``decompose_derive_two_children`` — derive happy path. Source stays
  active, 2 ``derived_from`` lineage rows, source's
  ``reference_count_lineage`` bumped twice (trigger from migration
  0021's whitelist), operation row exists.
* ``decompose_split_two_children`` — split happy path. Source retired
  + version bumped, 2 ``split_from`` lineage rows,
  ``reference_count_lineage`` NOT bumped (whitelist excludes
  ``split_from``), source outbox carries a tombstone.
* ``decompose_replay_basic`` — identical second call returns
  ``idempotency_replay=true`` + same children + same operation id; no
  new audit / outbox rows.
* ``decompose_replay_after_split_retire`` — replay succeeds after the
  source has been retired by the original call (dedupe-before-state-
  validation; C1.5 RD A.2).
* ``decompose_replay_with_stale_source`` — source ``status='stale'``
  succeeds on first call (positive ``_validate_source`` path).
* ``decompose_caller_idempotency_key`` — caller-supplied
  ``idempotency_key`` overrides the derived hash; second call with
  the same key replays even if the content "would have" produced a
  different hash.
* ``decompose_caller_key_fingerprint_mismatch`` — same
  ``idempotency_key`` but a different ``mode`` /
  child-set / ``source_id`` → ``InvalidInputError`` rather than a
  misleading replay.
* ``decompose_playbook_child_rejected`` — schema-layer 422 (envelope
  validation; avoids the B3d-style latent bug class where smoke never
  exercises validator paths).

The remaining ~12 matrix cases (concurrent-race, whitelist regression,
audit-row counts, lineage-enum CHECK, replay-stale-children-view,
RBAC-invisible-source, expected_version mismatch, etc.) live in C9.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from memory_mcp import composers as composers_mod
from memory_mcp import decomposers as decomposers_mod
from memory_mcp import memories as memories_mod
from memory_mcp.config import Settings
from memory_mcp.db.models import (
    Agent,
    AuditLog,
    DecomposeOperation,
    Environment,
    Memory,
    MemoryLineage,
    Outbox,
)
from memory_mcp.db.types import (
    LineageRelation,
    MemoryKind,
    MemoryStatus,
    OutboxOp,
)
from memory_mcp.errors import (
    InvalidInputError,
    InvalidTransitionError,
    NotFoundError,
    VersionConflictError,
)
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryWriteRequest, memory_write
from memory_mcp_schemas.decompose import MemDecomposeChild, MemDecomposeRequest

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


def _settings() -> Settings:
    return Settings(graph_backend="postgres")


def _patch_session_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route session_scope() through the test-controlled factory."""
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(composers_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(decomposers_mod, "session_scope", routed_session_scope)


async def _setup_env_and_agent(factory) -> tuple[UUID, UUID]:
    async with factory() as session:
        env = Environment(
            name=f"decompose-smoke-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"decompose-smoke-agent-{uuid4()}")
        session.add_all([env, agent])
        await session.commit()
        return env.id, agent.id


async def _write_source(
    factory,
    *,
    env_id: UUID,
    agent_id: UUID,
    title: str,
    body: str,
    kind: MemoryKind = MemoryKind.fact,
    tags: list[str] | None = None,
) -> UUID:
    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        resp = await memory_write(
            MemoryWriteRequest(
                kind=kind,
                title=title,
                body=body,
                env_id=env_id,
                tags=tags or [],
            ),
            ctx=ctx,
            settings=_settings(),
        )
        return resp.id
    finally:
        reset_session_factory(token)


async def _fetch_memory(factory, memory_id: UUID) -> Memory:
    async with factory() as session:
        return (await session.execute(
            select(Memory).where(Memory.id == memory_id)
        )).scalar_one()


async def _fetch_lineage(factory, child_id: UUID) -> list[tuple[UUID, str]]:
    async with factory() as session:
        rows = (await session.execute(
            select(MemoryLineage.parent_memory_id, MemoryLineage.relation)
            .where(MemoryLineage.child_memory_id == child_id)
        )).all()
        return [(r[0], r[1]) for r in rows]


async def _fetch_operation(factory, operation_id: UUID) -> DecomposeOperation | None:
    async with factory() as session:
        return (await session.execute(
            select(DecomposeOperation).where(DecomposeOperation.id == operation_id)
        )).scalar_one_or_none()


async def _count_audits(factory, memory_id: UUID, op_like: str | None = None) -> int:
    async with factory() as session:
        from sqlalchemy import func
        stmt = select(func.count()).select_from(AuditLog).where(
            AuditLog.record_id == memory_id
        )
        if op_like is not None:
            stmt = stmt.where(AuditLog.op.like(op_like))
        return int((await session.execute(stmt)).scalar_one())


async def _count_outbox(
    factory, memory_id: UUID, op: OutboxOp | None = None
) -> int:
    async with factory() as session:
        from sqlalchemy import func
        stmt = select(func.count()).select_from(Outbox).where(
            Outbox.aggregate_id == memory_id
        )
        if op is not None:
            stmt = stmt.where(Outbox.op == op.value)
        return int((await session.execute(stmt)).scalar_one())


def _child(title: str, body: str, kind: MemoryKind = MemoryKind.fact) -> MemDecomposeChild:
    return MemDecomposeChild(kind=kind, title=title, body=body)


# ---------------------------------------------------------------------------
# Baseline smoke cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_derive_two_children(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Derive mode: source stays active, lineage uses derived_from."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="origin source body",
    )

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=[_child("c1", "first child"), _child("c2", "second child")],
                mode="derive",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert resp.mode == "derive"
    assert resp.idempotency_replay is False
    assert resp.dedupe_key
    assert len(resp.children) == 2
    assert [c.title for c in resp.children] == ["c1", "c2"]
    assert {r.relation for r in resp.lineage_rows} == {"derived_from"}
    assert resp.source.id == src_id
    assert resp.source.status == MemoryStatus.active.value

    src = await _fetch_memory(factory, src_id)
    assert src.status == MemoryStatus.active.value
    # derived_from is whitelisted by migration 0021 — counter bumps by N.
    assert int(src.reference_count_lineage or 0) == 2

    op = await _fetch_operation(factory, resp.operation_id)
    assert op is not None
    assert op.mode == "derive"
    assert op.source_id == src_id
    assert list(op.child_ids) == [c.id for c in resp.children]


@pytest.mark.asyncio
async def test_decompose_split_two_children(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Split mode: source retired, lineage split_from, NO counter bump."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id,
        title="src", body="will retire",
    )
    src_before = await _fetch_memory(factory, src_id)
    initial_version = src_before.version

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=[_child("a", "alpha"), _child("b", "beta")],
                mode="split",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert resp.mode == "split"
    assert {r.relation for r in resp.lineage_rows} == {"split_from"}

    src_after = await _fetch_memory(factory, src_id)
    assert src_after.status == MemoryStatus.retired.value
    assert src_after.version == initial_version + 1
    # split_from is EXCLUDED from the popularity whitelist (E.11).
    assert int(src_after.reference_count_lineage or 0) == 0

    # Source tombstone outbox event (mode=split causes Qdrant drop).
    n_tombstone = await _count_outbox(factory, src_id, op=OutboxOp.tombstone)
    assert n_tombstone == 1
    # Each child gets one upsert event.
    for child in resp.children:
        n_upsert = await _count_outbox(factory, child.id, op=OutboxOp.upsert)
        assert n_upsert == 1


@pytest.mark.asyncio
async def test_decompose_replay_basic(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Identical second call returns idempotency_replay=true and same operation id."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="b",
    )

    req = MemDecomposeRequest(
        source_id=src_id,
        children=[_child("a", "alpha"), _child("b", "beta")],
        mode="derive",
    )

    token = use_session_factory(factory)
    try:
        first = await decomposers_mod.memory_decompose(
            req, ctx=ctx, settings=_settings(),
        )
        second = await decomposers_mod.memory_decompose(
            req, ctx=ctx, settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert first.idempotency_replay is False
    assert second.idempotency_replay is True
    assert second.operation_id == first.operation_id
    assert second.dedupe_key == first.dedupe_key
    assert [c.id for c in second.children] == [c.id for c in first.children]

    # Lineage rows were not re-inserted.
    for child in first.children:
        lineage = await _fetch_lineage(factory, child.id)
        assert len(lineage) == 1
        assert lineage[0][0] == src_id
        assert lineage[0][1] == LineageRelation.derived_from.value

    # No additional audit rows on the children from the second call.
    for child in first.children:
        n_create = await _count_audits(factory, child.id, op_like="create")
        assert n_create == 1


@pytest.mark.asyncio
async def test_decompose_replay_after_split_retire(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Replay still works after source has been retired by the original call."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="b",
    )

    req = MemDecomposeRequest(
        source_id=src_id,
        children=[_child("x", "x"), _child("y", "y")],
        mode="split",
    )

    token = use_session_factory(factory)
    try:
        first = await decomposers_mod.memory_decompose(
            req, ctx=ctx, settings=_settings(),
        )
        # Source is now retired; a retry must still replay.
        second = await decomposers_mod.memory_decompose(
            req, ctx=ctx, settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert first.idempotency_replay is False
    assert second.idempotency_replay is True
    assert second.operation_id == first.operation_id
    assert second.source.status == MemoryStatus.retired.value


@pytest.mark.asyncio
async def test_decompose_replay_with_stale_source(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Source.status='stale' is a valid first-call target (positive _validate_source path)."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="b",
    )
    # Flip the source to stale via raw UPDATE (no mem_supersede coupling).
    async with factory() as session:
        from sqlalchemy import update as sa_update
        await session.execute(
            sa_update(Memory)
            .where(Memory.id == src_id)
            .values(status=MemoryStatus.stale.value)
        )
        await session.commit()

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=[_child("p", "p"), _child("q", "q")],
                mode="derive",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert resp.idempotency_replay is False
    assert len(resp.children) == 2

    # Source stays stale (derive mode does not transition).
    src = await _fetch_memory(factory, src_id)
    assert src.status == MemoryStatus.stale.value


@pytest.mark.asyncio
async def test_decompose_caller_idempotency_key(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Caller-supplied idempotency_key overrides the derived hash."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="b",
    )

    children = [_child("a", "alpha"), _child("b", "beta")]
    token = use_session_factory(factory)
    try:
        with_key = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=children,
                mode="derive",
                idempotency_key="caller-decompose-001",
            ),
            ctx=ctx,
            settings=_settings(),
        )
        # Same content + same caller key → replay.
        replay = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=children,
                mode="derive",
                idempotency_key="caller-decompose-001",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert with_key.dedupe_key == "caller-decompose-001"
    assert replay.idempotency_replay is True
    assert replay.operation_id == with_key.operation_id


@pytest.mark.asyncio
async def test_decompose_caller_key_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Same idempotency_key with a different mode → fingerprint mismatch → reject."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="b",
    )

    children = [_child("a", "alpha"), _child("b", "beta")]
    token = use_session_factory(factory)
    try:
        await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=children,
                mode="derive",
                idempotency_key="reuse-key-002",
            ),
            ctx=ctx,
            settings=_settings(),
        )
        # Same key, but mode=split → fingerprint differs → reject.
        with pytest.raises(InvalidInputError, match="different scope"):
            await decomposers_mod.memory_decompose(
                MemDecomposeRequest(
                    source_id=src_id,
                    children=children,
                    mode="split",
                    idempotency_key="reuse-key-002",
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_decompose_playbook_child_rejected() -> None:
    """kind=playbook child rejected at the Pydantic boundary (no DB round-trip)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="playbook"):
        MemDecomposeRequest(
            source_id=uuid4(),
            children=[
                _child("a", "alpha"),
                MemDecomposeChild(kind=MemoryKind.playbook, title="bad", body="b"),
            ],
            mode="derive",
        )


# ---------------------------------------------------------------------------
# C9 — Matrix tests (validation, RBAC, version, race, whitelist, audit shape)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_too_few_children() -> None:
    """min_length=2 — single child rejected at the schema layer."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="at least 2 items"):
        MemDecomposeRequest(
            source_id=uuid4(),
            children=[_child("solo", "alpha")],
            mode="derive",
        )


@pytest.mark.asyncio
async def test_too_many_children() -> None:
    """max_length=20 — 21st child rejected at the schema layer."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="at most 20 items"):
        MemDecomposeRequest(
            source_id=uuid4(),
            children=[_child(f"c{i}", f"body{i}") for i in range(21)],
            mode="derive",
        )


@pytest.mark.asyncio
async def test_duplicate_child_canonical_hash(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Two children with identical canonical-JSON content rejected pre-lock."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )

    token = use_session_factory(factory)
    try:
        with pytest.raises(InvalidInputError, match="duplicate child content"):
            await decomposers_mod.memory_decompose(
                MemDecomposeRequest(
                    source_id=src_id,
                    children=[
                        _child("dup", "same body"),
                        _child("dup", "same body"),
                    ],
                    mode="derive",
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_decision_meta_on_non_decision_child(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """decision_meta on kind=fact rejected at _validate_children."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )

    bad_child = MemDecomposeChild(
        kind=MemoryKind.fact,
        title="bad",
        body="b",
        decision_meta={
            "status": "accepted",
            "rationale": "anything",
            "constraints": ["c"],
            "consequences": ["k"],
            "superseded_by": None,
        },
    )

    token = use_session_factory(factory)
    try:
        with pytest.raises(InvalidInputError, match="decision_meta only valid for kind=decision"):
            await decomposers_mod.memory_decompose(
                MemDecomposeRequest(
                    source_id=src_id,
                    children=[bad_child, _child("ok", "good")],
                    mode="derive",
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_decision_meta_on_decision_child_accepted(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """decision_meta on kind=decision passes _validate_children + deep validation."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )

    good_child = MemDecomposeChild(
        kind=MemoryKind.decision,
        title="adr",
        body="we chose X",
        decision_meta={
            "status": "accepted",
            "rationale": "X is the simplest path",
            "constraints": ["budget"],
            "consequences": ["faster"],
            "superseded_by": None,
        },
    )

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=[good_child, _child("fact", "supporting fact")],
                mode="derive",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert len(resp.children) == 2
    assert resp.children[0].kind == "decision"


@pytest.mark.asyncio
async def test_source_not_in_attached_env(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Source visible only in env X; caller has env Y attached → NotFoundError."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )
    # New env the caller has access to instead.
    async with factory() as session:
        other_env = Environment(
            name=f"decompose-other-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        session.add(other_env)
        await session.commit()
        other_env_id = other_env.id

    ctx_wrong = AgentContext(agent_id=agent_id, attached_env_ids=[other_env_id])

    token = use_session_factory(factory)
    try:
        with pytest.raises(NotFoundError):
            await decomposers_mod.memory_decompose(
                MemDecomposeRequest(
                    source_id=src_id,
                    children=[_child("a", "alpha"), _child("b", "beta")],
                    mode="derive",
                ),
                ctx=ctx_wrong,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_source_retired_rejected(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Retired source cannot be decomposed on first write (InvalidTransitionError)."""
    from sqlalchemy import update as sa_update

    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )

    async with factory() as session:
        await session.execute(
            sa_update(Memory)
            .where(Memory.id == src_id)
            .values(status=MemoryStatus.retired.value)
        )
        await session.commit()

    token = use_session_factory(factory)
    try:
        with pytest.raises(InvalidTransitionError) as exc_info:
            await decomposers_mod.memory_decompose(
                MemDecomposeRequest(
                    source_id=src_id,
                    children=[_child("a", "alpha"), _child("b", "beta")],
                    mode="derive",
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)
    assert exc_info.value.src == "retired"
    assert exc_info.value.dst == "decomposed"


@pytest.mark.asyncio
async def test_expected_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Wrong expected_version on source → VersionConflictError."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )

    token = use_session_factory(factory)
    try:
        with pytest.raises(VersionConflictError) as exc_info:
            await decomposers_mod.memory_decompose(
                MemDecomposeRequest(
                    source_id=src_id,
                    children=[_child("a", "alpha"), _child("b", "beta")],
                    mode="derive",
                    expected_version=99,
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)
    assert exc_info.value.expected == 99
    assert exc_info.value.actual == 1


@pytest.mark.asyncio
async def test_mixed_kind_children_split_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Mixed-kind children (fact + procedure) allowed in both modes (D.5 confirmed)."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id,
        title="proc", body="multi-step procedure body",
        kind=MemoryKind.procedure,
    )

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=[
                    _child("step", "a procedural fragment", kind=MemoryKind.procedure),
                    _child("fact", "an atomic fact", kind=MemoryKind.fact),
                ],
                mode="split",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    assert resp.mode == "split"
    kinds = sorted(c.kind for c in resp.children)
    assert kinds == ["fact", "procedure"]


@pytest.mark.asyncio
async def test_concurrent_identical_decompose_race(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Two concurrent identical decomposes → one mutation, one replay.

    Source FOR UPDATE lock serializes them; the loser sees the persisted
    dedupe_key in decompose_operations and returns via the replay path.
    Validates the C6 race-loss arbiter end-to-end.
    """
    import asyncio

    _patch_session_scope(monkeypatch)
    factory_1, factory_2 = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory_1)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    src_id = await _write_source(
        factory_1, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )

    request = MemDecomposeRequest(
        source_id=src_id,
        children=[_child("a", "alpha"), _child("b", "beta")],
        mode="derive",
    )

    async def call_decompose(factory):
        token = use_session_factory(factory)
        try:
            return await decomposers_mod.memory_decompose(
                request, ctx=ctx, settings=_settings(),
            )
        finally:
            reset_session_factory(token)

    results = await asyncio.gather(
        call_decompose(factory_1),
        call_decompose(factory_2),
        return_exceptions=True,
    )

    assert all(not isinstance(r, Exception) for r in results), (
        f"unexpected exceptions: {results}"
    )
    assert results[0].operation_id == results[1].operation_id
    replay_flags = sorted(r.idempotency_replay for r in results)
    assert replay_flags == [False, True]

    async with factory_1() as session:
        from sqlalchemy import func
        n = int((await session.execute(
            select(func.count()).select_from(DecomposeOperation).where(
                DecomposeOperation.source_id == src_id,
            )
        )).scalar_one())
        assert n == 1


@pytest.mark.asyncio
async def test_split_lineage_does_not_bump_reference_count_lineage(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """E.11 regression: split_from is excluded from popularity whitelist.

    Insert a raw ``split_from`` MemoryLineage row; parent's
    ``reference_count_lineage`` stays unchanged. Then insert a
    ``derived_from`` row; counter increments by 1. Locks in the
    trigger-whitelist semantic shipped by migration 0021.
    """
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )
    c1_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="c1", body="child1"
    )
    c2_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="c2", body="child2"
    )

    async with factory() as session:
        session.add(MemoryLineage(
            parent_memory_id=src_id,
            child_memory_id=c1_id,
            relation=LineageRelation.split_from.value,
        ))
        await session.commit()
    src_after_split = await _fetch_memory(factory, src_id)
    assert int(src_after_split.reference_count_lineage or 0) == 0

    async with factory() as session:
        session.add(MemoryLineage(
            parent_memory_id=src_id,
            child_memory_id=c2_id,
            relation=LineageRelation.derived_from.value,
        ))
        await session.commit()
    src_after_derive = await _fetch_memory(factory, src_id)
    assert int(src_after_derive.reference_count_lineage or 0) == 1


@pytest.mark.asyncio
async def test_audit_log_shape_split(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Split: per-child create + 1 mem_decompose:split + 1 retire on source."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=[_child("a", "alpha"), _child("b", "beta")],
                mode="split",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    for child in resp.children:
        n_create = await _count_audits(factory, child.id, op_like="create")
        assert n_create == 1

    n_decompose = await _count_audits(factory, src_id, op_like="mem_decompose:split")
    assert n_decompose == 1
    n_retire = await _count_audits(factory, src_id, op_like="retire")
    assert n_retire == 1


@pytest.mark.asyncio
async def test_audit_log_shape_derive(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Derive: per-child create + 1 mem_decompose:derive on source; NO retire."""
    _patch_session_scope(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    src_id = await _write_source(
        factory, env_id=env_id, agent_id=agent_id, title="s", body="origin"
    )

    token = use_session_factory(factory)
    try:
        resp = await decomposers_mod.memory_decompose(
            MemDecomposeRequest(
                source_id=src_id,
                children=[_child("a", "alpha"), _child("b", "beta")],
                mode="derive",
            ),
            ctx=ctx,
            settings=_settings(),
        )
    finally:
        reset_session_factory(token)

    for child in resp.children:
        n_create = await _count_audits(factory, child.id, op_like="create")
        assert n_create == 1

    n_decompose = await _count_audits(factory, src_id, op_like="mem_decompose:derive")
    assert n_decompose == 1
    n_retire = await _count_audits(factory, src_id, op_like="retire")
    assert n_retire == 0
