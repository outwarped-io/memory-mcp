"""v0.17 cross-cutting ``expires_at`` filter coverage.

Phase B threading is complete across nine read-path modules. These tests
exercise the contract end-to-end against a real Postgres testcontainer:

Plumbed surfaces (request flag opt-in / opt-out):
* ``mem_search`` (lex mode) — default hides expired; ``include_expired=True`` shows.
* ``mem_browse`` — same.
* ``mem_facets`` — counts (``total`` and ``by_env``) exclude expired by default.
* ``mem_top`` — default hides expired; ``include_expired=True`` shows.

Convenience surfaces (always filter; no opt-out):
* ``mem_resume`` (via ``resume_for_env``) — ``summary_stats.memory_count``
  excludes expired; ``recent_journal`` excludes expired observations.
* ``mem_context_pack`` (via ``pack``) — ``decisions`` section excludes
  expired decision memories.

Cursor invariant:
* Browse cursor fingerprint includes ``include_expired``; mid-pagination
  flag flip raises ``InvalidCursorError``.

``memory_auto_context`` is NOT covered here — it requires Qdrant, which
the integration harness does not provision. Its filter is exercised by
the unit schema test (``test_expires_at_filter_schemas.py``).
"""

from __future__ import annotations

import datetime as dt
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.browse import MemBrowseRequest, MemFacetsRequest
from memory_mcp_schemas.enums import DecisionStatus, MemoryKind
from memory_mcp_schemas.search import MemorySearchRequest
from memory_mcp_schemas.top import MemTopRequest

from memory_mcp import browse as browse_mod
from memory_mcp import entities as entities_mod
from memory_mcp import envs as envs_mod
from memory_mcp import memories as memories_mod
from memory_mcp import top as top_mod
from memory_mcp.config import Settings
from memory_mcp.context_pack import api as context_pack_mod
from memory_mcp.db.models import Agent, Environment
from memory_mcp.digest import api as digest_mod
from memory_mcp.errors import InvalidCursorError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryWriteRequest, memory_write
from memory_mcp.search import api as search_api_mod

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


def _settings() -> Settings:
    return Settings(graph_backend="postgres")


def _patch_session_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route all touched modules' ``session_scope`` through the test factory.

    Skips ``auto_context`` because it requires Qdrant; its filter is
    covered by the schema unit test.
    """
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(entities_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(envs_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(browse_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(top_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(search_api_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(digest_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(context_pack_mod, "session_scope", routed_session_scope)


async def _setup_env_and_agent(factory) -> tuple[UUID, UUID, str]:
    """Seed an env + agent. Returns ``(env_id, agent_id, env_name)``."""
    name = f"expires-test-{uuid4().hex[:8]}"
    async with factory() as session:
        env = Environment(
            name=name,
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"expires-agent-{uuid4().hex[:8]}")
        session.add_all([env, agent])
        await session.commit()
        return env.id, agent.id, env.name


def _past() -> dt.datetime:
    return dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)


def _future() -> dt.datetime:
    return dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)


async def _plant_pair(
    *,
    env_id: UUID,
    ctx: AgentContext,
    kind: MemoryKind = MemoryKind.fact,
    tag: str | None = None,
    decision_meta: dict | None = None,
):
    """Plant two memories (one fresh, one expired) and return ``(fresh, expired)``."""
    tags = [tag] if tag else []
    fresh = await memory_write(
        MemoryWriteRequest(
            kind=kind,
            title="fresh needle",
            body="fresh needle body content",
            env_id=env_id,
            tags=tags,
            expires_at=_future(),
            decision_meta=decision_meta,
        ),
        ctx=ctx,
        settings=_settings(),
    )
    expired = await memory_write(
        MemoryWriteRequest(
            kind=kind,
            title="stale needle",
            body="stale needle body content",
            env_id=env_id,
            tags=tags,
            expires_at=_past(),
            decision_meta=decision_meta,
        ),
        ctx=ctx,
        settings=_settings(),
    )
    return fresh, expired


# ---------------------------------------------------------------------------
# mem_search (plumbed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_lex_default_hides_expired(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, _ = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    token = use_session_factory(factory)
    try:
        fresh, expired = await _plant_pair(env_id=env_id, ctx=ctx)

        # Default: include_expired=False.
        from memory_mcp.search.api import memory_search

        default_resp = await memory_search(
            MemorySearchRequest(
                query="needle",
                env_ids=[env_id],
                mode="lex",
                fallback=False,
                limit=50,
            ),
            ctx=ctx,
            settings=_settings(),
        )
        ids_default = {hit.memory.id for hit in default_resp.hits}
        assert fresh.id in ids_default
        assert expired.id not in ids_default

        # Opt-in: include_expired=True.
        all_resp = await memory_search(
            MemorySearchRequest(
                query="needle",
                env_ids=[env_id],
                mode="lex",
                fallback=False,
                limit=50,
                include_expired=True,
            ),
            ctx=ctx,
            settings=_settings(),
        )
        ids_all = {hit.memory.id for hit in all_resp.hits}
        assert fresh.id in ids_all
        assert expired.id in ids_all
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# mem_browse (plumbed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browse_default_hides_expired(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, _ = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    token = use_session_factory(factory)
    try:
        fresh, expired = await _plant_pair(env_id=env_id, ctx=ctx)

        from memory_mcp.browse import memory_browse

        default_resp = await memory_browse(
            MemBrowseRequest(env_ids=[env_id], limit=50),
            ctx=ctx,
            settings=_settings(),
        )
        ids_default = {hit.id for hit in default_resp.hits}
        assert fresh.id in ids_default
        assert expired.id not in ids_default

        all_resp = await memory_browse(
            MemBrowseRequest(env_ids=[env_id], limit=50, include_expired=True),
            ctx=ctx,
            settings=_settings(),
        )
        ids_all = {hit.id for hit in all_resp.hits}
        assert fresh.id in ids_all
        assert expired.id in ids_all
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# mem_facets (plumbed) — counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facets_total_excludes_expired_by_default(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, _ = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    token = use_session_factory(factory)
    try:
        await _plant_pair(env_id=env_id, ctx=ctx)

        from memory_mcp.browse import memory_facets

        default_resp = await memory_facets(
            MemFacetsRequest(env_ids=[env_id]),
            ctx=ctx,
            settings=_settings(),
        )
        assert default_resp.total == 1
        assert default_resp.by_env.get(env_id) == 1

        all_resp = await memory_facets(
            MemFacetsRequest(env_ids=[env_id], include_expired=True),
            ctx=ctx,
            settings=_settings(),
        )
        assert all_resp.total == 2
        assert all_resp.by_env.get(env_id) == 2
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# mem_top (plumbed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_default_hides_expired(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, _ = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    token = use_session_factory(factory)
    try:
        fresh, expired = await _plant_pair(env_id=env_id, ctx=ctx)

        from memory_mcp.top import memory_top

        default_resp = await memory_top(
            MemTopRequest(env_ids=[env_id], by="salience", limit=50),
            ctx=ctx,
            settings=_settings(),
        )
        ids_default = {item.memory.id for item in default_resp.items}
        assert fresh.id in ids_default
        assert expired.id not in ids_default

        all_resp = await memory_top(
            MemTopRequest(
                env_ids=[env_id],
                by="salience",
                limit=50,
                include_expired=True,
            ),
            ctx=ctx,
            settings=_settings(),
        )
        ids_all = {item.memory.id for item in all_resp.items}
        assert fresh.id in ids_all
        assert expired.id in ids_all
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# mem_resume (convenience — always filters)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_summary_stats_excludes_expired(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``ResumeStats.memory_count`` counts only non-expired memories.

    Convenience surface: no opt-out. The filter is unconditional.
    """
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, _ = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    token = use_session_factory(factory)
    try:
        await _plant_pair(env_id=env_id, ctx=ctx)

        from memory_mcp.digest.api import resume_for_env

        resp = await resume_for_env(env_id, ctx=ctx)
        assert resp.summary_stats.memory_count == 1
    finally:
        reset_session_factory(token)


@pytest.mark.asyncio
async def test_resume_recent_journal_excludes_expired_observations(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Expired observations / journal_entry rows are absent from ``recent_journal``."""
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, _ = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    token = use_session_factory(factory)
    try:
        fresh, expired = await _plant_pair(
            env_id=env_id,
            ctx=ctx,
            kind=MemoryKind.observation,
        )

        from memory_mcp.digest.api import resume_for_env

        resp = await resume_for_env(env_id, ctx=ctx, journal_tail=50)
        ids = {entry.id for entry in resp.recent_journal}
        assert fresh.id in ids
        assert expired.id not in ids
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# mem_context_pack (convenience — always filters; decisions section)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_pack_decisions_section_excludes_expired(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Expired ``kind=decision`` memories don't appear in the ``decisions`` section.

    Plants two accepted decisions (one fresh, one expired) and asserts the
    expired one is absent. ``include_journal=False`` avoids unrelated noise.
    """
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, _ = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    token = use_session_factory(factory)
    try:
        decision_meta = {
            "status": DecisionStatus.accepted.value,
            "rationale": "test decision",
            "constraints": ["c1"],
            "consequences": None,
            "superseded_by": None,
        }
        fresh, expired = await _plant_pair(
            env_id=env_id,
            ctx=ctx,
            kind=MemoryKind.decision,
            decision_meta=decision_meta,
        )

        from memory_mcp.context_pack.api import pack

        resp = await pack(
            task_desc="any task description",
            env_id=env_id,
            token_budget=4000,
            include_journal=False,
            ctx=ctx,
        )
        decisions_section = next(
            (s for s in resp.sections if s.name == "decisions"),
            None,
        )
        assert decisions_section is not None, "decisions section missing"
        ids = {item.memory_id for item in decisions_section.items}
        assert fresh.id in ids
        assert expired.id not in ids
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# Browse cursor fingerprint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browse_cursor_mismatch_when_include_expired_flips(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Cursor encodes ``include_expired``; mid-pagination flip → ``InvalidCursorError``.

    Plants enough memories that page 1 returns a cursor, then attempts
    page 2 with the flag flipped — the fingerprint must reject.
    """
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, _ = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])

    token = use_session_factory(factory)
    try:
        # Plant 4 fresh memories so a limit=2 page emits a cursor.
        for i in range(4):
            await memory_write(
                MemoryWriteRequest(
                    kind=MemoryKind.fact,
                    title=f"fresh-{i}",
                    body=f"body-{i}",
                    env_id=env_id,
                    expires_at=_future(),
                ),
                ctx=ctx,
                settings=_settings(),
            )

        from memory_mcp.browse import memory_browse

        page1 = await memory_browse(
            MemBrowseRequest(env_ids=[env_id], limit=2),
            ctx=ctx,
            settings=_settings(),
        )
        assert page1.next_cursor, "expected a cursor for page 2"

        with pytest.raises(InvalidCursorError):
            await memory_browse(
                MemBrowseRequest(
                    env_ids=[env_id],
                    limit=2,
                    cursor=page1.next_cursor,
                    include_expired=True,
                ),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)
