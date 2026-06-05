"""Real-Postgres integration tests for the v0.17 inbox tools (Phase C-4).

Six cases cover the five UC invariants from the v0.17 design plan plus
a regression for the await-bug we patched in commit ``f77bf6e``:

1. ``reference_round_trip`` — open → send → list, with the server's
   ``mem-inbox://<env>/<slug>`` reference passed back verbatim each
   call. Confirms the canonical format and that subsequent tools accept
   their own predecessor's response unchanged.
2. ``env_mismatch_rejection`` — ``mem_inbox_send(to="mem-inbox://A/foo",
   env_id=<B-uuid>)`` raises ``InvalidInputError("INBOX_ENV_MISMATCH")``
   (UC2 invariant — no silent cross-env writes).
3. ``no_auto_create_on_send`` — ``mem_inbox_send`` to a never-opened
   slug raises ``InvalidInputError("INBOX_CHANNEL_NOT_FOUND")`` pointing
   the caller to ``mem_inbox_open``.
4. ``expired_default_hidden`` — message with ``expires_at`` in the past
   is excluded from ``mem_inbox(include_expired=False)`` and included
   when ``include_expired=True``. Test plants the fixture via
   ``memory_write`` directly (bypasses ``mem_inbox_send``'s
   ``INBOX_TTL_IN_PAST`` validator).
5. ``server_formatted_references_only`` — every reference in tool
   responses originates from the server (``open.reference`` ==
   ``send.reference`` == ``mem_inbox.reference`` == canonical format).
6. ``await_bug_regression_env_name`` — calls all three tools with
   ``env_name=`` (string) instead of ``env_id`` to confirm the
   ``await _resolve_env_refs`` fix (commit ``f77bf6e``) still routes
   the request through env-name resolution correctly.

Each test uses a unique channel slug derived from ``uuid4().hex[:8]`` so
runs don't leak channel entities across the suite — ``clean_db`` does
not TRUNCATE the ``entities`` table.
"""

from __future__ import annotations

import datetime as dt
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.enums import MemoryKind
from memory_mcp_schemas.inbox import (
    MemInboxOpenRequest,
    MemInboxRequest,
    MemInboxSendRequest,
)

from memory_mcp import entities as entities_mod
from memory_mcp import envs as envs_mod
from memory_mcp import inbox as inbox_mod
from memory_mcp import memories as memories_mod
from memory_mcp.config import Settings
from memory_mcp.db.models import Agent, Environment
from memory_mcp.errors import InvalidInputError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryWriteRequest, memory_write

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


def _settings() -> Settings:
    return Settings(graph_backend="postgres")


def _patch_session_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every module's ``session_scope`` import through the test's
    session factory. Mirrors ``test_compose_transaction.py`` pattern but
    covers the extra modules inbox tools reach into.
    """
    monkeypatch.setattr(inbox_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(entities_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(envs_mod, "session_scope", routed_session_scope)


async def _setup_env_and_agent(factory, env_name: str | None = None) -> tuple[UUID, UUID, str]:
    """Seed an env + agent. Returns ``(env_id, agent_id, env_name)``."""
    name = env_name or f"inbox-test-{uuid4().hex[:8]}"
    async with factory() as session:
        env = Environment(
            name=name,
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"inbox-agent-{uuid4().hex[:8]}")
        session.add_all([env, agent])
        await session.commit()
        return env.id, agent.id, env.name


def _unique_slug() -> str:
    """Generate a unique slug per test so the (entities) table doesn't
    leak across cases — ``clean_db`` only TRUNCATEs memory-related
    tables, not ``entities``.
    """
    return f"test-{uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# UC invariant 1 — reference round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """open → send → list, with the server's reference passed verbatim
    through every subsequent tool call.
    """
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, env_name = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    slug = _unique_slug()

    token = use_session_factory(factory)
    try:
        open_resp = await inbox_mod.mem_inbox_open(
            MemInboxOpenRequest(env_id=env_id, name=slug),
            ctx=ctx,
            settings=_settings(),
        )
        assert open_resp.created is True
        assert open_resp.reference == f"mem-inbox://{env_name}/{slug}"
        assert open_resp.canonical_name == slug
        assert open_resp.env_id == env_id

        # Send using the server's reference verbatim — no string mangling.
        send_resp = await inbox_mod.mem_inbox_send(
            MemInboxSendRequest(
                to=open_resp.reference,
                body="hello, recipient",
                title="first message",
            ),
            ctx=ctx,
            settings=_settings(),
        )
        assert send_resp.reference == open_resp.reference
        assert send_resp.recipient_entity_id == open_resp.entity_id

        # List using the same reference.
        list_resp = await inbox_mod.mem_inbox(
            MemInboxRequest(to=open_resp.reference),
            ctx=ctx,
            settings=_settings(),
        )
        assert list_resp.reference == open_resp.reference
        assert list_resp.count == 1
        assert list_resp.items[0].id == send_resp.id
        assert list_resp.items[0].body == "hello, recipient"
        assert list_resp.items[0].title == "first message"
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# UC invariant 2 — env mismatch rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_mismatch_rejection(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``mem_inbox_send`` with a URL ``to`` pointing to env A and an
    explicit ``env_id`` for env B must raise ``INBOX_ENV_MISMATCH``.
    """
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_a_id, agent_id, env_a_name = await _setup_env_and_agent(factory)
    env_b_id, _, env_b_name = await _setup_env_and_agent(factory)
    # Agent attached to BOTH envs so RBAC doesn't preempt the mismatch check.
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_a_id, env_b_id])
    slug = _unique_slug()

    token = use_session_factory(factory)
    try:
        # Open in env A.
        open_resp = await inbox_mod.mem_inbox_open(
            MemInboxOpenRequest(env_id=env_a_id, name=slug),
            ctx=ctx,
            settings=_settings(),
        )
        ref_a = open_resp.reference
        assert ref_a == f"mem-inbox://{env_a_name}/{slug}"

        # Send with URL=A but env_id=B → mismatch.
        with pytest.raises(InvalidInputError, match="INBOX_ENV_MISMATCH"):
            await inbox_mod.mem_inbox_send(
                MemInboxSendRequest(
                    to=ref_a,
                    body="should not land",
                    env_id=env_b_id,
                ),
                ctx=ctx,
                settings=_settings(),
            )

        # mem_inbox raises the same way.
        with pytest.raises(InvalidInputError, match="INBOX_ENV_MISMATCH"):
            await inbox_mod.mem_inbox(
                MemInboxRequest(to=ref_a, env_id=env_b_id),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# UC invariant 3 — no auto-create on send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_auto_create_on_send(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``mem_inbox_send`` to a never-opened slug raises
    ``INBOX_CHANNEL_NOT_FOUND`` and does NOT create the channel.
    """
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, env_name = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    slug = _unique_slug()
    ref = f"mem-inbox://{env_name}/{slug}"

    token = use_session_factory(factory)
    try:
        with pytest.raises(InvalidInputError, match="INBOX_CHANNEL_NOT_FOUND"):
            await inbox_mod.mem_inbox_send(
                MemInboxSendRequest(to=ref, body="should fail"),
                ctx=ctx,
                settings=_settings(),
            )

        # And mem_inbox raises the same way for symmetry.
        with pytest.raises(InvalidInputError, match="INBOX_CHANNEL_NOT_FOUND"):
            await inbox_mod.mem_inbox(
                MemInboxRequest(to=ref),
                ctx=ctx,
                settings=_settings(),
            )
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# UC invariant 4 — expired default hidden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_default_hidden(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """A message with ``expires_at`` in the past is excluded from
    ``mem_inbox(include_expired=False)`` and visible only with
    ``include_expired=True``.

    Fixture is planted via ``memory_write`` directly because
    ``mem_inbox_send`` rejects past ``expires_at`` with
    ``INBOX_TTL_IN_PAST``.
    """
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, env_name = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    slug = _unique_slug()

    token = use_session_factory(factory)
    try:
        # Open the channel.
        open_resp = await inbox_mod.mem_inbox_open(
            MemInboxOpenRequest(env_id=env_id, name=slug),
            ctx=ctx,
            settings=_settings(),
        )

        # Plant a live message (normal send path).
        live_send = await inbox_mod.mem_inbox_send(
            MemInboxSendRequest(to=open_resp.reference, body="still fresh"),
            ctx=ctx,
            settings=_settings(),
        )

        # Plant an already-expired message bypassing inbox_send's TTL guard.
        past = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
        expired = await memory_write(
            MemoryWriteRequest(
                kind=MemoryKind.message,
                title="stale",
                body="should be hidden by default",
                env_id=env_id,
                tags=["inbox"],
                entity_links=[open_resp.entity_id],
                expires_at=past,
            ),
            ctx=ctx,
            settings=_settings(),
        )

        # Default list — only the live message.
        default_resp = await inbox_mod.mem_inbox(
            MemInboxRequest(to=open_resp.reference),
            ctx=ctx,
            settings=_settings(),
        )
        ids_default = {item.id for item in default_resp.items}
        assert live_send.id in ids_default
        assert expired.id not in ids_default

        # Opt-in — both visible.
        all_resp = await inbox_mod.mem_inbox(
            MemInboxRequest(to=open_resp.reference, include_expired=True),
            ctx=ctx,
            settings=_settings(),
        )
        ids_all = {item.id for item in all_resp.items}
        assert live_send.id in ids_all
        assert expired.id in ids_all
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# UC invariant 5 — server-formatted references only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_formatted_references_only(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Every reference string in tool responses originates from the
    server and matches the canonical ``mem-inbox://<env>/<slug>``
    format. Clients never need to compose the URL themselves.
    """
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, env_name = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    slug = _unique_slug()
    expected = f"mem-inbox://{env_name}/{slug}"

    token = use_session_factory(factory)
    try:
        open_resp = await inbox_mod.mem_inbox_open(
            MemInboxOpenRequest(env_id=env_id, name=slug),
            ctx=ctx,
            settings=_settings(),
        )
        assert open_resp.reference == expected

        # A second open call with idempotent=True returns the same
        # server-formatted reference.
        open_resp2 = await inbox_mod.mem_inbox_open(
            MemInboxOpenRequest(env_id=env_id, name=slug, idempotent=True),
            ctx=ctx,
            settings=_settings(),
        )
        assert open_resp2.reference == expected
        assert open_resp2.created is False

        send_resp = await inbox_mod.mem_inbox_send(
            MemInboxSendRequest(to=open_resp.reference, body="ping"),
            ctx=ctx,
            settings=_settings(),
        )
        assert send_resp.reference == expected

        list_resp = await inbox_mod.mem_inbox(
            MemInboxRequest(to=open_resp.reference),
            ctx=ctx,
            settings=_settings(),
        )
        assert list_resp.reference == expected
    finally:
        reset_session_factory(token)


# ---------------------------------------------------------------------------
# Await-bug regression — env_name path through _resolve_env_refs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_await_bug_regression_env_name(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """All three tools called with ``env_name=`` (string) instead of
    ``env_id`` must succeed — verifies the ``await _resolve_env_refs``
    fix from commit ``f77bf6e`` still routes env-name resolution
    correctly.

    Before the fix, the coroutine was passed unawaited as ``request``
    to the rest of the pipeline; accessing ``.env_id`` on a coroutine
    raised ``AttributeError``.
    """
    _patch_session_scopes(monkeypatch)
    factory, _ = postgres_session_factories()
    env_id, agent_id, env_name = await _setup_env_and_agent(factory)
    ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
    slug = _unique_slug()

    token = use_session_factory(factory)
    try:
        # 1. mem_inbox_open with env_name (no env_id).
        open_resp = await inbox_mod.mem_inbox_open(
            MemInboxOpenRequest(env_name=env_name, name=slug),
            ctx=ctx,
            settings=_settings(),
        )
        assert open_resp.created is True
        assert open_resp.env_id == env_id
        assert open_resp.env_name == env_name

        # 2. mem_inbox_send with bare slug + env_name.
        send_resp = await inbox_mod.mem_inbox_send(
            MemInboxSendRequest(to=slug, body="via env_name", env_name=env_name),
            ctx=ctx,
            settings=_settings(),
        )
        assert send_resp.recipient_entity_id == open_resp.entity_id

        # 3. mem_inbox with bare slug + env_name.
        list_resp = await inbox_mod.mem_inbox(
            MemInboxRequest(to=slug, env_name=env_name),
            ctx=ctx,
            settings=_settings(),
        )
        assert list_resp.count == 1
        assert list_resp.items[0].id == send_resp.id
    finally:
        reset_session_factory(token)
