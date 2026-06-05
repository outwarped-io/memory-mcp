"""Tests for :mod:`memory_mcp.envs` — pure-Python surface.

DB-touching behavior (env_create, env_list, env_get round-trip) lives in
the integration smoke and the p1-tests integration suite.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from memory_mcp.envs import (
    AttachedEnvsResponse,
    EnvCreateRequest,
    EnvSessionState,
    env_attach,
    env_detach,
    env_get,
    env_list_attached,
    get_env_session_state,
)
from memory_mcp.errors import SessionRequiredError
from memory_mcp.identity import AgentContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx_no_session() -> AgentContext:
    return AgentContext(agent_id=uuid4(), session_id=None)


@pytest.fixture
def ctx_with_session() -> AgentContext:
    return AgentContext(agent_id=uuid4(), session_id=uuid4())


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Ensure each test gets a fresh process-wide session-state singleton."""
    get_env_session_state.cache_clear()


# ---------------------------------------------------------------------------
# EnvSessionState
# ---------------------------------------------------------------------------


def test_session_state_attach_idempotent() -> None:
    async def go() -> None:
        st = EnvSessionState()
        sid, eid = uuid4(), uuid4()
        a1 = await st.attach(sid, eid)
        a2 = await st.attach(sid, eid)
        assert a1 == a2 == {eid}

    asyncio.run(go())


def test_session_state_attach_multiple_envs() -> None:
    async def go() -> None:
        st = EnvSessionState()
        sid = uuid4()
        e1, e2 = uuid4(), uuid4()
        await st.attach(sid, e1)
        attached = await st.attach(sid, e2)
        assert attached == {e1, e2}

    asyncio.run(go())


def test_session_state_detach_idempotent_and_cleans_empty() -> None:
    async def go() -> None:
        st = EnvSessionState()
        sid, eid = uuid4(), uuid4()
        await st.attach(sid, eid)
        # First detach removes; second detach is a no-op (state row gone).
        a1 = await st.detach(sid, eid)
        a2 = await st.detach(sid, eid)
        assert a1 == set()
        assert a2 == set()
        # After both attached envs are gone, the session key should be cleared.
        assert sid not in st._state  # noqa: SLF001 — internal-state assertion

    asyncio.run(go())


def test_session_state_isolation_between_sessions() -> None:
    async def go() -> None:
        st = EnvSessionState()
        s1, s2 = uuid4(), uuid4()
        e1, e2 = uuid4(), uuid4()
        await st.attach(s1, e1)
        await st.attach(s2, e2)
        assert await st.attached_for(s1) == {e1}
        assert await st.attached_for(s2) == {e2}

    asyncio.run(go())


def test_session_state_clear() -> None:
    async def go() -> None:
        st = EnvSessionState()
        sid, eid = uuid4(), uuid4()
        await st.attach(sid, eid)
        await st.clear(sid)
        assert await st.attached_for(sid) == set()

    asyncio.run(go())


def test_get_env_session_state_is_singleton() -> None:
    a = get_env_session_state()
    b = get_env_session_state()
    assert a is b


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


def test_env_create_request_minimal() -> None:
    req = EnvCreateRequest(name="work")
    assert req.name == "work"
    assert req.kind is None
    assert req.retention_policy == {}
    assert req.default_embedding_model_id is None


def test_env_create_request_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        EnvCreateRequest(name="")


def test_env_create_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EnvCreateRequest(name="x", bogus="value")  # type: ignore[call-arg]


def test_attached_envs_response_empty() -> None:
    sid = uuid4()
    resp = AttachedEnvsResponse(session_id=sid, attached=[])
    assert resp.session_id == sid
    assert resp.attached == []


# ---------------------------------------------------------------------------
# Tool-level guards (pure validation paths)
# ---------------------------------------------------------------------------


def test_env_attach_requires_session(ctx_no_session: AgentContext) -> None:
    async def go() -> None:
        with pytest.raises(SessionRequiredError) as ei:
            await env_attach(name="any", ctx=ctx_no_session)
        assert ei.value.code == "SESSION_REQUIRED"

    asyncio.run(go())


def test_env_detach_requires_session(ctx_no_session: AgentContext) -> None:
    async def go() -> None:
        with pytest.raises(SessionRequiredError):
            await env_detach(name="any", ctx=ctx_no_session)

    asyncio.run(go())


def test_env_list_attached_requires_session(ctx_no_session: AgentContext) -> None:
    async def go() -> None:
        with pytest.raises(SessionRequiredError):
            await env_list_attached(ctx=ctx_no_session)

    asyncio.run(go())


def test_env_get_requires_exactly_one_selector(ctx_with_session: AgentContext) -> None:
    async def go_neither() -> None:
        with pytest.raises(ValueError, match="exactly one"):
            await env_get(ctx=ctx_with_session)

    async def go_both() -> None:
        with pytest.raises(ValueError, match="exactly one"):
            await env_get(name="x", env_id=UUID(int=1), ctx=ctx_with_session)

    asyncio.run(go_neither())
    asyncio.run(go_both())


# ---------------------------------------------------------------------------
# Forward-compat invariant: v1.5 will swap rbac.require to a denying impl.
# Tools must abort BEFORE any DB work when require() raises. We monkey-patch
# require() to a deny-shim and assert each tool surface aborts cleanly without
# touching the database (no engine init in unit tests proves this).
# ---------------------------------------------------------------------------


def test_rbac_deny_aborts_env_create(
    monkeypatch: pytest.MonkeyPatch,
    ctx_with_session: AgentContext,
) -> None:
    from memory_mcp import envs as envs_module
    from memory_mcp.errors import ForbiddenEnvError

    def _deny(*_a: object, **_k: object) -> None:
        raise ForbiddenEnvError("denied")

    monkeypatch.setattr(envs_module.rbac, "require", _deny)

    async def go() -> None:
        with pytest.raises(ForbiddenEnvError):
            # Would touch the DB if rbac.require didn't raise first.
            await envs_module.env_create(EnvCreateRequest(name="x"), ctx=ctx_with_session)

    asyncio.run(go())


def test_rbac_deny_aborts_env_list(
    monkeypatch: pytest.MonkeyPatch,
    ctx_with_session: AgentContext,
) -> None:
    from memory_mcp import envs as envs_module
    from memory_mcp.errors import ForbiddenEnvError

    def _deny(*_a: object, **_k: object) -> None:
        raise ForbiddenEnvError("denied")

    monkeypatch.setattr(envs_module.rbac, "require", _deny)

    async def go() -> None:
        with pytest.raises(ForbiddenEnvError):
            await envs_module.env_list(ctx=ctx_with_session)

    asyncio.run(go())
