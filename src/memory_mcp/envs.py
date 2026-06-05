"""Environment management for v1 (local-only build).

Surface
-------

* Pydantic schemas for tool I/O: :class:`EnvCreateRequest`,
  :class:`EnvResponse`, :class:`AttachedEnvsResponse`.
* Repo-level DB helpers: :func:`create_env`, :func:`get_env_by_name`,
  :func:`get_env_by_id`, :func:`list_envs`.
* Tool-facing async functions: :func:`env_create`, :func:`env_list`,
  :func:`env_get`, :func:`env_attach`, :func:`env_detach`,
  :func:`env_list_attached`. Each calls :func:`rbac.require` on entry —
  v1 always returns ``True``; v1.5 will flip the helper.
* :class:`EnvSessionState` — process-local in-memory store mapping
  ``session_id`` → set of attached ``env_id``. Cleared on process restart.

Why session-state lives in memory
---------------------------------

The ``sessions`` table doesn't carry an env list and we don't want a
destructive migration to add one. ``env_attach`` is a UX convenience, not
security: tools can also accept an explicit ``env_id`` per call. Memory
loss on restart is acceptable in v1 (local-only) and clients can re-attach
on reconnect.

v1.5 forward-compat
-------------------

When auth + ``env_grants`` lands, ``rbac.require`` will start enforcing
read/write/admin grants. The tool surface here doesn't change; only the
helper does.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any
from uuid import UUID

from memory_mcp_schemas.envs import (
    AttachedEnvsResponse,
    EnvCreateRequest,
    EnvResponse,
)
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import Environment
from memory_mcp.db.postgres import session_scope
from memory_mcp.errors import (
    AlreadyExistsError,
    EnvNotFoundError,
    EnvRefAmbiguousError,
    NotFoundError,
    SessionRequiredError,
)
from memory_mcp.identity import AgentContext

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Repo-level DB helpers
# ---------------------------------------------------------------------------


async def create_env(
    *,
    name: str,
    kind: str | None,
    retention_policy: dict[str, Any],
    default_embedding_model_id: str,
) -> Environment:
    """INSERT a new environment row. Raises :class:`AlreadyExistsError` on name collision."""
    row = Environment(
        name=name,
        kind=kind,
        retention_policy=retention_policy,
        default_embedding_model_id=default_embedding_model_id,
    )
    try:
        async with session_scope() as s:
            s.add(row)
            await s.flush()
            await s.refresh(row)
    except IntegrityError as exc:
        raise AlreadyExistsError(
            f"environment with name {name!r} already exists",
            name=name,
        ) from exc
    return row


async def get_env_by_name(name: str, *, include_deleted: bool = False) -> Environment | None:
    """SELECT … WHERE name = :name. Returns ``None`` if absent."""
    async with session_scope() as s:
        stmt = select(Environment).where(Environment.name == name)
        if not include_deleted:
            stmt = stmt.where(Environment.status == "active")
        return (await s.execute(stmt)).scalar_one_or_none()


async def get_env_by_name_ci(
    name: str,
    *,
    include_deleted: bool = False,
) -> Environment:
    """Case-insensitive env-by-name lookup.

    Raises:
        ENV_NOT_FOUND: no env matches ``name`` case-insensitively (under the
            active filter when ``include_deleted=False``).
        ENV_REF_AMBIGUOUS: multiple envs match case-insensitively. Only
            possible when ``include_deleted=True`` (active envs are unique
            case-insensitively per env_ops/rename.py invariant).

    Caller never sees ``None``.
    """
    async with session_scope() as s:
        stmt = select(Environment).where(func.lower(Environment.name) == name.lower())
        if not include_deleted:
            stmt = stmt.where(Environment.status == "active")
        rows = list((await s.execute(stmt)).scalars().all())
    if not rows:
        raise EnvNotFoundError(name=name)
    if len(rows) > 1:
        raise EnvRefAmbiguousError(name=name, candidate_ids=[row.id for row in rows])
    return rows[0]


async def get_env_by_id(env_id: UUID, *, include_deleted: bool = False) -> Environment | None:
    """SELECT … WHERE id = :id. Returns ``None`` if absent."""
    async with session_scope() as s:
        stmt = select(Environment).where(Environment.id == env_id)
        if not include_deleted:
            stmt = stmt.where(Environment.status == "active")
        return (await s.execute(stmt)).scalar_one_or_none()


async def list_envs(*, include_deleted: bool = False) -> list[Environment]:
    """Return all environments ordered by name."""
    async with session_scope() as s:
        stmt = select(Environment).order_by(Environment.name)
        if not include_deleted:
            stmt = stmt.where(Environment.status == "active")
        return list((await s.execute(stmt)).scalars().all())


# ---------------------------------------------------------------------------
# Session-state (in-memory)
# ---------------------------------------------------------------------------


class EnvSessionState:
    """Process-local map of ``session_id`` → attached ``env_id`` set.

    Cleared on process restart. Concurrency-safe under one event loop via
    a single :class:`asyncio.Lock` (the operations are O(1) and do no I/O,
    so coarse locking is fine for v1).
    """

    __slots__ = ("_lock", "_state")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._state: dict[UUID, set[UUID]] = {}

    async def attach(self, session_id: UUID, env_id: UUID) -> set[UUID]:
        async with self._lock:
            attached = self._state.setdefault(session_id, set())
            attached.add(env_id)
            return set(attached)

    async def detach(self, session_id: UUID, env_id: UUID) -> set[UUID]:
        async with self._lock:
            attached = self._state.get(session_id)
            if attached is None:
                return set()
            attached.discard(env_id)
            if not attached:
                self._state.pop(session_id, None)
                return set()
            return set(attached)

    async def attached_for(self, session_id: UUID) -> set[UUID]:
        async with self._lock:
            return set(self._state.get(session_id, set()))

    async def clear(self, session_id: UUID) -> None:
        async with self._lock:
            self._state.pop(session_id, None)

    # Test-only: full reset across all sessions.
    async def _reset(self) -> None:
        async with self._lock:
            self._state.clear()


@lru_cache(maxsize=1)
def get_env_session_state() -> EnvSessionState:
    """Process-wide cached state container. Tests can call ``cache_clear()``."""
    return EnvSessionState()


# ---------------------------------------------------------------------------
# Tool-facing functions (called from MCP transport in p1-mcp-transport)
# ---------------------------------------------------------------------------


async def env_create(
    request: EnvCreateRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> EnvResponse:
    """Create an environment. v1: no admin check (rbac.require is a no-op)."""
    rbac.require("admin", env_id=None, ctx=ctx)
    settings = settings or get_settings()

    default_model = request.default_embedding_model_id or settings.embedding_model_id
    row = await create_env(
        name=request.name,
        kind=request.kind,
        retention_policy=request.retention_policy,
        default_embedding_model_id=default_model,
    )
    log.info("envs: created env %s (%s)", row.id, row.name)
    return EnvResponse.model_validate(row)


async def env_list(*, ctx: AgentContext, include_deleted: bool = False) -> list[EnvResponse]:
    """List every environment. v1: read on all envs (no grants enforced)."""
    rbac.require("read", env_id=None, ctx=ctx)
    rows = await list_envs(include_deleted=include_deleted)
    return [EnvResponse.model_validate(r) for r in rows]


async def env_get(
    *,
    name: str | None = None,
    env_id: UUID | None = None,
    ctx: AgentContext,
    include_deleted: bool = False,
) -> EnvResponse:
    """Resolve an environment by name or id. Exactly one of ``name`` / ``env_id`` required."""
    if (name is None) == (env_id is None):
        raise ValueError("env_get requires exactly one of 'name' or 'env_id'")

    row = await (
        get_env_by_name(name, include_deleted=include_deleted)
        if name is not None
        else get_env_by_id(env_id, include_deleted=include_deleted)
    )  # type: ignore[arg-type]
    if row is None:
        raise NotFoundError(
            f"environment not found: {name or env_id}",
            name=name,
            id=str(env_id) if env_id else None,
        )
    rbac.require("read", env_id=row.id, ctx=ctx)
    return EnvResponse.model_validate(row)


async def env_attach(
    *,
    name: str,
    ctx: AgentContext,
) -> AttachedEnvsResponse:
    """Mark ``name`` as attached for the current session.

    Raises :class:`SessionRequiredError` if the caller didn't supply
    ``X-Session-Id`` — env_attach state is per-session and we have no
    other stable key in v1.
    """
    if ctx.session_id is None:
        raise SessionRequiredError(
            "env_attach requires X-Session-Id header (per-session state is in-memory)",
        )
    row = await get_env_by_name(name)
    if row is None:
        raise NotFoundError(f"environment not found: {name!r}", name=name)
    rbac.require("read", env_id=row.id, ctx=ctx)

    state = get_env_session_state()
    attached_ids = await state.attach(ctx.session_id, row.id)

    # Also reflect in ctx for the rest of the request — convenience.
    ctx.attached_env_ids = list(attached_ids)

    return await _build_attached_response(ctx.session_id, attached_ids)


async def env_detach(
    *,
    name: str,
    ctx: AgentContext,
) -> AttachedEnvsResponse:
    """Remove ``name`` from the session's attached set. Idempotent."""
    if ctx.session_id is None:
        raise SessionRequiredError(
            "env_detach requires X-Session-Id header (per-session state is in-memory)",
        )
    row = await get_env_by_name(name)
    if row is None:
        # detach on a missing env is a no-op for the session state, but
        # raise so the client knows the name was bad.
        raise NotFoundError(f"environment not found: {name!r}", name=name)

    state = get_env_session_state()
    attached_ids = await state.detach(ctx.session_id, row.id)
    ctx.attached_env_ids = list(attached_ids)

    return await _build_attached_response(ctx.session_id, attached_ids)


async def env_list_attached(*, ctx: AgentContext) -> AttachedEnvsResponse:
    """Return the session's currently attached envs."""
    if ctx.session_id is None:
        raise SessionRequiredError(
            "env_list_attached requires X-Session-Id header",
        )
    state = get_env_session_state()
    attached_ids = await state.attached_for(ctx.session_id)
    return await _build_attached_response(ctx.session_id, attached_ids)


async def _build_attached_response(
    session_id: UUID,
    attached_ids: set[UUID],
) -> AttachedEnvsResponse:
    """Hydrate attached env_ids into full :class:`EnvResponse` objects."""
    if not attached_ids:
        return AttachedEnvsResponse(session_id=session_id, attached=[])
    async with session_scope() as s:
        stmt = (
            select(Environment)
            .where(Environment.id.in_(attached_ids))
            .where(Environment.status == "active")
            .order_by(Environment.name)
        )
        rows = list((await s.execute(stmt)).scalars().all())
    return AttachedEnvsResponse(
        session_id=session_id,
        attached=[EnvResponse.model_validate(r) for r in rows],
    )


# ---------------------------------------------------------------------------
# Single entry point for the transport layer.
# ---------------------------------------------------------------------------


async def build_request_context(
    *,
    agent_id_header: str | None,
    agent_name_header: str | None,
    session_id_header: str | None,
    settings: Settings | None = None,
) -> AgentContext:
    """Build an :class:`AgentContext` with hydrated ``attached_env_ids``.

    This is the **only** function the MCP transport should call to construct
    a request-scoped identity. It does two things in order:

    1. ``IdentityResolver.resolve(...)`` — header-driven lookup-or-create
       in ``agents``; falls back to the server-default agent.
    2. If a ``session_id`` is present, hydrate ``ctx.attached_env_ids`` from
       :class:`EnvSessionState`. Tools may read ``ctx.attached_env_ids``
       knowing it is the authoritative set for this request.

    Tools must NOT read attached envs through any other path. The duplicate
    accessor (``EnvSessionState.attached_for``) exists only for the
    ``env_attach`` / ``env_detach`` family which mutates the canonical set.
    """
    # Local import to break the would-be cycle envs ↔ identity.
    from memory_mcp.identity import get_identity_resolver

    settings = settings or get_settings()
    resolver = get_identity_resolver(settings)
    ctx = await resolver.resolve(
        agent_id_header=agent_id_header,
        agent_name_header=agent_name_header,
        session_id_header=session_id_header,
    )
    if ctx.session_id is not None:
        state = get_env_session_state()
        attached = await state.attached_for(ctx.session_id)
        ctx.attached_env_ids = sorted(attached)
    return ctx


__all__ = [
    "AttachedEnvsResponse",
    "EnvCreateRequest",
    "EnvResponse",
    "EnvSessionState",
    "build_request_context",
    "create_env",
    "env_attach",
    "env_create",
    "env_detach",
    "env_get",
    "env_list",
    "env_list_attached",
    "get_env_by_id",
    "get_env_by_name",
    "get_env_by_name_ci",
    "get_env_session_state",
    "list_envs",
]
