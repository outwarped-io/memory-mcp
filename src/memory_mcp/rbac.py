"""RBAC helper — **no-op pass-through in v1 (local-only build)**.

Forward-compat invariant: every tool calls ``rbac.require(role, env_id, ctx)``
on entry, even though it always returns ``None`` in v1. When v1.5 introduces
multi-tenancy + bearer auth, only this module changes — tool code is
untouched, because :func:`require` raises on denial rather than returning a
boolean the caller would have to check.

The signature is the one v1.5 will need:

* ``role``    — minimum role required (``read`` < ``write`` < ``admin``)
* ``env_id``  — environment being accessed; ``None`` for global / pre-env
                 operations such as ``env_create`` or ``env_list``
* ``ctx``     — request-scoped :class:`AgentContext`

Denial contract (v1.5): raise :class:`memory_mcp.errors.UnauthorizedError`
or :class:`memory_mcp.errors.ForbiddenEnvError`. Callers must therefore call
``rbac.require(...)`` for the side effect, not for the (now ``None``) return
value — confirmed by the test suite, which monkey-patches a denying
implementation and asserts every tool surface raises before doing work.

Callers should treat the helper as side-effect-free and idempotent in v1.
v1.5 will add audit-log emission on denial.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import UUID

if TYPE_CHECKING:
    from memory_mcp.identity import AgentContext

Role = Literal["read", "write", "admin"]


def require(  # noqa: ARG001 — v1 no-op consumes args for forward-compat shape
    role: Role,
    env_id: UUID | None,
    ctx: AgentContext,
) -> None:
    """No-op in v1; raises on denial in v1.5.

    Returns ``None`` — callers must NOT inspect the return value. The contract
    is that v1.5 will raise :class:`UnauthorizedError` or
    :class:`ForbiddenEnvError` to deny; v1 just lets the call through.
    """
    return None


__all__ = ["Role", "require"]
