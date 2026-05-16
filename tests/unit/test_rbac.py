"""Unit tests for the no-op RBAC shim.

The whole point of v1's rbac module is that it never denies. These tests
encode that contract so v1.5 PRs that flip the helper to a real check will
deliberately remove this file rather than silently break it.

Contract (v1.5 must preserve, v1 satisfies trivially):

1. :func:`rbac.require` returns ``None`` — callers must NOT inspect a return.
2. Denial is signalled by **raising** :class:`UnauthorizedError` or
   :class:`ForbiddenEnvError` — never by ``return False``.
3. ``env_id`` is ``UUID | None``; ``None`` means "global/pre-env operation"
   (e.g. ``env_create`` before any env exists, ``env_list`` across all envs).
"""

from __future__ import annotations

import uuid

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from memory_mcp import rbac
from memory_mcp.identity import AgentContext


def _ctx() -> AgentContext:
    return AgentContext(agent_id=uuid.uuid4(), agent_name="test", attached_env_ids=[])


def test_require_returns_none_for_read() -> None:
    assert rbac.require("read", uuid.uuid4(), _ctx()) is None


def test_require_returns_none_for_write() -> None:
    assert rbac.require("write", uuid.uuid4(), _ctx()) is None


def test_require_returns_none_for_admin() -> None:
    assert rbac.require("admin", uuid.uuid4(), _ctx()) is None


def test_require_accepts_env_id_none_for_global_operations() -> None:
    """``env_create`` and ``env_list`` legitimately call with ``env_id=None``."""
    assert rbac.require("admin", None, _ctx()) is None
    assert rbac.require("read", None, _ctx()) is None


def test_require_does_not_raise_when_ctx_has_no_attached_envs() -> None:
    """Even if caller hasn't attached the env, v1 still allows access."""
    ctx = AgentContext(agent_id=uuid.uuid4(), attached_env_ids=[])
    target_env = uuid.uuid4()
    assert rbac.require("admin", target_env, ctx) is None


def test_mem_auto_context_requires_requested_env_attachment() -> None:
    from memory_mcp.errors import EnvNotAttachedError
    from memory_mcp.mcp_app import _require_env_attached

    target_env = uuid.uuid4()
    other_env = uuid.uuid4()
    ctx = AgentContext(agent_id=uuid.uuid4(), attached_env_ids=[other_env])

    with pytest.raises(EnvNotAttachedError) as exc_info:
        _require_env_attached(target_env, ctx)

    assert exc_info.value.code == "ENV_NOT_ATTACHED"
    assert exc_info.value.details["env_id"] == str(target_env)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("task_substep", lambda task_id, memory_id: {"parent_task_id": task_id, "title": "child"}),
        ("task_dep_link", lambda task_id, memory_id: {"src_task_id": task_id, "dst_task_id": uuid.uuid4()}),
        ("task_status_set", lambda task_id, memory_id: {
            "task_id": task_id,
            "status": "in_progress",
            "expected_version": 1,
        }),
        ("task_link_memory", lambda task_id, memory_id: {
            "request": {
                "task_id": task_id,
                "memory_id": memory_id,
                "relation": "motivated_by",
            },
        }),
        ("adr_export", lambda task_id, memory_id: {"memory_id": memory_id}),
    ],
)
async def test_new_task_and_adr_tools_require_attached_env_at_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    args,
) -> None:
    from memory_mcp import mcp_app

    env_id = uuid.uuid4()
    task_id = uuid.uuid4()
    memory_id = uuid.uuid4()

    async def resolve_task_env(_task_id):
        return env_id

    async def resolve_memory_env(_memory_id):
        return env_id

    async def resolve_ctx(*, agent_id, attached_env_ids, attached_env_names=None, settings=None):
        return mcp_app.AgentContext(
            agent_id=agent_id or uuid.uuid4(),
            agent_name="test",
            attached_env_ids=list(attached_env_ids or []),
        )

    async def unexpected_impl(*_args, **_kwargs):
        raise AssertionError("domain implementation should not be called")

    monkeypatch.setattr(mcp_app, "_resolve_ctx", resolve_ctx)
    monkeypatch.setattr(mcp_app, "_resolve_task_env", resolve_task_env)
    monkeypatch.setattr(mcp_app, "_resolve_memory_env", resolve_memory_env)
    monkeypatch.setattr(mcp_app, "task_substep_impl", unexpected_impl)
    monkeypatch.setattr(mcp_app, "task_dep_link_impl", unexpected_impl)
    monkeypatch.setattr(mcp_app, "task_status_set_impl", unexpected_impl)
    monkeypatch.setattr(mcp_app, "task_link_memory_impl", unexpected_impl)
    monkeypatch.setattr(mcp_app, "adr_export_memory", unexpected_impl)

    mcp = mcp_app.build_mcp_server()
    call_args = args(task_id, memory_id)
    call_args["agent_id"] = uuid.uuid4()
    with pytest.raises(ToolError, match=r"\[ENV_NOT_ATTACHED\]"):
        await mcp.call_tool(tool_name, call_args)
