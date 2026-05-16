from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from memory_mcp.db.models import Memory, Task
from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.errors import EnvNotAttachedError, ForbiddenEnvError, InvalidInputError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import (
    MemoryResponse,
    MemoryWriteRequest,
    _ensure_macro_available,
    memory_write,
)
from memory_mcp import memories as memories_mod
from memory_mcp.playbooks import api as playbook_api


def _ctx(env_id: UUID) -> AgentContext:
    return AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])


def _memory(
    *,
    env_id: UUID,
    kind: str = "fact",
    status: str = "active",
    body: str = "body",
    title: str | None = "title",
    steps: list[str] | None = None,
    macro: str | None = None,
    memory_id: UUID | None = None,
) -> Memory:
    now = dt.datetime(2026, 5, 12, tzinfo=dt.UTC)
    return Memory(
        id=memory_id or uuid4(),
        env_id=env_id,
        kind=kind,
        status=status,
        title=title,
        body=body,
        trigger_description=None,
        steps=steps,
        macro=macro,
        salience=0.5,
        confidence=0.9,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        metadata_={},
        decision_meta=None,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _task_ref(
    *,
    task_id: UUID,
    env_id: UUID,
    status: str = "pending",
    description: str = "do the thing",
) -> Task:
    now = dt.datetime(2026, 5, 12, tzinfo=dt.UTC)
    return Task(
        id=task_id,
        env_id=env_id,
        title="task title",
        description=description,
        status=status,
        priority=50,
        playbook_id=None,
        version=1,
        created_at=now,
        updated_at=now,
        created_by_agent_id=None,
    )


def _response(env_id: UUID, *, steps: list[str], macro: str) -> MemoryResponse:
    return MemoryResponse(
        id=uuid4(),
        env_id=env_id,
        kind=MemoryKind.playbook,
        status=MemoryStatus.active,
        title="Runbook",
        body="body",
        trigger_description=None,
        steps=steps,
        macro=macro,
        tags=[],
        metadata={},
        salience=0.5,
        confidence=0.9,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        decision_meta=None,
        version=1,
        created_at=dt.datetime(2026, 5, 12, tzinfo=dt.UTC),
        updated_at=dt.datetime(2026, 5, 12, tzinfo=dt.UTC),
    )


def _patch_memory_session(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def fake_session_scope():
        yield object()

    monkeypatch.setattr(memories_mod, "session_scope", fake_session_scope)


@pytest.mark.asyncio
async def test_write_playbook_happy_path_normalizes_macro(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    captured: dict[str, Any] = {}
    _patch_memory_session(monkeypatch)

    async def fake_write_in_session(req: MemoryWriteRequest, **kwargs: Any) -> MemoryResponse:
        captured.update(kwargs)
        return _response(env_id, steps=kwargs["steps"], macro=kwargs["macro"])

    monkeypatch.setattr("memory_mcp.memories._memory_write_in_session", fake_write_in_session)

    out = await memory_write(
        MemoryWriteRequest(kind=MemoryKind.playbook, body="body", env_id=env_id, steps=[" step 1 "], macro=" Deploy "),
        ctx=_ctx(env_id),
    )

    assert captured["steps"] == ["step 1"]
    assert captured["macro"] == "deploy"
    assert out.steps == ["step 1"]
    assert out.macro == "deploy"


@pytest.mark.asyncio
async def test_write_playbook_missing_steps_rejected() -> None:
    env_id = uuid4()
    with pytest.raises(InvalidInputError, match="steps"):
        await memory_write(
            MemoryWriteRequest(kind=MemoryKind.playbook, body="body", env_id=env_id, macro="deploy"),
            ctx=_ctx(env_id),
        )


@pytest.mark.asyncio
async def test_write_playbook_missing_macro_rejected() -> None:
    env_id = uuid4()
    with pytest.raises(InvalidInputError, match="macro"):
        await memory_write(
            MemoryWriteRequest(kind=MemoryKind.playbook, body="body", env_id=env_id, steps=["one"]),
            ctx=_ctx(env_id),
        )


@pytest.mark.asyncio
async def test_write_playbook_empty_steps_rejected() -> None:
    env_id = uuid4()
    with pytest.raises(InvalidInputError, match="steps"):
        await memory_write(
            MemoryWriteRequest(kind=MemoryKind.playbook, body="body", env_id=env_id, steps=[], macro="deploy"),
            ctx=_ctx(env_id),
        )


@pytest.mark.asyncio
async def test_write_non_playbook_with_steps_rejected() -> None:
    env_id = uuid4()
    with pytest.raises(InvalidInputError, match="only valid"):
        await memory_write(
            MemoryWriteRequest(kind=MemoryKind.procedure, body="body", env_id=env_id, steps=["one"]),
            ctx=_ctx(env_id),
        )


class _ScalarResult:
    def __init__(self, value: UUID | None) -> None:
        self.value = value

    def scalar_one_or_none(self) -> UUID | None:
        return self.value


class _MacroSession:
    def __init__(self, existing: UUID | None) -> None:
        self.existing = existing

    async def execute(self, _stmt: Any) -> _ScalarResult:
        return _ScalarResult(self.existing)


class _OrigConstraint:
    def __init__(self, constraint_name: str) -> None:
        self.constraint_name = constraint_name


@pytest.mark.asyncio
async def test_macro_duplicate_same_env_friendly_select_rejected() -> None:
    with pytest.raises(InvalidInputError, match="macro already in use"):
        await _ensure_macro_available(_MacroSession(uuid4()), env_id=uuid4(), macro="deploy")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_macro_duplicate_integrity_error_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    _patch_memory_session(monkeypatch)

    async def raise_integrity(*_args: Any, **_kwargs: Any) -> MemoryResponse:
        raise IntegrityError("insert", {}, _OrigConstraint("ix_memories_macro_per_env"))

    monkeypatch.setattr("memory_mcp.memories._memory_write_in_session", raise_integrity)

    with pytest.raises(InvalidInputError, match="macro already in use"):
        await memory_write(
            MemoryWriteRequest(kind=MemoryKind.playbook, body="body", env_id=env_id, steps=["one"], macro="deploy"),
            ctx=_ctx(env_id),
        )


@pytest.mark.asyncio
async def test_macro_duplicate_across_different_envs_allowed() -> None:
    await _ensure_macro_available(_MacroSession(None), env_id=uuid4(), macro="deploy")  # type: ignore[arg-type]


class _MemoryResult:
    def __init__(self, *, one: Memory | None = None, many: list[Memory] | None = None, tags: list[tuple[UUID, str]] | None = None) -> None:
        self.one = one
        self.many = many or []
        self.tags = tags or []

    def scalar_one_or_none(self) -> Memory | None:
        return self.one

    def scalars(self) -> _MemoryResult:
        return self

    def all(self) -> list[Any]:
        if self.tags:
            return self.tags
        return self.many


class _PlaybookSession:
    def __init__(
        self,
        *,
        playbook: Memory | None,
        refs: list[Memory] | None = None,
        task_refs: list[Task] | None = None,
    ) -> None:
        self.playbook = playbook
        self.refs = refs or []
        self.task_refs = task_refs
        self.calls = 0

    async def execute(self, _stmt: Any) -> _MemoryResult:
        self.calls += 1
        if self.calls == 1:
            return _MemoryResult(one=self.playbook)
        if self.task_refs is not None and self.calls == 2:
            return _MemoryResult(many=self.task_refs)
        if self.calls == 2:
            return _MemoryResult(many=self.refs)
        return _MemoryResult(tags=[])


def _patch_playbook_session(monkeypatch: pytest.MonkeyPatch, session: _PlaybookSession) -> None:
    @asynccontextmanager
    async def fake_session_scope():
        yield session

    monkeypatch.setattr(playbook_api, "session_scope", fake_session_scope)


@pytest.mark.asyncio
async def test_playbook_invoke_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    playbook = _memory(env_id=env_id, kind="playbook", body="body", steps=["do one", "do two"], macro="deploy")
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=playbook))

    out = await playbook_api.playbook_invoke("DEPLOY", env_id, _ctx(env_id))

    assert out.playbook.id == playbook.id
    assert out.steps == ["do one", "do two"]
    assert out.missing_refs == []


@pytest.mark.asyncio
async def test_playbook_invoke_unknown_macro_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=None))

    with pytest.raises(NotFoundError):
        await playbook_api.playbook_invoke("missing", env_id, _ctx(env_id))


@pytest.mark.asyncio
async def test_playbook_invoke_env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    env_a, env_b = uuid4(), uuid4()
    playbook_a = _memory(env_id=env_a, kind="playbook", steps=["a"], macro="deploy")
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=playbook_a))

    with pytest.raises(EnvNotAttachedError):
        await playbook_api.playbook_invoke("deploy", env_a, _ctx(env_b))


@pytest.mark.asyncio
async def test_playbook_invoke_resolves_memory_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    ref = _memory(env_id=env_id, body="resolved body")
    playbook = _memory(
        env_id=env_id,
        kind="playbook",
        steps=[f"Use {{{{memory:{ref.id}}}}}"],
        macro="deploy",
    )
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=playbook, refs=[ref]))

    out = await playbook_api.playbook_invoke("deploy", env_id, _ctx(env_id))

    assert out.steps == ["Use resolved body"]
    assert [m.id for m in out.referenced_memories] == [ref.id]
    assert out.missing_refs == []


@pytest.mark.asyncio
async def test_playbook_invoke_archived_placeholder_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    archived_id = uuid4()
    playbook = _memory(
        env_id=env_id,
        kind="playbook",
        steps=[f"Use {{{{memory:{archived_id}}}}}"],
        macro="deploy",
    )
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=playbook, refs=[]))

    out = await playbook_api.playbook_invoke("deploy", env_id, _ctx(env_id))

    assert out.steps == [f"Use {{{{memory:{archived_id}}}}}"]
    assert out.missing_refs == [archived_id]


@pytest.mark.asyncio
async def test_playbook_invoke_resolves_task_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    task_id = uuid4()
    task = _task_ref(task_id=task_id, env_id=env_id, status="in_progress", description="ship H4")
    playbook = _memory(
        env_id=env_id,
        kind="playbook",
        steps=[f"Run {{{{task:{task_id}}}}}"],
        macro="deploy",
    )
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=playbook, task_refs=[task]))

    out = await playbook_api.playbook_invoke("deploy", env_id, _ctx(env_id))

    assert out.steps == [f"Run [task {str(task_id)[:8]}] in_progress: ship H4"]
    assert out.missing_task_refs == []


@pytest.mark.asyncio
async def test_playbook_invoke_unknown_task_placeholder_stays_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    missing_id = uuid4()
    playbook = _memory(
        env_id=env_id,
        kind="playbook",
        steps=[f"Run {{{{task:{missing_id}}}}}"],
        macro="deploy",
    )
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=playbook, task_refs=[]))

    out = await playbook_api.playbook_invoke("deploy", env_id, _ctx(env_id))

    assert out.steps == [f"Run {{{{task:{missing_id}}}}}"]
    assert out.missing_task_refs == [missing_id]


@pytest.mark.asyncio
async def test_playbook_invoke_cross_env_task_placeholder_stays_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    env_a, env_b = uuid4(), uuid4()
    task_id = uuid4()
    task = _task_ref(task_id=task_id, env_id=env_a, status="pending", description="other env")
    playbook = _memory(
        env_id=env_b,
        kind="playbook",
        steps=[f"Run {{{{task:{task_id}}}}}"],
        macro="deploy",
    )
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=playbook, task_refs=[task]))

    out = await playbook_api.playbook_invoke("deploy", env_b, _ctx(env_b))

    assert out.steps == [f"Run {{{{task:{task_id}}}}}"]
    assert out.missing_task_refs == [task_id]


@pytest.mark.asyncio
async def test_playbook_invoke_malformed_placeholder_stays_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    playbook = _memory(
        env_id=env_id,
        kind="playbook",
        steps=["Use {{memory:not-a-uuid}}"],
        macro="deploy",
    )
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=playbook))

    out = await playbook_api.playbook_invoke("deploy", env_id, _ctx(env_id))

    assert out.steps == ["Use {{memory:not-a-uuid}}"]
    assert out.missing_refs == []


@pytest.mark.asyncio
async def test_rbac_read_only_rejected_for_write_allowed_for_invoke(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()

    def require(role: str, env_id_arg: UUID | None, ctx: AgentContext) -> None:
        if role == "write":
            raise ForbiddenEnvError("read only")

    monkeypatch.setattr("memory_mcp.memories.rbac.require", require)
    with pytest.raises(ForbiddenEnvError):
        await memory_write(
            MemoryWriteRequest(kind=MemoryKind.playbook, body="body", env_id=env_id, steps=["one"], macro="deploy"),
            ctx=_ctx(env_id),
        )

    playbook = _memory(env_id=env_id, kind="playbook", steps=["one"], macro="deploy")
    _patch_playbook_session(monkeypatch, _PlaybookSession(playbook=playbook))
    monkeypatch.setattr(playbook_api.rbac, "require", require)

    out = await playbook_api.playbook_invoke("deploy", env_id, _ctx(env_id))
    assert out.steps == ["one"]
