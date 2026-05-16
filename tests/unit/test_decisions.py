"""Unit tests for ADR-lite decision metadata."""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from memory_mcp.db.types import DecisionStatus, MemoryKind
from memory_mcp.decisions import api as decision_api
from memory_mcp.decisions.api import adr_export, validate_decision_meta
from memory_mcp.decisions.models import DecisionMeta
from memory_mcp.errors import ForbiddenEnvError, InvalidInputError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryWriteRequest, _validate_decision_meta_for_kind, memory_write


@dataclass
class FakeMemory:
    id: UUID
    env_id: UUID
    kind: str
    title: str | None = "Use ADRs"
    body: str = "We need decisions to be portable."
    status: str = "active"
    decision_meta: dict[str, Any] | None = None
    created_at: dt.datetime = dt.datetime(2026, 5, 12, tzinfo=dt.UTC)
    updated_at: dt.datetime = dt.datetime(2026, 5, 12, tzinfo=dt.UTC)


class FakeResult:
    def __init__(self, scalar: Any = None) -> None:
        self.scalar = scalar

    def scalar_one_or_none(self) -> Any:
        return self.scalar


class FakeSession:
    def __init__(self, *, target: Any = None, memory: Any = None) -> None:
        self.target = target
        self.memory = memory
        self.execute_calls = 0

    async def execute(self, _stmt: Any) -> FakeResult:
        self.execute_calls += 1
        return FakeResult(self.target)

    async def get(self, _model: Any, _id: UUID) -> Any:
        return self.memory


@asynccontextmanager
async def session_scope_returning(session: FakeSession):  # type: ignore[no-untyped-def]
    yield session


def ctx(*envs: UUID) -> AgentContext:
    return AgentContext(agent_id=uuid4(), agent_name="test", attached_env_ids=list(envs))


def accepted_meta(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "accepted",
        "rationale": "Use ADR-lite for durable decisions.",
        "constraints": ["Keep exports markdown", "Validate only when metadata is present"],
        "superseded_by": None,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_write_decision_with_full_decision_meta_happy() -> None:
    env_id = uuid4()
    out = await _validate_decision_meta_for_kind(
        kind=MemoryKind.decision.value,
        decision_meta=accepted_meta(),
        env_id=env_id,
        session=FakeSession(),  # type: ignore[arg-type]
    )
    assert out == accepted_meta(consequences=None)


@pytest.mark.asyncio
async def test_write_decision_with_decision_meta_null_happy() -> None:
    out = await _validate_decision_meta_for_kind(
        kind=MemoryKind.decision.value,
        decision_meta=None,
        env_id=uuid4(),
        session=FakeSession(),  # type: ignore[arg-type]
    )
    assert out is None


@pytest.mark.asyncio
async def test_write_non_decision_with_decision_meta_rejected() -> None:
    with pytest.raises(InvalidInputError, match="decision_meta only valid"):
        await _validate_decision_meta_for_kind(
            kind=MemoryKind.fact.value,
            decision_meta=accepted_meta(),
            env_id=uuid4(),
            session=FakeSession(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_decision_meta_missing_rationale_rejected() -> None:
    payload = accepted_meta()
    del payload["rationale"]
    with pytest.raises(InvalidInputError):
        await validate_decision_meta(payload, uuid4(), FakeSession())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_decision_meta_missing_status_rejected() -> None:
    payload = accepted_meta()
    del payload["status"]
    with pytest.raises(InvalidInputError):
        await validate_decision_meta(payload, uuid4(), FakeSession())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_superseded_status_requires_superseded_by() -> None:
    with pytest.raises(InvalidInputError, match="requires superseded_by"):
        await validate_decision_meta(
            accepted_meta(status="superseded", superseded_by=None),
            uuid4(),
            FakeSession(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_non_superseded_status_rejects_superseded_by() -> None:
    with pytest.raises(InvalidInputError, match="only valid when status='superseded'"):
        await validate_decision_meta(
            accepted_meta(status="accepted", superseded_by=str(uuid4())),
            uuid4(),
            FakeSession(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_superseded_by_non_decision_memory_rejected() -> None:
    with pytest.raises(InvalidInputError, match="kind=decision memory in the same env"):
        await validate_decision_meta(
            accepted_meta(status="superseded", superseded_by=str(uuid4())),
            uuid4(),
            FakeSession(target=None),  # SELECT filters non-decisions out.
        )  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_superseded_by_different_env_rejected() -> None:
    with pytest.raises(InvalidInputError, match="kind=decision memory in the same env"):
        await validate_decision_meta(
            accepted_meta(status="superseded", superseded_by=str(uuid4())),
            uuid4(),
            FakeSession(target=None),  # SELECT filters other envs out.
        )  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_superseded_by_archived_decision_same_env_allowed() -> None:
    env_id = uuid4()
    target_id = uuid4()
    target = FakeMemory(id=target_id, env_id=env_id, kind=MemoryKind.decision.value, status="archived")
    meta = await validate_decision_meta(
        accepted_meta(status="superseded", superseded_by=str(target_id)),
        env_id,
        FakeSession(target=target),  # type: ignore[arg-type]
    )
    assert meta is not None
    assert meta.status == DecisionStatus.superseded
    assert meta.superseded_by == target_id


@pytest.mark.asyncio
async def test_adr_export_full_decision_renders_all_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = FakeMemory(
        id=memory_id,
        env_id=env_id,
        kind=MemoryKind.decision.value,
        title="Adopt ADR-lite",
        body="Decision context body.",
        decision_meta=accepted_meta(rationale="Because lightweight records are enough.", constraints=["No hard FK"]),
    )
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    out = await adr_export(memory_id, ctx(env_id))

    assert out.status == "accepted"
    assert out.memory_id == memory_id
    assert "# Adopt ADR-lite" in out.markdown
    assert "**Status:** accepted" in out.markdown
    assert "## Context\n\nDecision context body." in out.markdown
    assert "## Decision\n\nBecause lightweight records are enough." in out.markdown
    assert "## Consequences\n\n_(none recorded)_" in out.markdown
    assert "## Constraints\n\n- No hard FK" in out.markdown
    assert "## Superseded By\n\n_(none)_" in out.markdown


@pytest.mark.asyncio
async def test_adr_export_null_consequences_renders_none_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = FakeMemory(
        id=memory_id,
        env_id=env_id,
        kind=MemoryKind.decision.value,
        decision_meta=accepted_meta(consequences=None),
    )
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    out = await adr_export(memory_id, ctx(env_id))

    assert "## Consequences\n\n_(none recorded)_" in out.markdown


@pytest.mark.asyncio
async def test_adr_export_empty_consequences_renders_none_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = FakeMemory(
        id=memory_id,
        env_id=env_id,
        kind=MemoryKind.decision.value,
        decision_meta=accepted_meta(consequences=[]),
    )
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    out = await adr_export(memory_id, ctx(env_id))

    assert "## Consequences\n\n_(none recorded)_" in out.markdown


@pytest.mark.asyncio
async def test_adr_export_populated_consequences_renders_bullets(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = FakeMemory(
        id=memory_id,
        env_id=env_id,
        kind=MemoryKind.decision.value,
        decision_meta=accepted_meta(consequences=["faster builds", "higher cost"]),
    )
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    out = await adr_export(memory_id, ctx(env_id))

    assert "## Consequences\n\n- faster builds\n- higher cost" in out.markdown


@pytest.mark.asyncio
async def test_adr_export_whitespace_only_consequences_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = FakeMemory(
        id=memory_id,
        env_id=env_id,
        kind=MemoryKind.decision.value,
        decision_meta=accepted_meta(consequences=["", "  "]),
    )
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    with pytest.raises(InvalidInputError, match="decision_meta is malformed"):
        await adr_export(memory_id, ctx(env_id))


@pytest.mark.asyncio
async def test_adr_export_missing_consequences_key_renders_none_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    meta = accepted_meta()
    meta.pop("consequences", None)
    memory = FakeMemory(
        id=memory_id,
        env_id=env_id,
        kind=MemoryKind.decision.value,
        decision_meta=meta,
    )
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    out = await adr_export(memory_id, ctx(env_id))

    assert "## Consequences\n\n_(none recorded)_" in out.markdown


@pytest.mark.asyncio
async def test_adr_export_decision_without_meta_renders_skeleton(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = FakeMemory(id=memory_id, env_id=env_id, kind=MemoryKind.decision.value, decision_meta=None)
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    out = await adr_export(memory_id, ctx(env_id))

    assert out.status is None
    assert out.markdown.startswith("> **Note:** This decision has no structured metadata.")
    assert "**Status:** (unset)" in out.markdown
    assert "_(no rationale captured)_" in out.markdown


@pytest.mark.asyncio
async def test_adr_export_non_decision_memory_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = FakeMemory(id=memory_id, env_id=env_id, kind=MemoryKind.fact.value)
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    with pytest.raises(InvalidInputError, match="kind=decision"):
        await adr_export(memory_id, ctx(env_id))


@pytest.mark.asyncio
async def test_adr_export_malformed_stored_decision_meta_raises_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = FakeMemory(
        id=memory_id,
        env_id=env_id,
        kind=MemoryKind.decision.value,
        decision_meta={"status": "accepted"},
    )
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    with pytest.raises(InvalidInputError, match="decision_meta is malformed"):
        await adr_export(memory_id, ctx(env_id))


@pytest.mark.asyncio
async def test_read_only_rbac_allows_adr_export_and_rejects_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = FakeMemory(id=memory_id, env_id=env_id, kind=MemoryKind.decision.value, decision_meta=accepted_meta())
    monkeypatch.setattr(decision_api, "session_scope", lambda: session_scope_returning(FakeSession(memory=memory)))

    def read_only(role: str, _env_id: UUID | None, _ctx: AgentContext) -> None:
        if role != "read":
            raise ForbiddenEnvError("read-only")

    monkeypatch.setattr(decision_api.rbac, "require", read_only)
    out = await adr_export(memory_id, ctx(env_id))
    assert out.status == "accepted"

    from memory_mcp import memories

    monkeypatch.setattr(memories.rbac, "require", read_only)
    with pytest.raises(ForbiddenEnvError):
        await memory_write(
            MemoryWriteRequest(
                kind=MemoryKind.decision,
                body="body",
                env_id=env_id,
                decision_meta=accepted_meta(),
            ),
            ctx=ctx(env_id),
        )


def test_decision_meta_strips_rationale_and_constraints() -> None:
    meta = DecisionMeta.model_validate(
        accepted_meta(rationale="  because  ", constraints=["  c1  ", "c2"]),
    )
    assert meta.rationale == "because"
    assert meta.constraints == ["c1", "c2"]
