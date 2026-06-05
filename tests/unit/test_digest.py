"""Unit tests for session digest + resume."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from memory_mcp.db.types import MemoryKind, MemorySourceType, MemoryStatus
from memory_mcp.digest import api as digest_api
from memory_mcp.digest.api import DigestInputs, digest_for_env, resume_for_env
from memory_mcp.digest.models import ResumeStats
from memory_mcp.digest.templates import (
    DigestMemorySnapshot,
    build_digest_context,
    build_template_sections,
    parse_digest_markdown,
    serialize_sections,
)
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryResponse


def _ctx(*envs: UUID) -> AgentContext:
    return AgentContext(agent_id=uuid4(), agent_name="digest-test", attached_env_ids=list(envs))


def _settings(kind: str = "template") -> SimpleNamespace:
    return SimpleNamespace(dream_summarizer=kind, llm_backend="null")


def _now(offset: int = 0) -> dt.datetime:
    return dt.datetime(2026, 5, 12, 17, 20 + offset, tzinfo=dt.UTC)


def _snap(
    body: str,
    *,
    env_id: UUID,
    kind: str = "fact",
    title: str | None = None,
    salience: float = 0.5,
    offset: int = 0,
) -> DigestMemorySnapshot:
    return DigestMemorySnapshot(
        id=uuid4(),
        env_id=env_id,
        kind=kind,
        title=title,
        body=body,
        salience=salience,
        created_at=_now(offset),
        updated_at=_now(offset),
    )


def _inputs(
    *,
    env_id: UUID,
    memories: list[DigestMemorySnapshot] | None = None,
    journals: list[DigestMemorySnapshot] | None = None,
    entity_count: int = 0,
) -> DigestInputs:
    memories = memories or []
    journals = journals or []
    return DigestInputs(
        memories=memories,
        journals=journals,
        latest_digest=None,
        memory_count=len(memories) + len(journals),
        entity_count=entity_count,
        last_journal_ts=max((j.created_at for j in journals), default=None),
    )


def _memory_response(env_id: UUID, memory_id: UUID | None = None) -> MemoryResponse:
    return MemoryResponse(
        id=memory_id or uuid4(),
        env_id=env_id,
        kind=MemoryKind.session_digest,
        status=MemoryStatus.active,
        title="Session digest 2026-05-12 17:20 UTC",
        body="digest body",
        trigger_description=None,
        tags=[],
        metadata={},
        salience=0.9,
        confidence=0.5,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        version=1,
        created_at=_now(),
        updated_at=_now(),
    )


@pytest.mark.asyncio
async def test_empty_env_digest_returns_placeholder_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    monkeypatch.setattr(digest_api, "_load_digest_inputs", AsyncMock(return_value=_inputs(env_id=env_id)))
    monkeypatch.setattr(digest_api, "memory_write", AsyncMock(return_value=_memory_response(env_id)))

    out = await digest_for_env(env_id, ctx=_ctx(env_id), settings=_settings())

    assert out.sections.brief.startswith(f"Environment {env_id} has no active memories")
    assert "No recent journal entries" in out.sections.active_context
    assert out.source_type == "digest-template"


@pytest.mark.asyncio
async def test_small_env_digest_writes_session_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memories = [
        _snap("FastMCP registers memory CRUD tools.", env_id=env_id, title="MCP tools", salience=0.8),
        _snap("Postgres is the source of truth.", env_id=env_id, title="Storage", salience=0.7),
        _snap("Qdrant stores vector projections.", env_id=env_id, title="Vectors", salience=0.6),
        _snap("Dream mode emits reviewable proposals.", env_id=env_id, title="Dream", salience=0.5),
        _snap("Use local-only loopback binding.", env_id=env_id, title="Safety", salience=0.4),
    ]
    captured = {}

    async def fake_write(request, **kwargs):  # type: ignore[no-untyped-def]
        captured["request"] = request
        captured["kwargs"] = kwargs
        return _memory_response(env_id)

    monkeypatch.setattr(
        digest_api,
        "_load_digest_inputs",
        AsyncMock(return_value=_inputs(env_id=env_id, memories=memories, entity_count=3)),
    )
    monkeypatch.setattr(digest_api, "memory_write", fake_write)

    out = await digest_for_env(env_id, ctx=_ctx(env_id), settings=_settings())

    assert "FastMCP" in out.sections.system_patterns
    request = captured["request"]
    assert request.kind == MemoryKind.session_digest
    assert request.salience == 0.9
    assert request.source_type == MemorySourceType.digest_template
    assert request.env_id == env_id


def test_large_env_truncates_older_low_salience_without_verbatim_body() -> None:
    env_id = uuid4()
    high = _snap("IMPORTANT current architecture summary", env_id=env_id, salience=0.99)
    lows = [_snap(f"LOW_SECRET_{i} " + ("x" * 500), env_id=env_id, salience=0.01, offset=-i) for i in range(20)]

    context = build_digest_context([high, *lows], [], max_chars=350)
    sections = build_template_sections(
        env_id=env_id,
        memories=[high, *lows],
        journals=[],
        entity_count=0,
        context=context,
    )
    body = serialize_sections(sections)

    assert "IMPORTANT current architecture" in body
    assert "LOW_SECRET_" not in body
    assert "Summarized 20 older/lower-salience memories" in body


@pytest.mark.asyncio
async def test_digest_idempotency_writes_two_rows_and_resume_uses_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_id = uuid4()
    written: list[UUID] = []

    async def fake_write(_request, **_kwargs):  # type: ignore[no-untyped-def]
        memory_id = uuid4()
        written.append(memory_id)
        return _memory_response(env_id, memory_id=memory_id)

    monkeypatch.setattr(digest_api, "_load_digest_inputs", AsyncMock(return_value=_inputs(env_id=env_id)))
    monkeypatch.setattr(digest_api, "memory_write", fake_write)

    first = await digest_for_env(env_id, ctx=_ctx(env_id), settings=_settings())
    second = await digest_for_env(env_id, ctx=_ctx(env_id), settings=_settings())

    assert first.memory_id != second.memory_id
    assert written == [first.memory_id, second.memory_id]

    latest = _snap(serialize_sections(second.sections), env_id=env_id, kind="session_digest")
    monkeypatch.setattr(
        digest_api,
        "_load_resume_inputs",
        AsyncMock(return_value=(latest, [], ResumeStats(memory_count=2, entity_count=0, last_journal_ts=None))),
    )
    resumed = await resume_for_env(env_id, ctx=_ctx(env_id))
    assert resumed.latest_digest == second.sections


@pytest.mark.asyncio
async def test_resume_returns_latest_digest_and_recent_journal(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    sections_body = serialize_sections(
        parse_digest_markdown(
            "## brief\nProject summary\n\n## active_context\nNow\n\n## system_patterns\nPatterns\n\n"
            "## tech_context\nTech\n\n## progress\nProgress\n\n## open_questions\nQuestions"
        )
    )
    latest = _snap(sections_body, env_id=env_id, kind="session_digest")
    journals = [
        _snap("newest journal", env_id=env_id, kind="journal_entry", offset=2),
        _snap("older journal", env_id=env_id, kind="journal_entry", offset=1),
    ]
    stats = ResumeStats(memory_count=3, entity_count=4, last_journal_ts=journals[0].created_at)
    monkeypatch.setattr(digest_api, "_load_resume_inputs", AsyncMock(return_value=(latest, journals, stats)))

    out = await resume_for_env(env_id, journal_tail=999, ctx=_ctx(env_id))

    assert out.latest_digest is not None
    assert out.latest_digest.brief == "Project summary"
    assert [j.body for j in out.recent_journal] == ["newest journal", "older journal"]
    assert out.summary_stats.entity_count == 4


@pytest.mark.asyncio
async def test_llm_unavailable_falls_back_to_template_source_type(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    captured = {}

    async def fake_write(request, **_kwargs):  # type: ignore[no-untyped-def]
        captured["source_type"] = request.source_type
        return _memory_response(env_id)

    monkeypatch.setattr(digest_api, "_load_digest_inputs", AsyncMock(return_value=_inputs(env_id=env_id)))
    monkeypatch.setattr(digest_api, "_bounded_llm_probe", AsyncMock(return_value={"status": "error"}))
    monkeypatch.setattr(digest_api, "memory_write", fake_write)

    out = await digest_for_env(env_id, ctx=_ctx(env_id), settings=_settings("llm"))

    assert out.summarizer_kind == "template"
    assert out.source_type == "digest-template"
    assert captured["source_type"] == MemorySourceType.digest_template


@pytest.mark.asyncio
async def test_env_isolation_digest_uses_only_requested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    env_a, env_b = uuid4(), uuid4()
    a_memory = _snap("env A only", env_id=env_a, title="A", salience=0.9)
    b_secret = "env B secret must not appear"

    async def fake_load(env_id: UUID, **_kwargs):  # type: ignore[no-untyped-def]
        assert env_id == env_a
        return _inputs(env_id=env_a, memories=[a_memory])

    monkeypatch.setattr(digest_api, "_load_digest_inputs", fake_load)
    monkeypatch.setattr(digest_api, "memory_write", AsyncMock(return_value=_memory_response(env_a)))

    out = await digest_for_env(env_a, ctx=_ctx(env_a, env_b), settings=_settings())
    body = serialize_sections(out.sections)

    assert "env A only" in body
    assert b_secret not in body


@pytest.mark.asyncio
async def test_llm_digest_fills_missing_required_sections_from_template(monkeypatch: pytest.MonkeyPatch) -> None:
    env_id = uuid4()
    memory = _snap("Current deployment task", env_id=env_id, title="Deploy", salience=0.9)
    captured = {}

    class FakeLLMClient:
        async def summarize(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return "## brief\nLLM brief only"

    async def fake_get_llm_client(_settings):  # type: ignore[no-untyped-def]
        return FakeLLMClient()

    async def fake_write(request, **_kwargs):  # type: ignore[no-untyped-def]
        captured["request"] = request
        return _memory_response(env_id)

    import memory_mcp.llm.base as llm_base

    monkeypatch.setattr(
        digest_api, "_load_digest_inputs", AsyncMock(return_value=_inputs(env_id=env_id, memories=[memory]))
    )
    monkeypatch.setattr(digest_api, "_bounded_llm_probe", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(llm_base, "get_llm_client", fake_get_llm_client)
    monkeypatch.setattr(digest_api, "memory_write", fake_write)

    out = await digest_for_env(env_id, ctx=_ctx(env_id), settings=_settings("llm"))

    assert out.summarizer_kind == "llm"
    assert out.source_type == "digest"
    assert out.sections.brief == "LLM brief only"
    assert out.sections.active_context
    assert captured["request"].source_type == MemorySourceType.digest
