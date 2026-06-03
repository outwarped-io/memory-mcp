"""Session digest and resume orchestration."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select

from memory_mcp import rbac
from memory_mcp._filters import exclude_expired_clause
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import Entity, Memory
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import MemoryKind, MemorySourceType, MemoryStatus
from memory_mcp.digest.models import (
    DigestMemoryEntry,
    DigestResponse,
    DigestSections,
    ResumeResponse,
    ResumeStats,
)
from memory_mcp.digest.templates import (
    DigestContext,
    DigestMemorySnapshot,
    build_digest_context,
    build_digest_prompt,
    build_template_sections,
    parse_digest_markdown,
    serialize_sections,
)
from memory_mcp.dream.api import _bounded_llm_probe
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryResponse, MemoryWriteRequest, memory_write

log = logging.getLogger(__name__)

_JOURNAL_KINDS = (MemoryKind.journal_entry.value, MemoryKind.observation.value)
_DIGEST_MAX_TOKENS = 1_200
_DIGEST_TEMPERATURE = 0.2
_DIGEST_CANDIDATE_LIMIT = 200
_REQUIRED_LLM_SECTIONS = ("brief", "active_context")


@dataclass(frozen=True)
class DigestInputs:
    memories: list[DigestMemorySnapshot]
    journals: list[DigestMemorySnapshot]
    latest_digest: DigestMemorySnapshot | None
    memory_count: int
    entity_count: int
    last_journal_ts: dt.datetime | None


async def digest_for_env(
    env_id: UUID,
    *,
    since_ts: dt.datetime | None = None,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> DigestResponse:
    settings = settings or get_settings()
    rbac.require("read", env_id, ctx)
    rbac.require("write", env_id, ctx)

    now = dt.datetime.now(dt.UTC)
    inputs = await _load_digest_inputs(env_id, since_ts=since_ts)
    context = build_digest_context(inputs.memories, inputs.journals)
    sections, summarizer_kind, source_type = await _summarize_digest(
        env_id=env_id,
        now=now,
        inputs=inputs,
        context=context,
        settings=settings,
    )
    body = serialize_sections(sections)
    written = await _write_digest_memory(
        env_id=env_id,
        now=now,
        body=body,
        source_type=source_type,
        ctx=ctx,
        settings=settings,
    )
    return DigestResponse(
        memory_id=written.id,
        sections=sections,
        summarizer_kind=summarizer_kind,
        source_type=source_type.value,
    )


async def resume_for_env(
    env_id: UUID,
    *,
    journal_tail: int = 20,
    ctx: AgentContext,
) -> ResumeResponse:
    rbac.require("read", env_id, ctx)
    tail = min(200, max(0, journal_tail))
    latest, journals, stats = await _load_resume_inputs(env_id, journal_tail=tail)
    return ResumeResponse(
        latest_digest=parse_digest_markdown(latest.body) if latest is not None else None,
        recent_journal=[_entry_from_snapshot(j) for j in journals],
        summary_stats=stats,
    )


async def _summarize_digest(
    *,
    env_id: UUID,
    now: dt.datetime,
    inputs: DigestInputs,
    context: DigestContext,
    settings: Settings,
) -> tuple[DigestSections, str, MemorySourceType]:
    template_sections = build_template_sections(
        env_id=env_id,
        memories=inputs.memories,
        journals=inputs.journals,
        entity_count=inputs.entity_count,
        context=context,
        total_memory_count=inputs.memory_count,
    )

    if settings.dream_summarizer != "llm":
        return template_sections, "template", MemorySourceType.digest_template

    llm_status = await _bounded_llm_probe(settings)
    if llm_status.get("status") != "ok":
        return template_sections, "template", MemorySourceType.digest_template

    try:
        from memory_mcp.llm.base import get_llm_client

        client = await get_llm_client(settings)
        raw = await client.summarize(
            build_digest_prompt(context, env_id=env_id, now=now),
            max_tokens=_DIGEST_MAX_TOKENS,
            temperature=_DIGEST_TEMPERATURE,
        )
        parsed_sections = parse_digest_markdown(raw)
        if not any(getattr(parsed_sections, name).strip() for name in DigestSections.model_fields):
            return template_sections, "template", MemorySourceType.digest_template
        sections = _validate_llm_sections(parsed_sections, template_sections)
        return sections, "llm", MemorySourceType.digest
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "mem_digest LLM summarization failed (%s); using template",
            exc.__class__.__name__,
        )
        return template_sections, "template", MemorySourceType.digest_template


def _validate_llm_sections(sections: DigestSections, template_sections: DigestSections) -> DigestSections:
    """Ensure required LLM digest sections are non-empty.

    Policy: if the LLM returns no parseable sections, the caller falls back to
    the full template digest. If only required sections are missing, keep the
    LLM output and fill those sections from the deterministic template so a
    partial response never stores an empty ``brief`` or ``active_context``.
    """
    updates: dict[str, str] = {}
    for name in _REQUIRED_LLM_SECTIONS:
        if not getattr(sections, name).strip():
            updates[name] = getattr(template_sections, name)
    return sections.model_copy(update=updates) if updates else sections


async def _write_digest_memory(
    *,
    env_id: UUID,
    now: dt.datetime,
    body: str,
    source_type: MemorySourceType,
    ctx: AgentContext,
    settings: Settings,
) -> MemoryResponse:
    title = f"Session digest {now.strftime('%Y-%m-%d %H:%M UTC')}"
    request = MemoryWriteRequest(
        kind=MemoryKind.session_digest,
        title=title,
        body=body,
        env_id=env_id,
        salience=0.9,
        entity_links=[],
        source_type=source_type,
        source_ref=f"env:{env_id}:until:{now.isoformat()}",
    )
    return await memory_write(request, ctx=ctx, settings=settings)


async def _load_digest_inputs(
    env_id: UUID,
    *,
    since_ts: dt.datetime | None,
) -> DigestInputs:
    async with session_scope() as s:
        latest_digest = (
            await s.execute(
                select(Memory)
                .where(Memory.env_id == env_id, Memory.kind == MemoryKind.session_digest.value)
                .where(Memory.status != MemoryStatus.archived.value)
                .where(exclude_expired_clause())
                .order_by(Memory.created_at.desc(), Memory.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        journal_cutoff = since_ts
        if journal_cutoff is None and latest_digest is not None:
            journal_cutoff = latest_digest.created_at

        mem_stmt = (
            select(Memory)
            .where(Memory.env_id == env_id)
            .where(Memory.status != MemoryStatus.archived.value)
            .where(exclude_expired_clause())
            .where(Memory.kind.notin_([MemoryKind.session_digest.value, *_JOURNAL_KINDS]))
        )
        if since_ts is not None:
            mem_stmt = mem_stmt.where(Memory.updated_at >= since_ts)
        memory_rows = (
            await s.execute(
                mem_stmt
                .order_by(Memory.salience.desc(), Memory.updated_at.desc(), Memory.id.desc())
                .limit(_DIGEST_CANDIDATE_LIMIT)
            )
        ).scalars().all()

        journal_stmt = (
            select(Memory)
            .where(Memory.env_id == env_id)
            .where(Memory.status != MemoryStatus.archived.value)
            .where(exclude_expired_clause())
            .where(Memory.kind.in_(_JOURNAL_KINDS))
        )
        if journal_cutoff is not None:
            journal_stmt = journal_stmt.where(Memory.created_at >= journal_cutoff)
        journal_rows = (
            await s.execute(
                journal_stmt
                .order_by(Memory.salience.desc(), Memory.updated_at.desc(), Memory.id.desc())
                .limit(_DIGEST_CANDIDATE_LIMIT)
            )
        ).scalars().all()

        memory_count = int(
            (
                await s.execute(
                    select(func.count())
                    .select_from(Memory)
                    .where(Memory.env_id == env_id)
                    .where(Memory.status != MemoryStatus.archived.value)
                    .where(exclude_expired_clause())
                )
            ).scalar_one()
        )
        entity_count = int(
            (
                await s.execute(
                    select(func.count()).select_from(Entity).where(Entity.env_id == env_id)
                )
            ).scalar_one()
        )
        last_journal_ts = (
            await s.execute(
                select(func.max(Memory.created_at))
                .where(Memory.env_id == env_id)
                .where(Memory.kind.in_(_JOURNAL_KINDS))
                .where(Memory.status != MemoryStatus.archived.value)
                .where(exclude_expired_clause())
            )
        ).scalar_one()

    return DigestInputs(
        memories=[_snapshot(m) for m in memory_rows],
        journals=[_snapshot(m) for m in journal_rows],
        latest_digest=_snapshot(latest_digest) if latest_digest is not None else None,
        memory_count=memory_count,
        entity_count=entity_count,
        last_journal_ts=last_journal_ts,
    )


async def _load_resume_inputs(
    env_id: UUID,
    *,
    journal_tail: int,
) -> tuple[DigestMemorySnapshot | None, list[DigestMemorySnapshot], ResumeStats]:
    async with session_scope() as s:
        latest = (
            await s.execute(
                select(Memory)
                .where(Memory.env_id == env_id, Memory.kind == MemoryKind.session_digest.value)
                .where(Memory.status != MemoryStatus.archived.value)
                .where(exclude_expired_clause())
                .order_by(Memory.created_at.desc(), Memory.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        journal_stmt = (
            select(Memory)
            .where(Memory.env_id == env_id)
            .where(Memory.kind.in_(_JOURNAL_KINDS))
            .where(Memory.status != MemoryStatus.archived.value)
            .where(exclude_expired_clause())
            .order_by(Memory.created_at.desc(), Memory.id.desc())
            .limit(journal_tail)
        )
        journals = (await s.execute(journal_stmt)).scalars().all() if journal_tail else []

        memory_count = int(
            (
                await s.execute(
                    select(func.count())
                    .select_from(Memory)
                    .where(Memory.env_id == env_id)
                    .where(Memory.status != MemoryStatus.archived.value)
                    .where(exclude_expired_clause())
                )
            ).scalar_one()
        )
        entity_count = int(
            (
                await s.execute(
                    select(func.count()).select_from(Entity).where(Entity.env_id == env_id)
                )
            ).scalar_one()
        )
        last_journal_ts = (
            await s.execute(
                select(func.max(Memory.created_at))
                .where(Memory.env_id == env_id)
                .where(Memory.kind.in_(_JOURNAL_KINDS))
                .where(Memory.status != MemoryStatus.archived.value)
                .where(exclude_expired_clause())
            )
        ).scalar_one()

    return (
        _snapshot(latest) if latest is not None else None,
        [_snapshot(j) for j in journals],
        ResumeStats(
            memory_count=memory_count,
            entity_count=entity_count,
            last_journal_ts=last_journal_ts,
        ),
    )


def _snapshot(memory: Memory) -> DigestMemorySnapshot:
    return DigestMemorySnapshot(
        id=memory.id,
        env_id=memory.env_id,
        kind=memory.kind,
        title=memory.title,
        body=memory.body,
        salience=float(memory.salience),
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


def _entry_from_snapshot(memory: DigestMemorySnapshot) -> DigestMemoryEntry:
    return DigestMemoryEntry(
        id=memory.id,
        env_id=memory.env_id,
        kind=memory.kind,
        title=memory.title,
        body=memory.body,
        salience=memory.salience,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )
