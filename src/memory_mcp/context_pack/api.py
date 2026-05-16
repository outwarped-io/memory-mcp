"""Orchestrator for the ``mem_context_pack`` compound primitive."""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any
from uuid import UUID

from qdrant_client.http.exceptions import UnexpectedResponse
from sqlalchemy import exists, func, literal, or_, select
from sqlalchemy.orm import aliased

from memory_mcp import rbac
from memory_mcp.context_pack.budget import (
    MIN_TOKEN_BUDGET,
    calculate_section_caps,
    clamp_token_budget,
    estimate_tokens,
    truncate_to_token_budget,
)
from memory_mcp.context_pack.models import (
    ContextPackHit,
    ContextPackResponse,
    ContextPackSection,
    ContextPackSectionName,
)
from memory_mcp.db.models import GraphNode, Memory, Relation, Task
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import MemoryKind, TaskRelationKind, TaskStatus
from memory_mcp.errors import InvalidInputError, MemoryMCPError
from memory_mcp.identity import AgentContext

log = logging.getLogger(__name__)


_SECTION_ORDER: tuple[ContextPackSectionName, ...] = (
    "digest",
    "trigger_matched",
    "recent_journal",
    "tasks",
    "decisions",
    "playbooks",
    "archival",
)
_DIGEST_SECTION_NAMES = {"brief", "active_context"}
_DIGEST_KIND = "session_digest"
_JOURNAL_KINDS = {"journal_entry", "observation"}
_VISIBLE_STATUSES = ("proposed", "active")
_TERMINAL_TASK_STATUSES = (TaskStatus.done.value, TaskStatus.cancelled.value)
_TASK_DESC_TOKEN_RE = re.compile(r"[\W_]+")


async def pack(
    task_desc: str,
    env_id: UUID,
    token_budget: int = 4000,
    include_core: bool = True,
    include_journal: bool = True,
    *,
    ctx: AgentContext | None = None,
) -> ContextPackResponse:
    """Build an LLM-context-ready memory bundle for ``task_desc``.

    ``include_core`` is accepted as a reserved v0.6+ flag for future
    ``core_pinned`` support and is currently a no-op.
    """
    task_desc_used = task_desc.strip()
    if not task_desc_used:
        raise InvalidInputError("task_desc cannot be empty")
    if token_budget < MIN_TOKEN_BUDGET:
        raise InvalidInputError("token_budget too small")

    _ = include_core
    token_budget = clamp_token_budget(token_budget)
    if ctx is not None:
        rbac.require("read", env_id, ctx)

    sections_skipped: list[str] = []
    digest = await _fetch_section("digest", sections_skipped, _fetch_latest_digest(env_id))
    trigger_matches = await _fetch_section(
        "trigger_matched",
        sections_skipped,
        _fetch_trigger_matches(task_desc_used, env_id, top_k=20),
        default=[],
    )
    recent_journal = (
        await _fetch_section(
            "recent_journal",
            sections_skipped,
            _fetch_recent_journal(env_id),
            default=[],
        )
        if include_journal
        else []
    )
    tasks = await _fetch_section("tasks", sections_skipped, _fetch_tasks(env_id), default=[])

    excluded_ids = _collect_memory_ids([digest, *trigger_matches, *recent_journal])
    decisions = await _fetch_section(
        "decisions",
        sections_skipped,
        _fetch_decisions(env_id, exclude_ids=excluded_ids),
        default=[],
    )
    excluded_ids.update(_collect_memory_ids(decisions))
    playbooks = await _fetch_section(
        "playbooks",
        sections_skipped,
        _fetch_playbooks(task_desc_used, env_id, exclude_ids=excluded_ids),
        default=[],
    )
    excluded_ids.update(_collect_memory_ids(playbooks))
    archival = await _fetch_section(
        "archival",
        sections_skipped,
        _fetch_archival(env_id, exclude_ids=excluded_ids),
        default=[],
    )

    candidates: dict[ContextPackSectionName, list[Any]] = {
        "digest": [digest] if digest is not None else [],
        "trigger_matched": trigger_matches,
        "recent_journal": recent_journal,
        "tasks": tasks,
        "decisions": decisions,
        "playbooks": playbooks,
        "archival": archival,
    }
    available_sections = {
        name for name, items in candidates.items()
        if items and (include_journal or name != "recent_journal")
    }
    caps = calculate_section_caps(
        token_budget,
        include_journal=include_journal,
        available_sections=available_sections,
    )

    sections: list[ContextPackSection] = []
    seen_ids: set[UUID] = set()
    for name in _SECTION_ORDER:
        if name == "recent_journal" and not include_journal:
            if name not in sections_skipped:
                sections_skipped.append(name)
            continue
        items = candidates[name]
        if not items:
            if name not in sections_skipped:
                sections_skipped.append(name)
            continue
        if name == "tasks":
            section = _pack_task_section(name, items, caps.get(name, 0), seen_ids=seen_ids)
        else:
            section = _pack_section(
                name,
                items,
                caps.get(name, 0),
                seen_ids=seen_ids,
            )
        if section.items:
            sections.append(section)
        else:
            if name not in sections_skipped:
                sections_skipped.append(name)

    total_tokens = sum(section.tokens_used for section in sections)
    return ContextPackResponse(
        sections=sections,
        total_tokens=total_tokens,
        budget_used_pct=round((total_tokens / token_budget) * 100.0, 2),
        sections_skipped=sections_skipped,
        task_desc_used=task_desc_used,
    )


async def _fetch_latest_digest(env_id: UUID) -> Any | None:
    async with session_scope() as session:
        row = await session.execute(
            select(Memory)
            .where(Memory.env_id == env_id, Memory.kind == _DIGEST_KIND)
            .where(Memory.status.in_(list(_VISIBLE_STATUSES)))
            .order_by(Memory.created_at.desc(), Memory.id.desc())
            .limit(1)
        )
        return row.scalar_one_or_none()


async def _fetch_recent_journal(env_id: UUID, *, limit: int = 50) -> list[Any]:
    async with session_scope() as session:
        rows = await session.execute(
            select(Memory)
            .where(Memory.env_id == env_id, Memory.kind.in_(sorted(_JOURNAL_KINDS)))
            .where(Memory.status.in_(list(_VISIBLE_STATUSES)))
            .order_by(Memory.created_at.desc(), Memory.id.desc())
            .limit(limit)
        )
        return list(rows.scalars().all())


async def _fetch_playbooks(
    task_desc: str,
    env_id: UUID,
    *,
    top_k: int = 3,
    exclude_ids: set[UUID] | None = None,
) -> list[Any]:
    """Return up to ``top_k`` playbook memories matched by macro, then body vector.

    Pass 1 tokenizes ``task_desc`` and performs case-insensitive substring
    matches against the playbook ``macro`` column. Pass 2 tops up from Qdrant's
    body vector, restricted to ``kind=playbook``. Vector failures degrade to the
    pass-1 rows only, matching trigger-search fail-open behavior.
    """
    if top_k <= 0:
        return []
    exclude_ids = exclude_ids or set()
    tokens = _task_desc_tokens(task_desc)
    normalized_desc = " ".join(task_desc.lower().split())
    macro_conditions = [Memory.macro.ilike(f"%{token}%") for token in tokens]
    if normalized_desc:
        macro_conditions.extend([
            Memory.macro.ilike(f"%{normalized_desc}%"),
            literal(normalized_desc).ilike(func.concat("%", Memory.macro, "%")),
        ])
    out: list[Any] = []
    seen_ids: set[UUID] = set(exclude_ids)

    if macro_conditions:
        async with session_scope() as session:
            stmt = (
                select(Memory)
                .where(
                    Memory.env_id == env_id,
                    Memory.kind == MemoryKind.playbook.value,
                    Memory.status.in_(list(_VISIBLE_STATUSES)),
                    Memory.macro.is_not(None),
                    or_(*macro_conditions),
                )
                .order_by(Memory.salience.desc(), Memory.updated_at.desc(), Memory.id.desc())
                .limit(top_k)
            )
            if exclude_ids:
                stmt = stmt.where(Memory.id.notin_(list(exclude_ids)))
            rows = (await session.execute(stmt)).scalars().all()
        for memory in rows:
            if memory.id not in seen_ids:
                out.append(memory)
                seen_ids.add(memory.id)

    remaining = top_k - len(out)
    if remaining <= 0:
        return out

    try:
        semantic_ids = await _search_playbook_body_ids(task_desc, env_id, top_k=top_k)
    except (MemoryMCPError, UnexpectedResponse, RuntimeError, ValueError) as exc:
        log.warning(
            "context_pack playbook body search failed for env %s (%s); using macro matches only",
            env_id,
            exc.__class__.__name__,
        )
        return out

    semantic_ids = [memory_id for memory_id in semantic_ids if memory_id not in seen_ids][:remaining]
    if not semantic_ids:
        return out
    out.extend(await _fetch_memories_by_ids(env_id, semantic_ids))
    return out[:top_k]


async def _fetch_tasks(env_id: UUID, *, top_k: int = 5) -> list[Any]:
    """Return active task rows as context hits.

    Task IDs reuse the UUID field carried by ``ContextPackHit.memory_id``; task
    IDs and memory IDs share a UUID namespace in the API surface.
    """
    if top_k <= 0:
        return []
    first_limit = min(3, top_k)
    async with session_scope() as session:
        in_progress = list((await session.execute(
            select(Task)
            .where(Task.env_id == env_id, Task.status == TaskStatus.in_progress.value)
            .order_by(Task.priority.asc(), Task.updated_at.desc(), Task.id.asc())
            .limit(first_limit)
        )).scalars().all())

        remaining = top_k - len(in_progress)
        if remaining <= 0:
            return in_progress

        src_node = aliased(GraphNode)
        dst_node = aliased(GraphNode)
        dep = aliased(Relation)
        dst_task = aliased(Task)
        pending = list((await session.execute(
            select(Task)
            .join(src_node, src_node.task_id == Task.id)
            .where(
                Task.env_id == env_id,
                Task.status == TaskStatus.pending.value,
                ~exists(
                    select(1)
                    .select_from(dep)
                    .join(dst_node, dst_node.id == dep.dst_node_id)
                    .join(dst_task, dst_task.id == dst_node.task_id)
                    .where(
                        dep.src_node_id == src_node.id,
                        dep.type == TaskRelationKind.depends_on.value,
                        dst_task.status.not_in(_TERMINAL_TASK_STATUSES),
                    )
                ),
            )
            .order_by(Task.priority.asc(), Task.created_at.asc(), Task.id.asc())
            .limit(remaining)
        )).scalars().all())
    return [*in_progress, *pending]


async def _fetch_decisions(
    env_id: UUID,
    *,
    top_k: int = 5,
    exclude_ids: set[UUID] | None = None,
) -> list[Any]:
    if top_k <= 0:
        return []
    exclude_ids = exclude_ids or set()
    async with session_scope() as session:
        stmt = (
            select(Memory)
            .where(
                Memory.env_id == env_id,
                Memory.kind == MemoryKind.decision.value,
                Memory.decision_meta.is_not(None),
                Memory.decision_meta["status"].astext == "accepted",
                Memory.status.in_(list(_VISIBLE_STATUSES)),
            )
            .order_by(Memory.salience.desc(), Memory.updated_at.desc(), Memory.id.desc())
            .limit(top_k)
        )
        if exclude_ids:
            stmt = stmt.where(Memory.id.notin_(list(exclude_ids)))
        rows = await session.execute(stmt)
        return list(rows.scalars().all())


async def _fetch_archival(
    env_id: UUID,
    *,
    exclude_ids: set[UUID],
    limit: int = 100,
) -> list[Any]:
    async with session_scope() as session:
        stmt = (
            select(Memory)
            .where(
                Memory.env_id == env_id,
                Memory.kind.notin_([_DIGEST_KIND, *_JOURNAL_KINDS]),
                Memory.status.in_(list(_VISIBLE_STATUSES)),
            )
            .order_by(Memory.salience.desc(), Memory.updated_at.desc(), Memory.id.desc())
            .limit(limit)
        )
        if exclude_ids:
            stmt = stmt.where(Memory.id.notin_(list(exclude_ids)))
        rows = await session.execute(stmt)
        return list(rows.scalars().all())


async def _fetch_trigger_matches(task_desc: str, env_id: UUID, *, top_k: int) -> list[Any]:
    # TODO(F1): Prefer the internal trigger helper once F1 lands. The fallback
    # keeps F7 testable and returns an empty trigger section instead of failing.
    try:
        from memory_mcp.search.api import _search_by_trigger  # type: ignore[attr-defined]
    except ImportError:
        return []

    try:
        try:
            result = _search_by_trigger(task_desc=task_desc, env_id=env_id, top_k=top_k)
        except TypeError:
            result = _search_by_trigger(task_desc, env_id, top_k)
        if inspect.isawaitable(result):
            result = await result
    except (MemoryMCPError, UnexpectedResponse, RuntimeError, ValueError) as exc:
        log.warning(
            "context_pack trigger search failed for env %s (%s); skipping trigger_matched section",
            env_id,
            exc.__class__.__name__,
        )
        return []
    if result is None:
        return []
    items = getattr(result, "hits", result)
    tuple_ids = [
        item[0] for item in items
        if isinstance(item, tuple) and item and isinstance(item[0], UUID)
    ]
    if tuple_ids:
        return await _fetch_memories_by_ids(env_id, tuple_ids)

    out: list[Any] = []
    for item in items:
        memory = _unwrap_memory(item)
        if _memory_id(memory) is None:
            memory_id = _get(memory, "memory_id")
            if isinstance(memory_id, UUID):
                out.extend(await _fetch_memories_by_ids(env_id, [memory_id]))
                continue
        if memory is not None:
            out.append(memory)
    return out


async def _fetch_memories_by_ids(env_id: UUID, memory_ids: list[UUID]) -> list[Any]:
    if not memory_ids:
        return []
    ordered_ids = list(dict.fromkeys(memory_ids))
    async with session_scope() as session:
        rows = (await session.execute(
            select(Memory).where(
                Memory.env_id == env_id,
                Memory.id.in_(ordered_ids),
                Memory.status.in_(list(_VISIBLE_STATUSES)),
            )
        )).scalars().all()
    by_id = {m.id: m for m in rows}
    return [by_id[memory_id] for memory_id in ordered_ids if memory_id in by_id]


async def _search_playbook_body_ids(task_desc: str, env_id: UUID, *, top_k: int) -> list[UUID]:
    if not task_desc.strip() or top_k <= 0:
        return []
    from memory_mcp.config import get_settings
    from memory_mcp.embeddings.base import get_embedder
    from memory_mcp.search.api import _default_vector_store
    from memory_mcp.search.sem import sem_search

    settings = get_settings()
    vector_store = _default_vector_store(settings)
    embedder = get_embedder(settings)
    async with session_scope() as session:
        hits = await sem_search(
            session,
            vector_store=vector_store,
            embedder=embedder,
            query=task_desc,
            env_ids=[env_id],
            statuses=list(_VISIBLE_STATUSES),
            kinds=[MemoryKind.playbook.value],
            limit=top_k,
        )
    return [hit.memory_id for hit in hits]


async def _fetch_section(
    name: str,
    sections_skipped: list[str],
    awaitable: Any,
    *,
    default: Any = None,
) -> Any:
    try:
        return await awaitable
    except (MemoryMCPError, UnexpectedResponse, RuntimeError, ValueError) as exc:
        log.warning(
            "context_pack %s fetch failed (%s); skipping section",
            name,
            exc.__class__.__name__,
        )
        if name not in sections_skipped:
            sections_skipped.append(name)
        return default


def _collect_memory_ids(items: list[Any]) -> set[UUID]:
    return {
        memory_id
        for memory_id in (_memory_id(item) for item in items if item is not None)
        if memory_id is not None
    }


def _task_desc_tokens(task_desc: str) -> list[str]:
    return [
        token
        for token in (_TASK_DESC_TOKEN_RE.split(task_desc.lower()))
        if len(token) >= 2
    ]


def _pack_section(
    name: ContextPackSectionName,
    memories: list[Any],
    cap_tokens: int,
    *,
    seen_ids: set[UUID],
) -> ContextPackSection:
    items: list[ContextPackHit] = []
    tokens_used = 0
    truncation_count = 0
    for memory in memories:
        memory_id = _memory_id(memory)
        if memory_id is None or memory_id in seen_ids:
            continue
        remaining = cap_tokens - tokens_used
        if remaining <= 0:
            break
        hit = _build_hit(memory, cap_tokens=remaining, digest=name == "digest")
        if hit is None:
            continue
        items.append(hit)
        seen_ids.add(memory_id)
        tokens_used += hit.tokens_used
        if hit.body_truncated:
            truncation_count += 1
        if tokens_used >= cap_tokens:
            break
    return ContextPackSection(
        name=name,
        items=items,
        tokens_used=tokens_used,
        cap_tokens=cap_tokens,
        truncation_count=truncation_count,
    )


def _pack_task_section(
    name: ContextPackSectionName,
    tasks: list[Any],
    cap_tokens: int,
    *,
    seen_ids: set[UUID],
) -> ContextPackSection:
    items: list[ContextPackHit] = []
    tokens_used = 0
    truncation_count = 0
    for task in tasks:
        task_id = _memory_id(task)
        if task_id is None or task_id in seen_ids:
            continue
        remaining = cap_tokens - tokens_used
        if remaining <= 0:
            break
        hit = _build_task_hit(task, cap_tokens=remaining)
        if hit is None:
            continue
        items.append(hit)
        seen_ids.add(task_id)
        tokens_used += hit.tokens_used
        if hit.body_truncated:
            truncation_count += 1
        if tokens_used >= cap_tokens:
            break
    return ContextPackSection(
        name=name,
        items=items,
        tokens_used=tokens_used,
        cap_tokens=cap_tokens,
        truncation_count=truncation_count,
    )


def _build_hit(memory: Any, *, cap_tokens: int, digest: bool = False) -> ContextPackHit | None:
    title = _memory_title(memory)
    body = _memory_body(memory)
    if digest and estimate_tokens(body) > cap_tokens:
        selected = _extract_digest_sections(body)
        if selected:
            body = selected

    title_tokens = estimate_tokens(title)
    body_cap = max(0, cap_tokens - title_tokens)
    if body_cap <= 0 and body:
        return None
    body, truncated, body_tokens = truncate_to_token_budget(body, body_cap)
    tokens_used = title_tokens + body_tokens
    if tokens_used > cap_tokens:
        return None
    memory_id = _memory_id(memory)
    if memory_id is None:
        return None
    return ContextPackHit(
        memory_id=memory_id,
        title=title,
        body=body,
        kind=_memory_kind(memory),
        salience=_memory_salience(memory),
        tokens_used=tokens_used,
        body_truncated=truncated,
    )


def _build_task_hit(task: Any, *, cap_tokens: int) -> ContextPackHit | None:
    task_id = _memory_id(task)
    if task_id is None:
        return None
    title = str(_get(task, "title", "") or "")
    body = f"[priority={_get(task, 'priority', 0)}] status={_get(task, 'status', '')}"
    title_tokens = estimate_tokens(title)
    body_cap = max(0, cap_tokens - title_tokens)
    if body_cap <= 0 and body:
        return None
    body, truncated, body_tokens = truncate_to_token_budget(body, body_cap)
    tokens_used = title_tokens + body_tokens
    if tokens_used > cap_tokens:
        return None
    return ContextPackHit(
        memory_id=task_id,
        title=title,
        body=body,
        kind="task",
        salience=0.0,
        tokens_used=tokens_used,
        body_truncated=truncated,
    )


def _extract_digest_sections(body: str) -> str:
    sections: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in body.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = _normalize_digest_heading(match.group(2))
            current_lines = [line]
            continue
        if current_heading is not None:
            current_lines.append(line)
    if current_heading is not None:
        sections.append((current_heading, current_lines))

    selected: list[str] = []
    for heading, lines in sections:
        if heading in _DIGEST_SECTION_NAMES:
            selected.extend(lines)
    return "\n".join(selected).strip()


def _normalize_digest_heading(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")


def _unwrap_memory(item: Any) -> Any | None:
    if isinstance(item, dict):
        return item.get("memory", item)
    return getattr(item, "memory", item)


def _memory_id(memory: Any) -> UUID | None:
    value = _get(memory, "id", _get(memory, "memory_id"))
    return value if isinstance(value, UUID) else None


def _memory_title(memory: Any) -> str:
    title = _get(memory, "title")
    return str(title) if title else ""


def _memory_body(memory: Any) -> str:
    body = _get(memory, "body")
    return str(body) if body else ""


def _memory_kind(memory: Any) -> str:
    kind = _get(memory, "kind")
    value = getattr(kind, "value", kind)
    return str(value) if value else ""


def _memory_salience(memory: Any) -> float:
    try:
        return float(_get(memory, "salience", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
