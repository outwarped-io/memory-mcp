"""Playbook macro invocation API."""

from __future__ import annotations

import re
from uuid import UUID

from sqlalchemy import func, select

from memory_mcp import rbac
from memory_mcp.db.models import Memory, MemoryTag, Tag, Task
from memory_mcp.db.postgres import session_scope
from memory_mcp.errors import EnvNotAttachedError, InvalidInputError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import _to_response
from memory_mcp.playbooks.models import PlaybookInvokeResponse

_VISIBLE_STATUSES = ("active", "proposed")
_PLACEHOLDER_RE = re.compile(r"\{\{memory:([0-9a-f-]{36})\}\}", re.IGNORECASE)
_TASK_PLACEHOLDER_RE = re.compile(r"\{\{task:([0-9a-f-]{36})\}\}")


def _require_env_attached(env_id: UUID, ctx: AgentContext) -> None:
    if env_id not in set(ctx.attached_env_ids):
        raise EnvNotAttachedError(
            f"ENV_NOT_ATTACHED: env {env_id} is not attached to this session",
            env_id=str(env_id),
            attached_env_ids=[str(e) for e in ctx.attached_env_ids],
        )


async def playbook_invoke(
    macro: str,
    env_id: UUID,
    ctx: AgentContext,
) -> PlaybookInvokeResponse:
    """Fetch a playbook by macro and resolve same-env memory placeholders."""
    normalized_macro = macro.strip().lower()
    if not normalized_macro:
        raise InvalidInputError("playbook macro cannot be empty")

    _require_env_attached(env_id, ctx)
    rbac.require("read", env_id, ctx)

    async with session_scope() as session:
        playbook = (
            await session.execute(
                select(Memory).where(
                    Memory.env_id == env_id,
                    Memory.kind == "playbook",
                    Memory.macro.is_not(None),
                    func.lower(Memory.macro) == normalized_macro,
                    Memory.status.in_(list(_VISIBLE_STATUSES)),
                )
            )
        ).scalar_one_or_none()
        if playbook is None:
            raise NotFoundError(
                f"playbook macro {normalized_macro!r} not found",
                env_id=str(env_id),
                macro=normalized_macro,
            )

        steps = list(playbook.steps or [])
        ordered_refs: list[UUID] = []
        ordered_task_refs: list[UUID] = []
        for step in steps:
            for match in _PLACEHOLDER_RE.finditer(step):
                try:
                    memory_id = UUID(match.group(1))
                except ValueError:
                    continue
                if memory_id not in ordered_refs:
                    ordered_refs.append(memory_id)
            for match in _TASK_PLACEHOLDER_RE.finditer(step):
                try:
                    task_id = UUID(match.group(1))
                except ValueError:
                    continue
                if task_id not in ordered_task_refs:
                    ordered_task_refs.append(task_id)

        refs_by_id: dict[UUID, Memory] = {}
        missing_refs: list[UUID] = []
        if ordered_refs:
            rows = (
                (
                    await session.execute(
                        select(Memory).where(
                            Memory.env_id == env_id,
                            Memory.id.in_(ordered_refs),
                            Memory.status.in_(list(_VISIBLE_STATUSES)),
                        )
                    )
                )
                .scalars()
                .all()
            )
            refs_by_id = {row.id: row for row in rows}
            missing_refs = [memory_id for memory_id in ordered_refs if memory_id not in refs_by_id]

        tasks_by_id: dict[UUID, Task] = {}
        missing_task_refs: list[UUID] = []
        if ordered_task_refs:
            task_rows = (await session.execute(select(Task).where(Task.id.in_(ordered_task_refs)))).scalars().all()
            tasks_by_id = {row.id: row for row in task_rows}
            missing_task_refs = [
                task_id
                for task_id in ordered_task_refs
                if task_id not in tasks_by_id or tasks_by_id[task_id].env_id != env_id
            ]

        resolved_steps = [_resolve_step(step, refs_by_id, tasks_by_id, env_id) for step in steps]
        memories_for_tags = [
            playbook,
            *[refs_by_id[memory_id] for memory_id in ordered_refs if memory_id in refs_by_id],
        ]
        tags_by_id = await _load_tags(session, [memory.id for memory in memories_for_tags])

        return PlaybookInvokeResponse(
            playbook=_to_response(playbook, tags_by_id.get(playbook.id, [])),
            steps=resolved_steps,
            referenced_memories=[
                _to_response(refs_by_id[memory_id], tags_by_id.get(memory_id, []))
                for memory_id in ordered_refs
                if memory_id in refs_by_id
            ],
            missing_refs=missing_refs,
            missing_task_refs=missing_task_refs,
        )


def _resolve_step(step: str, refs_by_id: dict[UUID, Memory], tasks_by_id: dict[UUID, Task], env_id: UUID) -> str:
    def memory_repl(match: re.Match[str]) -> str:
        try:
            memory_id = UUID(match.group(1))
        except ValueError:
            return match.group(0)
        memory = refs_by_id.get(memory_id)
        return memory.body if memory is not None else match.group(0)

    def task_repl(match: re.Match[str]) -> str:
        try:
            task_id = UUID(match.group(1))
        except ValueError:
            return match.group(0)
        task = tasks_by_id.get(task_id)
        if task is None or task.env_id != env_id:
            return match.group(0)
        return _render_task_token_ref(task)

    return _TASK_PLACEHOLDER_RE.sub(task_repl, _PLACEHOLDER_RE.sub(memory_repl, step))


def _render_task_token_ref(task: Task) -> str:
    desc = task.description or task.title
    return f"[task {str(task.id)[:8]}] {task.status}: {desc}"


async def _load_tags(session, memory_ids: list[UUID]) -> dict[UUID, list[str]]:  # type: ignore[no-untyped-def]
    if not memory_ids:
        return {}
    rows = await session.execute(
        select(MemoryTag.memory_id, Tag.name)
        .join(Tag, Tag.id == MemoryTag.tag_id)
        .where(MemoryTag.memory_id.in_(memory_ids))
        .order_by(MemoryTag.memory_id, Tag.name)
    )
    tags_by_id: dict[UUID, list[str]] = {memory_id: [] for memory_id in memory_ids}
    for memory_id, name in rows.all():
        tags_by_id.setdefault(memory_id, []).append(name)
    return tags_by_id


__all__ = ["PlaybookInvokeResponse", "playbook_invoke"]
