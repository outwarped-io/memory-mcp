"""Environment full-purge implementation for v0.8 env operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp.db.models import (
    AuditLog,
    DreamProposal,
    DreamRun,
    Entity,
    EntityAlias,
    EnvGrant,
    Environment,
    GraphNode,
    Memory,
    MemoryLineage,
    MemorySource,
    MemoryTag,
    Outbox,
    OutboxDelivery,
    Relation,
    Tag,
    Task,
)
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import OutboxAggregateType, OutboxOp, OutboxSink
from memory_mcp.errors import InvalidInputError, MemoryMCPError, NotFoundError
from memory_mcp.identity import AgentContext

from memory_mcp_schemas.env_ops import EnvDeleteRequest, EnvDeleteResponse


class RefsBlockingDeleteError(MemoryMCPError):
    """External references into or out of the environment block deletion."""

    code = "REFS_BLOCKING_DELETE"


async def delete_env(request: EnvDeleteRequest, *, ctx: AgentContext) -> EnvDeleteResponse:
    """Soft-delete an environment with cascade. See §9 + §17.4.

    The environment row is NOT hard-deleted — only marked status='deleted' so its UUID
    remains valid forever (avoids breaking lineage edges in other envs). All env-scoped
    rows ARE hard-deleted in dependency order.

    Refuses unless ``confirm_destroy=True``.

    With ``cascade_external_refs=False`` (default), the call fails fast if any memory in
    this env is referenced from another env. With ``cascade_external_refs=True``, external
    rows pointing in are dropped (lineage) or nulled (task/playbook and supersession FKs).

    Already-deleted environments return idempotent success. This is safer for scripts that
    retry destructive operations after a partial caller-side failure.
    """

    if not request.confirm_destroy:
        exc = InvalidInputError("env_delete requires confirm_destroy=True")
        exc.code = "CONFIRM_DESTROY_REQUIRED"
        raise exc

    _require_delete(request.env_id, ctx)

    async with session_scope() as session:
        env = await session.get(Environment, request.env_id)
        if env is None:
            raise NotFoundError(
                f"environment not found: {request.env_id}",
                id=str(request.env_id),
            )

        if env.status == "deleted":
            return EnvDeleteResponse(
                env_id=request.env_id,
                confirm_destroy=True,
                cascade_external_refs=request.cascade_external_refs,
                counts=_zero_counts(),
                external_lineage_exit_dropped=0,
                external_lineage_entry_dropped=0,
            )

        refs = await _scan_external_refs(session, request.env_id)
        if _has_refs(refs) and not request.cascade_external_refs:
            raise RefsBlockingDeleteError(
                "env_delete blocked by references crossing environment boundaries",
                samples=refs,
                count=sum(len(samples) for samples in refs.values()),
            )

        external_exit_dropped = 0
        external_entry_dropped = 0
        if request.cascade_external_refs:
            external_exit_dropped = await _drop_external_lineage(
                session,
                env_id=request.env_id,
                direction="exit",
            )
            external_entry_dropped = await _drop_external_lineage(
                session,
                env_id=request.env_id,
                direction="entry",
            )
            await _neutralize_external_non_lineage_refs(session, request.env_id)

        counts = await _delete_env_rows(session, request.env_id)

        await session.execute(
            update(Environment)
            .where(Environment.id == request.env_id)
            .values(status="deleted", deleted_at=func.now())
        )

        await enqueue_event(
            session,
            aggregate_type=OutboxAggregateType.env,
            aggregate_id=request.env_id,
            aggregate_version=1,
            env_id=request.env_id,
            op=OutboxOp.tombstone,
            payload={
                "event": "EnvDeleted",
                "env_id": str(request.env_id),
                "counts": dict(counts),
                "cascade_external_refs": request.cascade_external_refs,
                "external_lineage_exit_dropped": external_exit_dropped,
                "external_lineage_entry_dropped": external_entry_dropped,
            },
            # Env events are metadata-only by default. Force a durable row so
            # deletion is auditable even though no projection sink subscribes.
            sinks=(OutboxSink.qdrant,),
        )

    return EnvDeleteResponse(
        env_id=request.env_id,
        confirm_destroy=True,
        cascade_external_refs=request.cascade_external_refs,
        counts=counts,
        external_lineage_exit_dropped=external_exit_dropped,
        external_lineage_entry_dropped=external_entry_dropped,
    )


def _require_delete(env_id: UUID, ctx: AgentContext) -> None:
    try:
        rbac.require("delete", env_id, ctx)  # type: ignore[arg-type]
    except (KeyError, ValueError):
        rbac.require("admin", env_id=None, ctx=ctx)


def _memory_ids_for_env(env_id: UUID) -> select[tuple[UUID]]:
    return select(Memory.id).where(Memory.env_id == env_id)


async def _scan_external_refs(session: AsyncSession, env_id: UUID) -> dict[str, list[str]]:
    memory_ids = _memory_ids_for_env(env_id)
    refs: dict[str, list[str]] = {
        "external_lineage_exit": [],
        "external_lineage_entry": [],
        "external_task_playbook": [],
        "external_superseded_by": [],
    }

    exit_rows = await session.execute(
        select(
            MemoryLineage.parent_memory_id,
            MemoryLineage.child_memory_id,
            MemoryLineage.relation,
        )
        .where(MemoryLineage.parent_memory_id.in_(memory_ids))
        .where(MemoryLineage.child_memory_id.not_in(memory_ids))
        .limit(20)
    )
    refs["external_lineage_exit"] = [
        f"{parent}:{child}:{relation}" for parent, child, relation in exit_rows.all()
    ]

    entry_rows = await session.execute(
        select(
            MemoryLineage.parent_memory_id,
            MemoryLineage.child_memory_id,
            MemoryLineage.relation,
        )
        .where(MemoryLineage.child_memory_id.in_(memory_ids))
        .where(MemoryLineage.parent_memory_id.not_in(memory_ids))
        .limit(20)
    )
    refs["external_lineage_entry"] = [
        f"{parent}:{child}:{relation}" for parent, child, relation in entry_rows.all()
    ]

    task_rows = await session.execute(
        select(Task.id, Task.playbook_id)
        .where(Task.env_id != env_id)
        .where(Task.playbook_id.in_(memory_ids))
        .limit(20)
    )
    refs["external_task_playbook"] = [
        f"{task_id}:{memory_id}" for task_id, memory_id in task_rows.all()
    ]

    superseded_rows = await session.execute(
        select(Memory.id, Memory.superseded_by)
        .where(Memory.env_id != env_id)
        .where(Memory.superseded_by.in_(memory_ids))
        .limit(20)
    )
    refs["external_superseded_by"] = [
        f"{memory_id}:{superseded_by}" for memory_id, superseded_by in superseded_rows.all()
    ]

    return {key: samples for key, samples in refs.items() if samples}


def _has_refs(refs: Mapping[str, list[str]]) -> bool:
    return any(refs.values())


async def _drop_external_lineage(
    session: AsyncSession,
    *,
    env_id: UUID,
    direction: str,
) -> int:
    memory_ids = _memory_ids_for_env(env_id)
    if direction == "exit":
        stmt = (
            delete(MemoryLineage)
            .where(MemoryLineage.parent_memory_id.in_(memory_ids))
            .where(MemoryLineage.child_memory_id.not_in(memory_ids))
        )
    else:
        stmt = (
            delete(MemoryLineage)
            .where(MemoryLineage.child_memory_id.in_(memory_ids))
            .where(MemoryLineage.parent_memory_id.not_in(memory_ids))
        )
    result = await session.execute(stmt)
    return int(result.rowcount or 0)


async def _neutralize_external_non_lineage_refs(session: AsyncSession, env_id: UUID) -> None:
    memory_ids = _memory_ids_for_env(env_id)
    await session.execute(
        update(Task)
        .where(Task.env_id != env_id)
        .where(Task.playbook_id.in_(memory_ids))
        .values(playbook_id=None)
    )
    await session.execute(
        update(Memory)
        .where(Memory.env_id != env_id)
        .where(Memory.superseded_by.in_(memory_ids))
        .values(superseded_by=None)
    )


async def _delete_env_rows(session: AsyncSession, env_id: UUID) -> dict[str, int]:
    counts = _zero_counts()
    outbox_ids = select(Outbox.event_id).where(Outbox.env_id == env_id)

    counts["outbox_delivery"] = await _delete_count(
        session,
        delete(OutboxDelivery).where(OutboxDelivery.event_id.in_(outbox_ids)),
    )
    counts["outbox"] = await _delete_count(
        session,
        delete(Outbox).where(Outbox.env_id == env_id),
    )
    counts["relations"] = await _delete_count(
        session,
        delete(Relation).where(Relation.env_id == env_id),
    )
    counts["memory_tags"] = await _delete_count(
        session,
        delete(MemoryTag).where(MemoryTag.env_id == env_id),
    )
    counts["memory_sources"] = await _delete_count(
        session,
        delete(MemorySource).where(MemorySource.memory_id.in_(_memory_ids_for_env(env_id))),
    )
    counts["memory_lineage"] = await _delete_count(
        session,
        delete(MemoryLineage).where(
            or_(
                MemoryLineage.parent_memory_id.in_(_memory_ids_for_env(env_id)),
                MemoryLineage.child_memory_id.in_(_memory_ids_for_env(env_id)),
            )
        ),
    )
    counts["graph_nodes"] = await _delete_count(
        session,
        delete(GraphNode).where(GraphNode.env_id == env_id),
    )
    counts["tasks"] = await _delete_count(
        session,
        delete(Task).where(Task.env_id == env_id),
    )
    counts["tags"] = await _delete_count(
        session,
        delete(Tag).where(Tag.env_id == env_id),
    )
    counts["entity_aliases"] = await _delete_count(
        session,
        delete(EntityAlias).where(EntityAlias.env_id == env_id),
    )
    counts["entities"] = await _delete_count(
        session,
        delete(Entity).where(Entity.env_id == env_id),
    )
    counts["memories"] = await _delete_count(
        session,
        delete(Memory).where(Memory.env_id == env_id),
    )
    counts["dream_proposals"] = await _delete_count(
        session,
        delete(DreamProposal).where(DreamProposal.env_id == env_id),
    )
    counts["dream_runs"] = await _delete_count(
        session,
        delete(DreamRun).where(DreamRun.env_id == env_id),
    )
    counts["env_grants"] = await _delete_count(
        session,
        delete(EnvGrant).where(EnvGrant.env_id == env_id),
    )
    counts["audit_log"] = await _delete_count(
        session,
        delete(AuditLog).where(AuditLog.env_id == env_id),
    )
    counts["snapshots"] = await _delete_snapshots_if_present(session, env_id)
    return counts


async def _delete_count(session: AsyncSession, stmt: Any) -> int:
    result = await session.execute(stmt)
    return int(result.rowcount or 0)


async def _delete_snapshots_if_present(session: AsyncSession, env_id: UUID) -> int:
    exists = await session.scalar(text("SELECT to_regclass('public.snapshots')"))
    if exists is None:
        return 0
    result = await session.execute(
        text("DELETE FROM snapshots WHERE env_id = :env_id"),
        {"env_id": env_id},
    )
    return int(result.rowcount or 0)


def _zero_counts() -> dict[str, int]:
    return {
        "outbox_delivery": 0,
        "outbox": 0,
        "relations": 0,
        "memory_tags": 0,
        "memory_sources": 0,
        "memory_lineage": 0,
        "graph_nodes": 0,
        "tasks": 0,
        "tags": 0,
        "entity_aliases": 0,
        "entities": 0,
        "memories": 0,
        "dream_proposals": 0,
        "dream_runs": 0,
        "env_grants": 0,
        "audit_log": 0,
        "snapshots": 0,
    }


__all__ = ["RefsBlockingDeleteError", "delete_env"]
