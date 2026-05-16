"""Cycle detection for task dependency edges."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp.db.models import GraphNode, Relation


async def would_cycle(
    session: AsyncSession,
    env_id: UUID,
    src_task_id: UUID,
    dst_task_id: UUID,
) -> bool:
    """Return True if adding ``src_task_id --depends_on--> dst_task_id`` cycles.

    The caller must hold the per-env advisory transaction lock before calling.
    We walk outgoing ``depends_on`` dependencies from ``dst_task_id``; if the
    proposed source is reachable, inserting the proposed edge would close a cycle.
    """
    if src_task_id == dst_task_id:
        return True

    from sqlalchemy.orm import aliased

    src_node = aliased(GraphNode)
    dst_node = aliased(GraphNode)
    frontier: list[UUID] = [dst_task_id]
    seen: set[UUID] = set()
    while frontier:
        current = frontier.pop()
        if current == src_task_id:
            return True
        if current in seen:
            continue
        seen.add(current)
        rows = await session.execute(
            select(dst_node.task_id)
            .select_from(Relation)
            .join(src_node, src_node.id == Relation.src_node_id)
            .join(dst_node, dst_node.id == Relation.dst_node_id)
            .where(
                Relation.env_id == env_id,
                Relation.type == "depends_on",
                src_node.node_type == "task",
                dst_node.node_type == "task",
                src_node.task_id == current,
            )
        )
        for (next_task_id,) in rows.all():
            if next_task_id is not None and next_task_id not in seen:
                frontier.append(next_task_id)
    return False
