"""Domain implementations for task tree tools."""

from __future__ import annotations

import base64
import datetime as dt
import json
from typing import Any
from uuid import UUID

from sqlalchemy import and_, exists, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import GraphNode, Memory, Relation, Task
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import (
    MemoryStatus,
    OutboxAggregateType,
    OutboxOp,
    TaskRelationKind,
    TaskStatus,
    is_valid_task_transition,
)
from memory_mcp.errors import (
    CycleDetectedError,
    EnvNotAttachedError,
    InvalidCursorError,
    InvalidInputError,
    InvalidTransitionError,
    NotFoundError,
    VersionConflictError,
)
from memory_mcp.identity import AgentContext
from memory_mcp.tasks.cycles import would_cycle
from memory_mcp.tasks.models import (
    TaskCreateRequest,
    TaskLinkMemoryRequest,
    TaskLinkMemoryResponse,
    TaskListRequest,
    TaskListResponse,
    TaskRelationRequest,
    TaskRelationResponse,
    TaskResponse,
    TaskTreeLine,
    TaskTreeResponse,
)

__all__ = [
    "task_create",
    "task_dep_link",
    "task_link_memory",
    "task_list",
    "task_next",
    "task_status_set",
    "task_substep",
    "task_tree",
]

_TERMINAL_DEP_STATUSES = {TaskStatus.done.value, TaskStatus.cancelled.value}
_VISIBLE_MEMORY_STATUSES = {
    MemoryStatus.proposed.value,
    MemoryStatus.active.value,
    MemoryStatus.stale.value,
}


def _assert_env_visible(env_id: UUID, ctx: AgentContext) -> None:
    if ctx.attached_env_ids and env_id not in set(ctx.attached_env_ids):
        raise NotFoundError(f"env {env_id} not visible in attached envs", env_id=str(env_id))


def _require_env_attached(env_id: UUID, ctx: AgentContext) -> None:
    if env_id not in set(ctx.attached_env_ids):
        raise EnvNotAttachedError(
            f"ENV_NOT_ATTACHED: env {env_id} is not attached to this session",
            env_id=str(env_id),
            attached_env_ids=[str(e) for e in ctx.attached_env_ids],
        )


def _task_to_response(task: Task) -> TaskResponse:
    return TaskResponse(
        id=task.id,
        env_id=task.env_id,
        title=task.title,
        description=task.description,
        status=TaskStatus(task.status),
        priority=task.priority,
        playbook_id=task.playbook_id,
        version=task.version,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def _task_payload(task: Task) -> dict[str, Any]:
    return {
        "task_id": str(task.id),
        "env_id": str(task.env_id),
        "title": task.title,
        "description": task.description,
        "status": task.status,
        "priority": task.priority,
        "playbook_id": str(task.playbook_id) if task.playbook_id else None,
        "version": task.version,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
    }


def _relation_payload(relation: Relation, src_node: GraphNode, dst_node: GraphNode) -> dict[str, Any]:
    return {
        "relation_id": str(relation.id),
        "env_id": str(relation.env_id),
        "type": relation.type,
        "properties": dict(relation.properties or {}),
        "src": {
            "kind": src_node.node_type,
            "id": str(src_node.task_id or src_node.memory_id or src_node.entity_id),
            "node_id": str(src_node.id),
        },
        "dst": {
            "kind": dst_node.node_type,
            "id": str(dst_node.task_id or dst_node.memory_id or dst_node.entity_id),
            "node_id": str(dst_node.id),
        },
        "version": relation.version,
        "created_at": relation.created_at.isoformat() if relation.created_at else None,
        "updated_at": relation.updated_at.isoformat() if relation.updated_at else None,
    }


async def _load_task(session: AsyncSession, task_id: UUID) -> Task:
    task = (await session.execute(select(Task).where(Task.id == task_id))).scalar_one_or_none()
    if task is None:
        raise NotFoundError(f"task {task_id} not found", task_id=str(task_id))
    return task


async def _ensure_task_graph_node(session: AsyncSession, *, env_id: UUID, task_id: UUID) -> GraphNode:
    node = (await session.execute(
        select(GraphNode).where(GraphNode.env_id == env_id, GraphNode.task_id == task_id)
    )).scalar_one_or_none()
    if node is None:
        node = GraphNode(env_id=env_id, node_type="task", task_id=task_id)
        session.add(node)
        await session.flush()
        await session.refresh(node)
    return node


async def _ensure_memory_graph_node(session: AsyncSession, *, env_id: UUID, memory_id: UUID) -> GraphNode:
    node = (await session.execute(
        select(GraphNode).where(GraphNode.env_id == env_id, GraphNode.memory_id == memory_id)
    )).scalar_one_or_none()
    if node is None:
        node = GraphNode(env_id=env_id, node_type="memory", memory_id=memory_id)
        session.add(node)
        await session.flush()
        await session.refresh(node)
    return node


async def _acquire_dep_lock(session: AsyncSession, env_id: UUID) -> None:
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended('task_dep_link:' || :env_id, 0))"),
        {"env_id": str(env_id)},
    )


async def _insert_relation(
    session: AsyncSession,
    *,
    env_id: UUID,
    src_node: GraphNode,
    dst_node: GraphNode,
    relation_type: TaskRelationKind,
) -> tuple[Relation, bool]:
    existing = (await session.execute(
        select(Relation).where(
            Relation.src_node_id == src_node.id,
            Relation.dst_node_id == dst_node.id,
            Relation.type == relation_type.value,
        )
    )).scalar_one_or_none()
    if existing is not None:
        return existing, False
    relation = Relation(
        env_id=env_id,
        src_node_id=src_node.id,
        dst_node_id=dst_node.id,
        type=relation_type.value,
        properties={},
    )
    session.add(relation)
    await session.flush()
    await session.refresh(relation)
    return relation, True


async def _insert_relation_idempotent(
    session: AsyncSession,
    *,
    env_id: UUID,
    src_node: GraphNode,
    dst_node: GraphNode,
    relation_type: TaskRelationKind,
) -> tuple[Relation, bool]:
    stmt = (
        pg_insert(Relation)
        .values(
            env_id=env_id,
            src_node_id=src_node.id,
            dst_node_id=dst_node.id,
            type=relation_type.value,
            properties={},
        )
        .on_conflict_do_nothing(
            index_elements=["src_node_id", "dst_node_id", "type"],
        )
        .returning(Relation.id)
    )
    relation_id = (await session.execute(stmt)).scalar_one_or_none()
    if relation_id is not None:
        relation = await session.get(Relation, relation_id)
        if relation is None:
            raise NotFoundError(f"relation {relation_id} not found", relation_id=str(relation_id))
        return relation, True
    relation = (await session.execute(
        select(Relation).where(
            Relation.src_node_id == src_node.id,
            Relation.dst_node_id == dst_node.id,
            Relation.type == relation_type.value,
        )
    )).scalar_one()
    return relation, False


async def _enqueue_task(session: AsyncSession, task: Task, *, op: OutboxOp, settings: Settings) -> None:
    await enqueue_event(
        session,
        aggregate_type=OutboxAggregateType.task,
        aggregate_id=task.id,
        aggregate_version=task.version,
        env_id=task.env_id,
        op=op,
        payload=_task_payload(task),
        settings=settings,
    )


async def _enqueue_relation(
    session: AsyncSession,
    relation: Relation,
    src_node: GraphNode,
    dst_node: GraphNode,
    *,
    op: OutboxOp,
    settings: Settings,
) -> None:
    await enqueue_event(
        session,
        aggregate_type=OutboxAggregateType.relation,
        aggregate_id=relation.id,
        aggregate_version=relation.version,
        env_id=relation.env_id,
        op=op,
        payload=_relation_payload(relation, src_node, dst_node),
        settings=settings,
    )


async def task_create(
    request: TaskCreateRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> TaskResponse:
    settings = settings or get_settings()
    _assert_env_visible(request.env_id, ctx)
    rbac.require("write", request.env_id, ctx)

    async with session_scope() as session:
        if request.playbook_id is not None:
            playbook = (await session.execute(
                select(Memory).where(Memory.id == request.playbook_id)
            )).scalar_one_or_none()
            if playbook is None or playbook.env_id != request.env_id or playbook.kind != "playbook":
                raise InvalidInputError("playbook_id must reference a playbook memory in the same env")

        task = Task(
            env_id=request.env_id,
            title=request.title,
            description=request.description,
            priority=request.priority,
            playbook_id=request.playbook_id,
            created_by_agent_id=ctx.agent_id,
        )
        session.add(task)
        await session.flush()
        await session.refresh(task)
        await _ensure_task_graph_node(session, env_id=task.env_id, task_id=task.id)
        await _enqueue_task(session, task, op=OutboxOp.upsert, settings=settings)
    return _task_to_response(task)


async def task_substep(
    parent_task_id: UUID,
    *,
    title: str,
    description: str | None = None,
    priority: int = 50,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> TaskResponse:
    settings = settings or get_settings()
    if not 1 <= priority <= 100:
        raise InvalidInputError("priority must be between 1 and 100")

    async with session_scope() as session:
        parent = await _load_task(session, parent_task_id)
        _assert_env_visible(parent.env_id, ctx)
        rbac.require("write", parent.env_id, ctx)

        subtask = Task(
            env_id=parent.env_id,
            title=title,
            description=description,
            priority=priority,
            created_by_agent_id=ctx.agent_id,
        )
        session.add(subtask)
        await session.flush()
        await session.refresh(subtask)

        parent_node = await _ensure_task_graph_node(session, env_id=parent.env_id, task_id=parent.id)
        sub_node = await _ensure_task_graph_node(session, env_id=parent.env_id, task_id=subtask.id)
        await _acquire_dep_lock(session, parent.env_id)
        if await would_cycle(session, parent.env_id, parent.id, subtask.id):
            raise CycleDetectedError("task_substep would create a dependency cycle")
        relation, _ = await _insert_relation(
            session,
            env_id=parent.env_id,
            src_node=parent_node,
            dst_node=sub_node,
            relation_type=TaskRelationKind.depends_on,
        )
        await _enqueue_task(session, subtask, op=OutboxOp.upsert, settings=settings)
        await _enqueue_relation(session, relation, parent_node, sub_node, op=OutboxOp.upsert, settings=settings)
    return _task_to_response(subtask)


async def task_dep_link(
    request: TaskRelationRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> TaskRelationResponse:
    settings = settings or get_settings()
    if request.src_task_id == request.dst_task_id:
        raise CycleDetectedError("task cannot depend on itself")
    if request.type != TaskRelationKind.depends_on:
        raise InvalidInputError("task_dep_link only supports depends_on")

    async with session_scope() as session:
        src = await _load_task(session, request.src_task_id)
        dst = await _load_task(session, request.dst_task_id)
        if src.env_id != dst.env_id:
            raise InvalidInputError("task dependencies must stay within one env")
        _assert_env_visible(src.env_id, ctx)
        rbac.require("write", src.env_id, ctx)

        src_node = await _ensure_task_graph_node(session, env_id=src.env_id, task_id=src.id)
        dst_node = await _ensure_task_graph_node(session, env_id=src.env_id, task_id=dst.id)
        await _acquire_dep_lock(session, src.env_id)
        if await would_cycle(session, src.env_id, src.id, dst.id):
            raise CycleDetectedError("task_dep_link would create a dependency cycle")
        relation, created = await _insert_relation(
            session,
            env_id=src.env_id,
            src_node=src_node,
            dst_node=dst_node,
            relation_type=TaskRelationKind.depends_on,
        )
        if created:
            await _enqueue_relation(session, relation, src_node, dst_node, op=OutboxOp.upsert, settings=settings)
    return TaskRelationResponse(
        src_task_id=src.id,
        dst_task_id=dst.id,
        type=TaskRelationKind.depends_on,
        created_at=relation.created_at,
    )


async def task_status_set(
    task_id: UUID,
    *,
    status: TaskStatus,
    expected_version: int,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> TaskResponse:
    settings = settings or get_settings()
    async with session_scope() as session:
        task = await _load_task(session, task_id)
        _assert_env_visible(task.env_id, ctx)
        rbac.require("write", task.env_id, ctx)
        current = TaskStatus(task.status)
        if not is_valid_task_transition(current, status):
            raise InvalidTransitionError(current.value, status.value)
        if task.version != expected_version:
            raise VersionConflictError(expected=expected_version, actual=task.version)

        result = await session.execute(
            update(Task)
            .where(Task.id == task_id, Task.version == expected_version)
            .values(status=status.value, version=expected_version + 1, updated_at=func.now())
            .returning(Task)
        )
        updated = result.scalar_one_or_none()
        if updated is None:
            raise VersionConflictError(expected=expected_version, actual=expected_version + 1)
        await _enqueue_task(session, updated, op=OutboxOp.update, settings=settings)
    return _task_to_response(updated)


def _encode_cursor(task: Task) -> str:
    raw = json.dumps(
        {
            "priority": task.priority,
            "created_at": task.created_at.isoformat(),
            "id": str(task.id),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str) -> tuple[int, dt.datetime, UUID]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        return (
            int(payload["priority"]),
            dt.datetime.fromisoformat(payload["created_at"]),
            UUID(str(payload["id"])),
        )
    except Exception as exc:  # noqa: BLE001
        raise InvalidCursorError("invalid task_list cursor") from exc


async def task_list(
    request: TaskListRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> TaskListResponse:
    _ = settings or get_settings()
    _assert_env_visible(request.env_id, ctx)
    rbac.require("read", request.env_id, ctx)

    async with session_scope() as session:
        stmt = select(Task).where(Task.env_id == request.env_id)
        if request.status is not None:
            stmt = stmt.where(Task.status == request.status.value)
        if request.priority_max is not None:
            stmt = stmt.where(Task.priority <= request.priority_max)
        if request.cursor:
            c_priority, c_created_at, c_id = _decode_cursor(request.cursor)
            stmt = stmt.where(
                (Task.priority > c_priority)
                | and_(Task.priority == c_priority, Task.created_at > c_created_at)
                | and_(Task.priority == c_priority, Task.created_at == c_created_at, Task.id > c_id)
            )
        stmt = stmt.order_by(Task.priority.asc(), Task.created_at.asc(), Task.id.asc()).limit(request.limit + 1)
        rows = list((await session.execute(stmt)).scalars().all())

    has_more = len(rows) > request.limit
    rows = rows[:request.limit]
    return TaskListResponse(
        hits=[_task_to_response(t) for t in rows],
        next_cursor=_encode_cursor(rows[-1]) if has_more and rows else None,
    )


async def task_next(
    env_id: UUID,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> TaskResponse | None:
    _ = settings or get_settings()
    _assert_env_visible(env_id, ctx)
    rbac.require("read", env_id, ctx)
    src_node = aliased(GraphNode)
    dst_node = aliased(GraphNode)
    dep = aliased(Relation)
    dst_task = aliased(Task)

    async with session_scope() as session:
        stmt = (
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
                        dst_task.status.not_in(_TERMINAL_DEP_STATUSES),
                    )
                ),
            )
            .order_by(Task.priority.asc(), Task.created_at.asc(), Task.id.asc())
            .limit(1)
        )
        task = (await session.execute(stmt)).scalar_one_or_none()
    return _task_to_response(task) if task is not None else None


async def _task_tree_children(session: AsyncSession, *, env_id: UUID, task_id: UUID) -> list[Task]:
    src_node = aliased(GraphNode)
    dst_node = aliased(GraphNode)
    stmt = (
        select(Task)
        .join(dst_node, dst_node.task_id == Task.id)
        .join(Relation, Relation.dst_node_id == dst_node.id)
        .join(src_node, src_node.id == Relation.src_node_id)
        .where(
            Task.env_id == env_id,
            src_node.env_id == env_id,
            dst_node.env_id == env_id,
            Relation.env_id == env_id,
            src_node.node_type == "task",
            dst_node.node_type == "task",
            src_node.task_id == task_id,
            Relation.type == TaskRelationKind.depends_on.value,
        )
        .order_by(Task.priority.asc(), Task.created_at.asc(), Task.id.asc())
    )
    return list((await session.execute(stmt)).scalars().all())


def _task_tree_line(task: Task, depth: int) -> TaskTreeLine:
    return TaskTreeLine(
        depth=depth,
        task_id=task.id,
        status=str(task.status),
        desc=task.title,
        version=task.version,
    )


async def task_tree(
    task_id: UUID,
    *,
    ctx: AgentContext,
    max_depth: int = 10,
    max_nodes: int = 200,
) -> TaskTreeResponse:
    if max_depth < 0:
        raise InvalidInputError("max_depth must be non-negative")
    if max_nodes < 1:
        raise InvalidInputError("max_nodes must be at least 1")

    lines: list[TaskTreeLine] = []
    visited: set[UUID] = set()
    truncated = False

    async with session_scope() as session:
        root = await _load_task(session, task_id)
        _require_env_attached(root.env_id, ctx)
        rbac.require("read", root.env_id, ctx)

        async def visit(task: Task, depth: int) -> bool:
            nonlocal truncated
            if task.id in visited:
                return True
            if len(lines) >= max_nodes:
                truncated = True
                return False

            visited.add(task.id)
            lines.append(_task_tree_line(task, depth))

            children = await _task_tree_children(session, env_id=root.env_id, task_id=task.id)
            unvisited_children = [child for child in children if child.id not in visited]
            if depth >= max_depth:
                if unvisited_children:
                    truncated = True
                    return False
                return True

            for child in unvisited_children:
                if len(lines) >= max_nodes:
                    truncated = True
                    return False
                if not await visit(child, depth + 1):
                    return False
            return True

        await visit(root, 0)

    return TaskTreeResponse(
        root_id=root.id,
        lines=lines,
        truncated=truncated,
        total_visited=len(lines),
    )


async def task_link_memory(
    request: TaskLinkMemoryRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> TaskLinkMemoryResponse:
    settings = settings or get_settings()
    if request.relation not in {
        TaskRelationKind.motivated_by,
        TaskRelationKind.produces,
        TaskRelationKind.references,
    }:
        raise InvalidInputError("relation must be motivated_by, produces, or references")

    async with session_scope() as session:
        task = await _load_task(session, request.task_id)
        memory = (await session.execute(select(Memory).where(Memory.id == request.memory_id))).scalar_one_or_none()
        if memory is None:
            raise NotFoundError(f"memory {request.memory_id} not found", memory_id=str(request.memory_id))
        if task.env_id != memory.env_id:
            raise InvalidInputError("task and memory must be in the same env")
        if memory.status not in _VISIBLE_MEMORY_STATUSES:
            raise InvalidInputError("cannot link task to archived/retired memory")
        _assert_env_visible(task.env_id, ctx)
        rbac.require("write", task.env_id, ctx)

        task_node = await _ensure_task_graph_node(session, env_id=task.env_id, task_id=task.id)
        memory_node = await _ensure_memory_graph_node(session, env_id=task.env_id, memory_id=memory.id)
        relation, created = await _insert_relation_idempotent(
            session,
            env_id=task.env_id,
            src_node=task_node,
            dst_node=memory_node,
            relation_type=request.relation,
        )
        if created:
            await _enqueue_relation(session, relation, task_node, memory_node, op=OutboxOp.upsert, settings=settings)
    return TaskLinkMemoryResponse(
        relation_id=relation.id,
        task_id=task.id,
        memory_id=memory.id,
        relation=request.relation,
        created_at=relation.created_at,
    )
