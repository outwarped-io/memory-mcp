"""Unit coverage for v0.7 task-tree primitives."""

from __future__ import annotations

import asyncio
import datetime as dt
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from memory_mcp.db.models import GraphNode, Relation, Task
from memory_mcp.db.types import (
    TaskRelationKind,
    TaskStatus,
    is_valid_task_transition,
)
from memory_mcp.errors import CycleDetectedError, EnvNotAttachedError, InvalidInputError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.tasks import api as task_api
from memory_mcp.tasks.cycles import would_cycle
from memory_mcp.tasks.models import (
    TaskCreateRequest,
    TaskLinkMemoryRequest,
    TaskListRequest,
    TaskRelationRequest,
    TaskTreeLine,
)


def _task(task_id: UUID, env_id: UUID, title: str = "task") -> Task:
    return Task(
        id=task_id,
        env_id=env_id,
        title=title,
        description=None,
        status=TaskStatus.pending.value,
        priority=50,
        version=1,
        created_at=dt.datetime(2026, 5, 12, tzinfo=dt.UTC),
        updated_at=dt.datetime(2026, 5, 12, tzinfo=dt.UTC),
    )


def test_task_create_request_defaults() -> None:
    env = uuid4()
    req = TaskCreateRequest(env_id=env, title="Ship B1")
    assert req.env_id == env
    assert req.priority == 50
    assert req.description is None
    assert req.playbook_id is None


def test_task_create_rejects_priority_out_of_range() -> None:
    with pytest.raises(ValidationError):
        TaskCreateRequest(env_id=uuid4(), title="x", priority=0)
    with pytest.raises(ValidationError):
        TaskCreateRequest(env_id=uuid4(), title="x", priority=101)


def test_task_list_limit_bounds() -> None:
    TaskListRequest(env_id=uuid4(), limit=1)
    TaskListRequest(env_id=uuid4(), limit=100)
    with pytest.raises(ValidationError):
        TaskListRequest(env_id=uuid4(), limit=0)
    with pytest.raises(ValidationError):
        TaskListRequest(env_id=uuid4(), limit=101)


def test_task_link_memory_relation_allowlist() -> None:
    TaskLinkMemoryRequest(task_id=uuid4(), memory_id=uuid4(), relation=TaskRelationKind.produces)
    with pytest.raises(ValidationError):
        TaskLinkMemoryRequest(
            task_id=uuid4(),
            memory_id=uuid4(),
            relation=TaskRelationKind.depends_on,
        )


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (TaskStatus.pending, TaskStatus.in_progress),
        (TaskStatus.pending, TaskStatus.blocked),
        (TaskStatus.pending, TaskStatus.cancelled),
        (TaskStatus.in_progress, TaskStatus.blocked),
        (TaskStatus.in_progress, TaskStatus.done),
        (TaskStatus.in_progress, TaskStatus.cancelled),
        (TaskStatus.in_progress, TaskStatus.pending),
        (TaskStatus.blocked, TaskStatus.pending),
        (TaskStatus.blocked, TaskStatus.in_progress),
        (TaskStatus.blocked, TaskStatus.cancelled),
    ],
)
def test_task_status_valid_transition_matrix(src: TaskStatus, dst: TaskStatus) -> None:
    assert is_valid_task_transition(src, dst) is True


@pytest.mark.parametrize("terminal", [TaskStatus.done, TaskStatus.cancelled])
@pytest.mark.parametrize("dst", list(TaskStatus))
def test_task_terminal_statuses_do_not_transition(terminal: TaskStatus, dst: TaskStatus) -> None:
    assert is_valid_task_transition(terminal, dst) is (terminal == dst)


def test_task_pending_cannot_go_directly_done() -> None:
    assert is_valid_task_transition(TaskStatus.pending, TaskStatus.done) is False


def test_task_cursor_round_trip() -> None:
    task = _task(uuid4(), uuid4())
    cursor = task_api._encode_cursor(task)  # noqa: SLF001
    assert task_api._decode_cursor(cursor) == (task.priority, task.created_at, task.id)  # noqa: SLF001


def test_task_payload_shape() -> None:
    task = _task(uuid4(), uuid4(), title="B1")
    payload = task_api._task_payload(task)  # noqa: SLF001
    assert payload["task_id"] == str(task.id)
    assert payload["title"] == "B1"
    assert payload["status"] == "pending"
    assert payload["priority"] == 50
    assert payload["version"] == 1


def _task_with_priority(
    task_id: UUID,
    env_id: UUID,
    title: str,
    *,
    priority: int,
    created_at: dt.datetime,
) -> Task:
    task = _task(task_id, env_id, title)
    task.priority = priority
    task.created_at = created_at
    task.updated_at = created_at
    return task


def _patch_task_tree(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tasks: dict[UUID, Task],
    edges: dict[UUID, list[UUID]],
) -> None:
    @asynccontextmanager
    async def fake_scope():
        yield _FakeSession()

    async def fake_load(_session, task_id):
        task = tasks.get(task_id)
        if task is None:
            raise NotFoundError(f"task {task_id} not found", task_id=str(task_id))
        return task

    async def fake_children(_session, *, env_id, task_id):
        children = [
            tasks[child_id]
            for child_id in edges.get(task_id, [])
            if tasks[child_id].env_id == env_id
        ]
        return sorted(children, key=lambda task: (task.priority, task.created_at, task.id))

    monkeypatch.setattr(task_api, "session_scope", fake_scope)
    monkeypatch.setattr(task_api, "_load_task", fake_load)
    monkeypatch.setattr(task_api, "_task_tree_children", fake_children)


@pytest.mark.asyncio
async def test_task_tree_empty_subtree(monkeypatch: pytest.MonkeyPatch) -> None:
    env = uuid4()
    root = _task(uuid4(), env, "root")
    _patch_task_tree(monkeypatch, tasks={root.id: root}, edges={})

    out = await task_api.task_tree(
        root.id,
        ctx=AgentContext(agent_id=uuid4(), attached_env_ids=[env]),
    )

    assert out.root_id == root.id
    assert out.lines == [
        TaskTreeLine(
            depth=0,
            task_id=root.id,
            status=TaskStatus.pending.value,
            desc="root",
            version=1,
        )
    ]
    assert out.truncated is False
    assert out.total_visited == 1


@pytest.mark.asyncio
async def test_task_tree_depth_cap_truncates_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    env = uuid4()
    tasks = {}
    ids = [uuid4() for _ in range(5)]
    for idx, task_id in enumerate(ids):
        tasks[task_id] = _task(task_id, env, f"t{idx}")
    edges = {ids[idx]: [ids[idx + 1]] for idx in range(len(ids) - 1)}
    _patch_task_tree(monkeypatch, tasks=tasks, edges=edges)

    out = await task_api.task_tree(
        ids[0],
        ctx=AgentContext(agent_id=uuid4(), attached_env_ids=[env]),
        max_depth=2,
    )

    assert [line.task_id for line in out.lines] == ids[:3]
    assert [line.depth for line in out.lines] == [0, 1, 2]
    assert out.truncated is True
    assert out.total_visited == 3


@pytest.mark.asyncio
async def test_task_tree_node_cap_truncates_in_deterministic_order(monkeypatch: pytest.MonkeyPatch) -> None:
    env = uuid4()
    now = dt.datetime(2026, 5, 12, tzinfo=dt.UTC)
    root = _task(uuid4(), env, "root")
    children = [
        _task_with_priority(
            uuid4(),
            env,
            f"child-{idx}",
            priority=priority,
            created_at=now + dt.timedelta(seconds=idx),
        )
        for idx, priority in enumerate([50, 10, 10, 30, 20, 70, 60, 40, 90, 80])
    ]
    tasks = {root.id: root, **{child.id: child for child in children}}
    _patch_task_tree(
        monkeypatch,
        tasks=tasks,
        edges={root.id: [child.id for child in children]},
    )

    out = await task_api.task_tree(
        root.id,
        ctx=AgentContext(agent_id=uuid4(), attached_env_ids=[env]),
        max_nodes=3,
    )

    expected_children = sorted(children, key=lambda task: (task.priority, task.created_at, task.id))[:2]
    assert [line.task_id for line in out.lines] == [
        root.id,
        expected_children[0].id,
        expected_children[1].id,
    ]
    assert [line.depth for line in out.lines] == [0, 1, 1]
    assert out.truncated is True
    assert out.total_visited == 3


@pytest.mark.asyncio
async def test_task_tree_rejects_unattached_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    env = uuid4()
    root = _task(uuid4(), env, "root")
    _patch_task_tree(monkeypatch, tasks={root.id: root}, edges={})

    with pytest.raises(EnvNotAttachedError):
        await task_api.task_tree(
            root.id,
            ctx=AgentContext(agent_id=uuid4(), attached_env_ids=[uuid4()]),
        )


def test_task_relation_payload_orientation() -> None:
    env = uuid4()
    src_id, dst_id = uuid4(), uuid4()
    src_node = GraphNode(id=uuid4(), env_id=env, node_type="task", task_id=src_id)
    dst_node = GraphNode(id=uuid4(), env_id=env, node_type="task", task_id=dst_id)
    rel = Relation(
        id=uuid4(),
        env_id=env,
        src_node_id=src_node.id,
        dst_node_id=dst_node.id,
        type=TaskRelationKind.depends_on.value,
        properties={},
        version=1,
        created_at=dt.datetime(2026, 5, 12, tzinfo=dt.UTC),
        updated_at=dt.datetime(2026, 5, 12, tzinfo=dt.UTC),
    )
    payload = task_api._relation_payload(rel, src_node, dst_node)  # noqa: SLF001
    assert payload["src"]["kind"] == "task"
    assert payload["src"]["id"] == str(src_id)
    assert payload["dst"]["kind"] == "task"
    assert payload["dst"]["id"] == str(dst_id)
    assert payload["type"] == "depends_on"


def test_env_visibility_rejects_attached_other_env() -> None:
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[uuid4()])
    with pytest.raises(NotFoundError):
        task_api._assert_env_visible(uuid4(), ctx)  # noqa: SLF001


@pytest.mark.asyncio
async def test_would_cycle_self_loop_is_true() -> None:
    task_id = uuid4()
    assert await would_cycle(object(), uuid4(), task_id, task_id) is True  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_task_dep_link_self_loop_rejected_before_db() -> None:
    task_id = uuid4()
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[uuid4()])
    with pytest.raises(CycleDetectedError):
        await task_api.task_dep_link(
            TaskRelationRequest(src_task_id=task_id, dst_task_id=task_id),
            ctx=ctx,
        )


@pytest.mark.asyncio
async def test_task_dep_link_rejects_non_depends_on_before_db() -> None:
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[uuid4()])
    with pytest.raises(InvalidInputError):
        await task_api.task_dep_link(
            TaskRelationRequest(
                src_task_id=uuid4(),
                dst_task_id=uuid4(),
                type=TaskRelationKind.references,
            ),
            ctx=ctx,
        )


@pytest.mark.asyncio
async def test_task_substep_priority_guard_before_db() -> None:
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[uuid4()])
    with pytest.raises(InvalidInputError):
        await task_api.task_substep(uuid4(), title="x", priority=0, ctx=ctx)


class _FakeSession:
    pass


class _FakeMemory:
    def __init__(self, *, memory_id: UUID, env_id: UUID) -> None:
        self.id = memory_id
        self.env_id = env_id
        self.status = "active"


class _FakeResult:
    def __init__(self, scalar: object | None = None) -> None:
        self.scalar = scalar

    def scalar_one_or_none(self) -> object | None:
        return self.scalar


def _install_fake_dep_link(monkeypatch: pytest.MonkeyPatch, *, cycle_on: set[tuple[UUID, UUID]] | None = None):
    env = uuid4()
    a, b, c = uuid4(), uuid4(), uuid4()
    tasks = {a: _task(a, env, "A"), b: _task(b, env, "B"), c: _task(c, env, "C")}
    edges: set[tuple[UUID, UUID]] = set()
    lock = asyncio.Lock()

    @asynccontextmanager
    async def fake_scope():
        try:
            yield _FakeSession()
        finally:
            if lock.locked():
                lock.release()

    async def fake_load(_session, task_id):
        return tasks[task_id]

    async def fake_node(_session, *, env_id, task_id):
        return GraphNode(id=task_id, env_id=env_id, node_type="task", task_id=task_id)

    async def fake_lock(_session, env_id):
        await lock.acquire()

    async def fake_cycle(_session, env_id, src_task_id, dst_task_id):
        if cycle_on and (src_task_id, dst_task_id) in cycle_on:
            return True
        frontier = [dst_task_id]
        seen = set()
        while frontier:
            cur = frontier.pop()
            if cur == src_task_id:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            frontier.extend(dst for src, dst in edges if src == cur)
        return False

    async def fake_insert(_session, *, env_id, src_node, dst_node, relation_type):
        edges.add((src_node.task_id, dst_node.task_id))
        rel = Relation(
            id=uuid4(),
            env_id=env_id,
            src_node_id=src_node.id,
            dst_node_id=dst_node.id,
            type=relation_type.value,
            properties={},
            version=1,
            created_at=dt.datetime.now(dt.UTC),
            updated_at=dt.datetime.now(dt.UTC),
        )
        return rel, True

    async def fake_enqueue(*args, **kwargs):
        return None

    monkeypatch.setattr(task_api, "session_scope", fake_scope)
    monkeypatch.setattr(task_api, "_load_task", fake_load)
    monkeypatch.setattr(task_api, "_ensure_task_graph_node", fake_node)
    monkeypatch.setattr(task_api, "_acquire_dep_lock", fake_lock)
    monkeypatch.setattr(task_api, "would_cycle", fake_cycle)
    monkeypatch.setattr(task_api, "_insert_relation", fake_insert)
    monkeypatch.setattr(task_api, "_enqueue_relation", fake_enqueue)
    return env, a, b, c, edges


@pytest.mark.asyncio
async def test_direct_cycle_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    env, a, b, _c, _edges = _install_fake_dep_link(monkeypatch)
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env])
    await task_api.task_dep_link(TaskRelationRequest(src_task_id=a, dst_task_id=b), ctx=ctx)
    with pytest.raises(CycleDetectedError):
        await task_api.task_dep_link(TaskRelationRequest(src_task_id=b, dst_task_id=a), ctx=ctx)


@pytest.mark.asyncio
async def test_indirect_cycle_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    env, a, b, c, _edges = _install_fake_dep_link(monkeypatch)
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env])
    await task_api.task_dep_link(TaskRelationRequest(src_task_id=a, dst_task_id=b), ctx=ctx)
    await task_api.task_dep_link(TaskRelationRequest(src_task_id=b, dst_task_id=c), ctx=ctx)
    with pytest.raises(CycleDetectedError):
        await task_api.task_dep_link(TaskRelationRequest(src_task_id=c, dst_task_id=a), ctx=ctx)


@pytest.mark.asyncio
async def test_concurrent_racing_cycle_allows_exactly_one(monkeypatch: pytest.MonkeyPatch) -> None:
    env, a, b, _c, _edges = _install_fake_dep_link(monkeypatch)
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env])
    results = await asyncio.gather(
        task_api.task_dep_link(TaskRelationRequest(src_task_id=a, dst_task_id=b), ctx=ctx),
        task_api.task_dep_link(TaskRelationRequest(src_task_id=b, dst_task_id=a), ctx=ctx),
        return_exceptions=True,
    )
    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert sum(isinstance(r, CycleDetectedError) for r in results) == 1


@pytest.mark.asyncio
async def test_task_substep_rolls_back_on_relation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    env, parent_id = uuid4(), uuid4()
    created: list[UUID] = []
    rolled_back = False

    @asynccontextmanager
    async def fake_scope():
        nonlocal rolled_back
        try:
            yield _FakeSession()
        except RuntimeError:
            rolled_back = True
            created.clear()
            raise

    async def fake_load(_session, task_id):
        return _task(parent_id, env, "parent")

    async def fake_node(_session, *, env_id, task_id):
        return GraphNode(id=task_id, env_id=env_id, node_type="task", task_id=task_id)

    async def fake_lock(_session, env_id):
        return None

    async def fake_cycle(*_args):
        return False

    class _Session(_FakeSession):
        def add(self, obj):
            created.append(obj.id)

        async def flush(self):
            if created and created[-1] is None:
                created[-1] = uuid4()

        async def refresh(self, obj):
            obj.id = obj.id or uuid4()
            obj.created_at = dt.datetime.now(dt.UTC)
            obj.updated_at = obj.created_at

    @asynccontextmanager
    async def fake_scope_with_session():
        nonlocal rolled_back
        try:
            yield _Session()
        except RuntimeError:
            rolled_back = True
            created.clear()
            raise

    async def failing_insert(*_args, **_kwargs):
        raise RuntimeError("relation insert failed")

    monkeypatch.setattr(task_api, "session_scope", fake_scope_with_session)
    monkeypatch.setattr(task_api, "_load_task", fake_load)
    monkeypatch.setattr(task_api, "_ensure_task_graph_node", fake_node)
    monkeypatch.setattr(task_api, "_acquire_dep_lock", fake_lock)
    monkeypatch.setattr(task_api, "would_cycle", fake_cycle)
    monkeypatch.setattr(task_api, "_insert_relation", failing_insert)
    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env])
    with pytest.raises(RuntimeError):
        await task_api.task_substep(parent_id, title="sub", ctx=ctx)
    assert rolled_back is True
    assert created == []


@pytest.mark.asyncio
async def test_task_link_memory_repeated_identical_call_returns_same_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = uuid4()
    task_id = uuid4()
    memory_id = uuid4()
    relation_id = uuid4()
    created_at = dt.datetime(2026, 5, 12, tzinfo=dt.UTC)
    enqueue_count = 0
    insert_count = 0

    class _Session(_FakeSession):
        async def execute(self, _stmt):
            return _FakeResult(_FakeMemory(memory_id=memory_id, env_id=env))

    @asynccontextmanager
    async def fake_scope():
        yield _Session()

    async def fake_load(_session, _task_id):
        return _task(task_id, env, "task")

    async def fake_node(_session, *, env_id, task_id=None, memory_id=None):
        node_id = task_id or memory_id
        return GraphNode(
            id=node_id,
            env_id=env_id,
            node_type="task" if task_id else "memory",
            task_id=task_id,
            memory_id=memory_id,
        )

    async def fake_insert(_session, *, env_id, src_node, dst_node, relation_type):
        nonlocal insert_count
        insert_count += 1
        return (
            Relation(
                id=relation_id,
                env_id=env_id,
                src_node_id=src_node.id,
                dst_node_id=dst_node.id,
                type=relation_type.value,
                properties={},
                version=1,
                created_at=created_at,
                updated_at=created_at,
            ),
            insert_count == 1,
        )

    async def fake_enqueue(*_args, **_kwargs):
        nonlocal enqueue_count
        enqueue_count += 1

    monkeypatch.setattr(task_api, "session_scope", fake_scope)
    monkeypatch.setattr(task_api, "_load_task", fake_load)
    monkeypatch.setattr(task_api, "_ensure_task_graph_node", fake_node)
    monkeypatch.setattr(task_api, "_ensure_memory_graph_node", fake_node)
    monkeypatch.setattr(task_api, "_insert_relation_idempotent", fake_insert)
    monkeypatch.setattr(task_api, "_enqueue_relation", fake_enqueue)

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env])
    request = TaskLinkMemoryRequest(
        task_id=task_id,
        memory_id=memory_id,
        relation=TaskRelationKind.motivated_by,
    )

    first = await task_api.task_link_memory(request, ctx=ctx)
    second = await task_api.task_link_memory(request, ctx=ctx)

    assert first.relation_id == relation_id
    assert second.relation_id == relation_id
    assert first.created_at == second.created_at == created_at
    assert enqueue_count == 1
