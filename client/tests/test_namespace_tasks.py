"""Happy-path coverage for the tasks namespace."""

from __future__ import annotations

from uuid import uuid4

import pytest

from memory_mcp_schemas.enums import TaskRelationKind, TaskStatus
from memory_mcp_schemas.tasks import (
    TaskCreateRequest,
    TaskLinkMemoryRequest,
    TaskLinkMemoryResponse,
    TaskListRequest,
    TaskListResponse,
    TaskRelationResponse,
    TaskResponse,
    TaskTreeResponse,
)
from tests.conftest import make_task_payload


pytestmark = pytest.mark.asyncio


def make_task_relation_payload(**overrides: object) -> dict[str, object]:
    """A minimal valid TaskRelationResponse payload."""
    base: dict[str, object] = {
        "src_task_id": str(uuid4()),
        "dst_task_id": str(uuid4()),
        "type": "depends_on",
        "created_at": "2026-05-13T00:00:00Z",
    }
    base.update(overrides)
    return base


def make_task_link_memory_payload(**overrides: object) -> dict[str, object]:
    """A minimal valid TaskLinkMemoryResponse payload."""
    base: dict[str, object] = {
        "relation_id": str(uuid4()),
        "task_id": str(uuid4()),
        "memory_id": str(uuid4()),
        "relation": "references",
        "created_at": "2026-05-13T00:00:00Z",
    }
    base.update(overrides)
    return base


async def test_create(client, fake_session) -> None:
    env_id = str(uuid4())
    payload = make_task_payload(env_id=env_id, title="fake")
    fake_session.set_response("task_create", payload)

    request = TaskCreateRequest(env_id=env_id, title="fake")
    out = await client.tasks.create(request)

    name, args = fake_session.calls[0]
    assert name == "task_create"
    assert isinstance(args["request"], dict)
    assert args["request"]["env_id"] == env_id
    assert args["request"]["title"] == "fake"
    assert isinstance(out, TaskResponse)


async def test_substep(client, fake_session) -> None:
    parent_task_id = str(uuid4())
    payload = make_task_payload(id=str(uuid4()), title="child", description="details", priority=25)
    fake_session.set_response("task_substep", payload)

    out = await client.tasks.substep(
        parent_task_id=parent_task_id,
        title="child",
        description="details",
        priority=25,
    )

    name, args = fake_session.calls[0]
    assert name == "task_substep"
    assert args == {
        "parent_task_id": parent_task_id,
        "title": "child",
        "description": "details",
        "priority": 25,
    }
    assert isinstance(out, TaskResponse)


async def test_dep_link(client, fake_session) -> None:
    src_task_id = str(uuid4())
    dst_task_id = str(uuid4())
    fake_session.set_response(
        "task_dep_link",
        make_task_relation_payload(src_task_id=src_task_id, dst_task_id=dst_task_id),
    )
    fake_session.set_response(
        "task_dep_link",
        make_task_relation_payload(src_task_id=src_task_id, dst_task_id=dst_task_id),
    )

    enum_relation = getattr(TaskRelationKind, "blocks", TaskRelationKind.depends_on)
    enum_out = await client.tasks.dep_link(src_task_id, dst_task_id, type=enum_relation)
    string_out = await client.tasks.dep_link(src_task_id, dst_task_id, type="blocks")

    enum_name, enum_args = fake_session.calls[0]
    string_name, string_args = fake_session.calls[1]
    assert enum_name == "task_dep_link"
    assert enum_args == {
        "src_task_id": src_task_id,
        "dst_task_id": dst_task_id,
        "type": enum_relation.value,
    }
    assert string_name == "task_dep_link"
    assert string_args == {
        "src_task_id": src_task_id,
        "dst_task_id": dst_task_id,
        "type": "blocks",
    }
    assert isinstance(enum_out, TaskRelationResponse)
    assert isinstance(string_out, TaskRelationResponse)


async def test_status_set(client, fake_session) -> None:
    task_id = str(uuid4())
    fake_session.set_response("task_status_set", make_task_payload(id=task_id, status="done"))
    fake_session.set_response("task_status_set", make_task_payload(id=task_id, status="done"))

    enum_out = await client.tasks.status_set(task_id, TaskStatus.done, expected_version=1)
    string_out = await client.tasks.status_set(task_id, "done", expected_version=2)

    enum_name, enum_args = fake_session.calls[0]
    string_name, string_args = fake_session.calls[1]
    assert enum_name == "task_status_set"
    assert enum_args == {
        "task_id": task_id,
        "status": "done",
        "expected_version": 1,
    }
    assert string_name == "task_status_set"
    assert string_args == {
        "task_id": task_id,
        "status": "done",
        "expected_version": 2,
    }
    assert isinstance(enum_out, TaskResponse)
    assert isinstance(string_out, TaskResponse)


async def test_list(client, fake_session) -> None:
    env_id = str(uuid4())
    fake_session.set_response(
        "task_list",
        {"hits": [make_task_payload(env_id=env_id)], "next_cursor": None},
    )

    request = TaskListRequest(env_id=env_id, status=TaskStatus.pending, limit=10)
    out = await client.tasks.list(request)

    name, args = fake_session.calls[0]
    assert name == "task_list"
    assert isinstance(args["request"], dict)
    assert args["request"]["env_id"] == env_id
    assert args["request"]["status"] == "pending"
    assert args["request"]["limit"] == 10
    assert isinstance(out, TaskListResponse)


async def test_next_returns_task(client, fake_session) -> None:
    env_id = str(uuid4())
    task_id = str(uuid4())
    fake_session.set_response("task_next", make_task_payload(id=task_id, env_id=env_id))

    out = await client.tasks.next(env_id)

    name, args = fake_session.calls[0]
    assert name == "task_next"
    assert args == {"env_id": env_id}
    assert isinstance(out, TaskResponse)
    assert str(out.id) == task_id


async def test_next_returns_none_when_no_task(client, fake_session) -> None:
    env_id = str(uuid4())
    fake_session.set_response("task_next", None)
    fake_session.set_response("task_next", {})

    first = await client.tasks.next(env_id)
    second = await client.tasks.next(env_id)

    assert first is None
    assert second is None
    assert fake_session.calls == [
        ("task_next", {"env_id": env_id}),
        ("task_next", {"env_id": env_id}),
    ]


async def test_tree(client, fake_session) -> None:
    task_id = str(uuid4())
    fake_session.set_response(
        "task_tree",
        {
            "root_id": task_id,
            "lines": [
                {
                    "depth": 0,
                    "task_id": task_id,
                    "status": "pending",
                    "desc": "fake-task",
                    "version": 1,
                }
            ],
            "truncated": False,
            "total_visited": 1,
        },
    )

    out = await client.tasks.tree(task_id=task_id, max_depth=3, max_nodes=5)

    name, args = fake_session.calls[0]
    assert name == "task_tree"
    assert args == {"task_id": task_id, "max_depth": 3, "max_nodes": 5}
    assert isinstance(out, TaskTreeResponse)


async def test_link_memory(client, fake_session) -> None:
    task_id = str(uuid4())
    memory_id = str(uuid4())
    fake_session.set_response(
        "task_link_memory",
        make_task_link_memory_payload(task_id=task_id, memory_id=memory_id),
    )

    request = TaskLinkMemoryRequest(
        task_id=task_id,
        memory_id=memory_id,
        relation=TaskRelationKind.references,
    )
    out = await client.tasks.link_memory(request)

    name, args = fake_session.calls[0]
    assert name == "task_link_memory"
    assert isinstance(args["request"], dict)
    assert args["request"] == {
        "task_id": task_id,
        "memory_id": memory_id,
        "relation": "references",
    }
    assert isinstance(out, TaskLinkMemoryResponse)
