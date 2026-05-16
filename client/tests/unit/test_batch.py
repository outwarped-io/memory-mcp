"""Unit tests for SDK batch helpers."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from memory_mcp_client import BatchFailure, BatchResult
from memory_mcp_client._batch import run_bounded
from memory_mcp_schemas.entities import EntityUpsertRequest
from memory_mcp_schemas.memories import MemoryWriteRequest
from memory_mcp_schemas.tasks import TaskCreateRequest
from tests.conftest import make_memory_payload, make_task_payload

pytestmark = pytest.mark.asyncio


async def test_run_bounded_happy_path_preserves_success_order() -> None:
    items = [0, 1, 2, 3, 4]

    async def handler(item: int) -> str:
        await asyncio.sleep(0.01 * (len(items) - item))
        return f"ok-{item}"

    out = await run_bounded(items, handler, max_concurrency=3)

    assert out.successes == ["ok-0", "ok-1", "ok-2", "ok-3", "ok-4"]
    assert out.failures == []
    assert out.success_count == 5
    assert out.failure_count == 0
    assert out.is_partial is False


async def test_run_bounded_all_failures_capture_indices() -> None:
    items = [0, 1, 2, 3, 4]

    async def handler(item: int) -> str:
        raise RuntimeError(f"boom-{item}")

    out = await run_bounded(items, handler)

    assert out.successes == []
    assert out.success_count == 0
    assert out.failure_count == 5
    assert [failure.index for failure in out.failures] == [0, 1, 2, 3, 4]
    assert [failure.item for failure in out.failures] == items
    assert [str(failure.exception) for failure in out.failures] == [
        "boom-0",
        "boom-1",
        "boom-2",
        "boom-3",
        "boom-4",
    ]


async def test_run_bounded_mixed_results_report_partial() -> None:
    items = [0, 1, 2, 3, 4]

    async def handler(item: int) -> str:
        if item in {1, 3}:
            raise ValueError(f"bad-{item}")
        return f"ok-{item}"

    out = await run_bounded(items, handler)

    assert out.successes == ["ok-0", "ok-2", "ok-4"]
    assert out.success_count == 3
    assert out.failure_count == 2
    assert out.is_partial is True
    assert [failure.index for failure in out.failures] == [1, 3]
    assert [failure.item for failure in out.failures] == [1, 3]
    assert [str(failure.exception) for failure in out.failures] == ["bad-1", "bad-3"]


async def test_run_bounded_respects_max_concurrency() -> None:
    items = list(range(6))
    active = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def handler(item: int) -> int:
        nonlocal active, max_seen
        async with lock:
            active += 1
            max_seen = max(max_seen, active)
        await asyncio.sleep(0.02)
        async with lock:
            active -= 1
        return item

    out = await run_bounded(items, handler, max_concurrency=2)

    assert out.successes == items
    assert out.failures == []
    assert max_seen <= 2


async def test_run_bounded_empty_input_returns_empty_result() -> None:
    out = await run_bounded([], lambda _item: asyncio.sleep(0))

    assert isinstance(out, BatchResult)
    assert out.successes == []
    assert out.failures == []
    assert out.success_count == 0
    assert out.failure_count == 0
    assert out.is_partial is False


async def test_memories_write_many_invokes_single_write(client, fake_session) -> None:
    env_id = uuid4()
    items = [
        MemoryWriteRequest(env_id=env_id, kind="fact", title="one", body="1"),
        MemoryWriteRequest(env_id=env_id, kind="fact", title="two", body="2"),
        MemoryWriteRequest(env_id=env_id, kind="fact", title="three", body="3"),
    ]
    fake_session.set_response("mem_write", make_memory_payload(title="one", body="1"))
    fake_session.set_error("mem_write", "rate_limited: slow down")
    fake_session.set_response("mem_write", make_memory_payload(title="three", body="3"))

    out = await client.memories.write_many(items, max_concurrency=1)

    assert [name for name, _ in fake_session.calls] == ["mem_write", "mem_write", "mem_write"]
    assert out.success_count == 2
    assert out.failure_count == 1
    assert [memory.title for memory in out.successes] == ["one", "three"]
    assert out.failures[0].index == 1
    assert out.failures[0].item == items[1]


async def test_entities_upsert_many_invokes_single_upsert(client, fake_session) -> None:
    items = [
        EntityUpsertRequest(kind="service", canonical_name="service-one"),
        EntityUpsertRequest(kind="service", canonical_name="service-two"),
        EntityUpsertRequest(kind="service", canonical_name="service-three"),
    ]
    fake_session.set_response("ent_upsert", _entity_payload(canonical_name="service-one"))
    fake_session.set_error("ent_upsert", "rate_limited: slow down")
    fake_session.set_response("ent_upsert", _entity_payload(canonical_name="service-three"))

    out = await client.entities.upsert_many(items, max_concurrency=1)

    assert [name for name, _ in fake_session.calls] == ["ent_upsert", "ent_upsert", "ent_upsert"]
    assert out.success_count == 2
    assert out.failure_count == 1
    assert [entity.canonical_name for entity in out.successes] == [
        "service-one",
        "service-three",
    ]
    assert out.failures[0].index == 1
    assert out.failures[0].item == items[1]


async def test_tasks_create_many_invokes_single_create(client, fake_session) -> None:
    env_id = uuid4()
    items = [
        TaskCreateRequest(env_id=env_id, title="task-one"),
        TaskCreateRequest(env_id=env_id, title="task-two"),
        TaskCreateRequest(env_id=env_id, title="task-three"),
    ]
    fake_session.set_response("task_create", make_task_payload(title="task-one", env_id=str(env_id)))
    fake_session.set_error("task_create", "rate_limited: slow down")
    fake_session.set_response("task_create", make_task_payload(title="task-three", env_id=str(env_id)))

    out = await client.tasks.create_many(items, max_concurrency=1)

    assert [name for name, _ in fake_session.calls] == ["task_create", "task_create", "task_create"]
    assert out.success_count == 2
    assert out.failure_count == 1
    assert [task.title for task in out.successes] == ["task-one", "task-three"]
    assert out.failures[0].index == 1
    assert out.failures[0].item == items[1]
    assert isinstance(out.failures[0], BatchFailure)


def _entity_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "00000000-0000-0000-0000-00000000e101",
        "env_id": "00000000-0000-0000-0000-0000000000e0",
        "kind": "service",
        "canonical_name": "service-one",
        "normalized_name": "service-one",
        "aliases": [],
        "metadata": {},
        "version": 1,
        "created_at": "2026-05-13T00:00:00Z",
        "updated_at": "2026-05-13T00:00:00Z",
    }
    payload.update(overrides)
    return payload
