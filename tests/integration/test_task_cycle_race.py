"""Real-Postgres race coverage for task dependency cycle hardening."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from memory_mcp.config import Settings
from memory_mcp.db.models import Environment, GraphNode, Relation, Task
from memory_mcp.db.types import TaskRelationKind
from memory_mcp.errors import CycleDetectedError
from memory_mcp.identity import AgentContext
from memory_mcp.tasks import api as task_api
from memory_mcp.tasks.models import TaskRelationRequest

from .conftest import Barrier, SessionPairFactory, reset_session_factory, routed_session_scope, use_session_factory

pytestmark = pytest.mark.integration


def _iterations() -> int:
    return int(os.environ.get("MEMORY_MCP_RACE_ITERATIONS", "20"))


def _settings() -> Settings:
    return Settings(graph_backend="postgres")


async def _create_env_with_tasks(factory, *, scenario: str, iteration: int):
    async with factory() as session:
        env = Environment(
            name=f"cycle-race-{scenario}-{iteration}-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        session.add(env)
        await session.flush()
        task_a = Task(env_id=env.id, title="A")
        task_b = Task(env_id=env.id, title="B")
        session.add_all([task_a, task_b])
        await session.flush()
        session.add_all(
            [
                GraphNode(env_id=env.id, node_type="task", task_id=task_a.id),
                GraphNode(env_id=env.id, node_type="task", task_id=task_b.id),
            ]
        )
        await session.commit()
        return env.id, task_a.id, task_b.id


async def _relation_count(factory, env_id) -> int:
    async with factory() as session:
        stmt = (
            select(func.count())
            .select_from(Relation)
            .where(
                Relation.env_id == env_id,
                Relation.type == TaskRelationKind.depends_on.value,
            )
        )
        return int((await session.execute(stmt)).scalar_one())


@pytest.mark.parametrize("scenario", ["plain", "reverse-call-order", "fresh-graph-nodes"])
@pytest.mark.asyncio
async def test_task_dep_link_cycle_race_allows_exactly_one(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
    scenario: str,
) -> None:
    current_barrier: dict[str, Barrier] = {}
    original_acquire_dep_lock = task_api._acquire_dep_lock  # noqa: SLF001

    async def gated_acquire_dep_lock(session, env_id):
        await current_barrier["barrier"].wait()
        await original_acquire_dep_lock(session, env_id)

    monkeypatch.setattr(task_api, "session_scope", routed_session_scope)
    monkeypatch.setattr(task_api, "_acquire_dep_lock", gated_acquire_dep_lock)

    factory_1, factory_2 = postgres_session_factories()
    for iteration in range(_iterations()):
        env_id, task_a, task_b = await _create_env_with_tasks(factory_1, scenario=scenario, iteration=iteration)
        ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])
        current_barrier["barrier"] = Barrier(2)

        async def link(factory, src_task_id, dst_task_id):
            token = use_session_factory(factory)
            try:
                return await task_api.task_dep_link(
                    TaskRelationRequest(src_task_id=src_task_id, dst_task_id=dst_task_id),
                    ctx=ctx,
                    settings=_settings(),
                )
            finally:
                reset_session_factory(token)

        left, right = ((task_a, task_b), (task_b, task_a))
        if scenario == "reverse-call-order":
            left, right = right, left
        results = await asyncio.gather(
            link(factory_1, *left),
            link(factory_2, *right),
            return_exceptions=True,
        )

        assert sum(not isinstance(result, Exception) for result in results) == 1
        assert sum(isinstance(result, CycleDetectedError) for result in results) == 1
        assert await _relation_count(factory_1, env_id) == 1
