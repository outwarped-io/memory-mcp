"""Real-Postgres race coverage for ``_ensure_graph_node`` idempotency.

Reproduces ``memory-mcp-gap-rel-link-fan-out-race`` — concurrent
``rel_link`` calls fanning out from the same src memory both SELECT
``None`` from ``graph_nodes`` then race on INSERT against the partial
unique index ``graph_nodes_memory_uniq``. Without the savepoint guard
the loser's flush raises ``IntegrityError`` and ``rel_link`` returns it
as an unhandled error.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from memory_mcp_schemas.relations import RelationEndpoint, RelationLinkRequest
from sqlalchemy import func, select

from memory_mcp import relations as relations_mod
from memory_mcp.config import Settings
from memory_mcp.db.models import Agent, Environment, GraphNode, Memory
from memory_mcp.db.types import MemoryKind
from memory_mcp.identity import AgentContext
from memory_mcp.relations import relation_link

from .conftest import (
    Barrier,
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


def _iterations() -> int:
    return int(os.environ.get("MEMORY_MCP_RACE_ITERATIONS", "20"))


def _settings() -> Settings:
    return Settings(graph_backend="postgres")


async def _create_env_agent_and_memories(factory, *, iteration: int):
    """Seed an env, an agent, one src memory, and two dst memories."""

    async with factory() as session:
        env = Environment(
            name=f"rel-link-race-{iteration}-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"rel-link-race-agent-{iteration}")
        session.add_all([env, agent])
        await session.flush()

        src = Memory(
            env_id=env.id,
            kind=MemoryKind.fact.value,
            title="src",
            body="src body",
        )
        dst_a = Memory(
            env_id=env.id,
            kind=MemoryKind.fact.value,
            title="dst a",
            body="dst a body",
        )
        dst_b = Memory(
            env_id=env.id,
            kind=MemoryKind.fact.value,
            title="dst b",
            body="dst b body",
        )
        session.add_all([src, dst_a, dst_b])
        await session.commit()
        return env.id, agent.id, src.id, dst_a.id, dst_b.id


async def _graph_node_count_for_memory(factory, memory_id) -> int:
    async with factory() as session:
        stmt = select(func.count()).select_from(GraphNode).where(
            GraphNode.memory_id == memory_id,
        )
        return int((await session.execute(stmt)).scalar_one())


@pytest.mark.asyncio
async def test_rel_link_fan_out_shared_src_memory_no_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Two parallel ``rel_link`` calls share the same src memory endpoint.

    Both reach ``_ensure_graph_node`` for the src, both SELECT ``None``
    from ``graph_nodes``, both attempt to INSERT a row keyed on the same
    ``memory_id``. Without the savepoint-protected helper one would
    raise ``IntegrityError`` against ``graph_nodes_memory_uniq``. With
    the fix, the loser's INSERT rolls back to the savepoint, the
    candidate is expunged, the re-SELECT returns the winner's row, and
    both ``rel_link`` calls succeed with a single graph_nodes row for
    the shared src memory.
    """

    current_barrier: dict[str, Barrier] = {}
    original_create = relations_mod._create_or_get_graph_node  # noqa: SLF001

    async def gated_create_or_get(session, *, candidate, re_select_stmt):
        # Wait for both calls to arrive at the INSERT point before either
        # of them attempts the flush. Only gate the new-memory candidate
        # for the shared src; the dst nodes don't share a memory_id and
        # don't need gating.
        if candidate.node_type == "memory" and candidate.memory_id is not None:
            barrier = current_barrier.get(str(candidate.memory_id))
            if barrier is not None:
                await barrier.wait()
        return await original_create(
            session, candidate=candidate, re_select_stmt=re_select_stmt
        )

    monkeypatch.setattr(relations_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(relations_mod, "_create_or_get_graph_node", gated_create_or_get)

    factory_1, factory_2 = postgres_session_factories()
    for iteration in range(_iterations()):
        env_id, agent_id, src_id, dst_a_id, dst_b_id = (
            await _create_env_agent_and_memories(factory_1, iteration=iteration)
        )
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        # Both gated calls share the src memory id; gate on it.
        current_barrier.clear()
        current_barrier[str(src_id)] = Barrier(2)

        async def link(factory, dst_id, *, src_id=src_id, env_id=env_id, ctx=ctx):
            token = use_session_factory(factory)
            try:
                return await relation_link(
                    RelationLinkRequest(
                        src=RelationEndpoint(kind="memory", id=src_id),
                        dst=RelationEndpoint(kind="memory", id=dst_id),
                        type="references",
                        properties={},
                        env_id=env_id,
                    ),
                    ctx=ctx,
                    settings=_settings(),
                )
            finally:
                reset_session_factory(token)

        results = await asyncio.gather(
            link(factory_1, dst_a_id),
            link(factory_2, dst_b_id),
            return_exceptions=True,
        )

        # Both calls must succeed. Without the fix, one (the loser) would
        # carry an IntegrityError against graph_nodes_memory_uniq.
        for r in results:
            assert not isinstance(r, Exception), f"unexpected exception: {r!r}"

        # Exactly one graph_nodes row for the shared src memory id.
        assert await _graph_node_count_for_memory(factory_1, src_id) == 1
        # And each dst memory got its own graph_nodes row.
        assert await _graph_node_count_for_memory(factory_1, dst_a_id) == 1
        assert await _graph_node_count_for_memory(factory_1, dst_b_id) == 1


@pytest.mark.asyncio
async def test_rel_link_fan_out_three_way_shared_src(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """Three parallel ``rel_link`` calls share the same src memory.

    Closer match to the original 4-call symptom report. One winner;
    two losers must both recover via expunge+re-SELECT.
    """

    current_barrier: dict[str, Barrier] = {}
    original_create = relations_mod._create_or_get_graph_node  # noqa: SLF001

    async def gated_create_or_get(session, *, candidate, re_select_stmt):
        if candidate.node_type == "memory" and candidate.memory_id is not None:
            barrier = current_barrier.get(str(candidate.memory_id))
            if barrier is not None:
                await barrier.wait()
        return await original_create(
            session, candidate=candidate, re_select_stmt=re_select_stmt
        )

    monkeypatch.setattr(relations_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(relations_mod, "_create_or_get_graph_node", gated_create_or_get)

    factory_1, factory_2 = postgres_session_factories()
    # Need a third factory for the third concurrent caller. Build one
    # ad-hoc via the pair-factory (each call returns a fresh pair from a
    # fresh engine pair).
    factory_3, _factory_4 = postgres_session_factories()

    for iteration in range(_iterations()):
        env_id, agent_id, src_id, dst_a_id, dst_b_id = (
            await _create_env_agent_and_memories(factory_1, iteration=iteration)
        )
        # Add a third dst.
        async with factory_1() as session:
            dst_c = Memory(
                env_id=env_id,
                kind=MemoryKind.fact.value,
                title="dst c",
                body="dst c body",
            )
            session.add(dst_c)
            await session.commit()
            dst_c_id = dst_c.id

        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        current_barrier.clear()
        current_barrier[str(src_id)] = Barrier(3)

        async def link(factory, dst_id, *, src_id=src_id, env_id=env_id, ctx=ctx):
            token = use_session_factory(factory)
            try:
                return await relation_link(
                    RelationLinkRequest(
                        src=RelationEndpoint(kind="memory", id=src_id),
                        dst=RelationEndpoint(kind="memory", id=dst_id),
                        type="references",
                        properties={},
                        env_id=env_id,
                    ),
                    ctx=ctx,
                    settings=_settings(),
                )
            finally:
                reset_session_factory(token)

        results = await asyncio.gather(
            link(factory_1, dst_a_id),
            link(factory_2, dst_b_id),
            link(factory_3, dst_c_id),
            return_exceptions=True,
        )

        for r in results:
            assert not isinstance(r, Exception), f"unexpected exception: {r!r}"

        assert await _graph_node_count_for_memory(factory_1, src_id) == 1
