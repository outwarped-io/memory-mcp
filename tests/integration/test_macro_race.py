"""Real-Postgres race coverage for playbook macro uniqueness hardening."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from memory_mcp import memories as memories_mod
from memory_mcp.config import Settings
from memory_mcp.db.models import Agent, Environment, Memory
from memory_mcp.db.types import MemoryKind
from memory_mcp.errors import InvalidInputError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryWriteRequest, memory_write

from .conftest import Barrier, SessionPairFactory, reset_session_factory, routed_session_scope, use_session_factory

pytestmark = pytest.mark.integration


def _iterations() -> int:
    return int(os.environ.get("MEMORY_MCP_RACE_ITERATIONS", "20"))


def _settings() -> Settings:
    return Settings(graph_backend="postgres")


async def _create_env_and_agent(factory, *, scenario: str, iteration: int):
    async with factory() as session:
        env = Environment(
            name=f"macro-race-{scenario}-{iteration}-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"macro-race-agent-{iteration}")
        session.add_all([env, agent])
        await session.commit()
        return env.id, agent.id


async def _macro_count(factory, env_id, macro: str) -> int:
    async with factory() as session:
        stmt = (
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.env_id == env_id,
                Memory.kind == MemoryKind.playbook.value,
                func.lower(Memory.macro) == macro.lower(),
            )
        )
        return int((await session.execute(stmt)).scalar_one())


@pytest.mark.parametrize(
    ("scenario", "macro_1", "macro_2"),
    [
        ("same-lowercase", "release", "release"),
        ("case-insensitive", "Release", "release"),
        ("whitespace-normalized", " release ", "RELEASE"),
    ],
)
@pytest.mark.asyncio
async def test_playbook_macro_race_allows_exactly_one(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
    scenario: str,
    macro_1: str,
    macro_2: str,
) -> None:
    current_barrier: dict[str, Barrier] = {}
    original_ensure_macro_available = memories_mod._ensure_macro_available  # noqa: SLF001

    async def gated_ensure_macro_available(session, *, env_id, macro, exclude_memory_id=None):
        await original_ensure_macro_available(
            session,
            env_id=env_id,
            macro=macro,
            exclude_memory_id=exclude_memory_id,
        )
        await current_barrier["barrier"].wait()

    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    monkeypatch.setattr(memories_mod, "_ensure_macro_available", gated_ensure_macro_available)

    factory_1, factory_2 = postgres_session_factories()
    for iteration in range(_iterations()):
        env_id, agent_id = await _create_env_and_agent(factory_1, scenario=scenario, iteration=iteration)
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        current_barrier["barrier"] = Barrier(2)

        async def write_playbook(factory, macro: str):
            token = use_session_factory(factory)
            try:
                return await memory_write(
                    MemoryWriteRequest(
                        kind=MemoryKind.playbook,
                        title="Release runbook",
                        body="Run the release safely.",
                        env_id=env_id,
                        steps=["prepare", "deploy"],
                        macro=macro,
                    ),
                    ctx=ctx,
                    settings=_settings(),
                )
            finally:
                reset_session_factory(token)

        results = await asyncio.gather(
            write_playbook(factory_1, macro_1),
            write_playbook(factory_2, macro_2),
            return_exceptions=True,
        )

        assert sum(not isinstance(result, Exception) for result in results) == 1
        assert sum(isinstance(result, InvalidInputError) for result in results) == 1
        assert await _macro_count(factory_1, env_id, "release") == 1
