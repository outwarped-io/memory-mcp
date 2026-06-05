"""R-B4 regression — access-bump must preserve citation contribution.

End-to-end coverage of the Phase 1e slice 1e-b' fix: when ``memory_get``
or ``memory_get_many`` runs the post-read access bump, the recomputed
salience must include the per-kind reference-count contribution. The
pre-fix code path constructed ``SalienceInputs`` without those fields,
silently letting them default to 0 — so every read on a cited memory
recomputed salience as if the citations did not exist.

Strategy: pin counter columns + initial salience via direct SQL (the
Postgres triggers wire the counter writes; we avoid setting up the full
``rel_link`` API surface for narrower assertions), then call the live
``memory_get`` and ``memory_get_many`` paths and assert the post-bump
salience is strictly greater than what an "uncited" baseline would
produce after the same bump.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from memory_mcp import memories as memories_mod
from memory_mcp.db.models import Agent, Environment
from memory_mcp.identity import AgentContext
from memory_mcp.memories import memory_get, memory_get_many

from .conftest import (
    SessionPairFactory,
    reset_session_factory,
    routed_session_scope,
    use_session_factory,
)

pytestmark = pytest.mark.integration


async def _seed_env_and_agent(factory) -> tuple[UUID, UUID]:
    async with factory() as session:
        env = Environment(
            name=f"refbump-{uuid4()}",
            kind="test",
            default_embedding_model_id="test-embedding",
        )
        agent = Agent(id=uuid4(), name=f"refbump-agent-{uuid4()}")
        session.add_all([env, agent])
        await session.commit()
        return env.id, agent.id


async def _seed_memory_with_citations(
    factory,
    env_id: UUID,
    *,
    rl: int = 0,
    ln: int = 0,
    tk: int = 0,
    pb: int = 0,
    initial_salience: float = 0.5,
) -> UUID:
    """Direct-SQL seed: populate counter columns + a known starting salience.

    Bypasses the trigger path so we can pin the exact pre-bump state
    without setting up the full ``relations`` / ``memory_lineage`` edge
    fixtures. The triggers' correctness is exercised in
    ``test_reference_counts.py`` — this test is narrowly about whether
    ``memory_get``'s bump-path preserves whatever counters are stored.
    """
    mem_id = uuid4()
    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO memories ("
                "id, env_id, kind, status, body, salience, "
                "reference_count_rel_link, reference_count_lineage, "
                "reference_count_task, reference_count_playbook"
                ") VALUES ("
                ":id, :env_id, 'fact', 'active', 'body', :sal, "
                ":rl, :ln, :tk, :pb"
                ")"
            ),
            {
                "id": mem_id,
                "env_id": env_id,
                "sal": Decimal(str(initial_salience)),
                "rl": rl,
                "ln": ln,
                "tk": tk,
                "pb": pb,
            },
        )
        await session.commit()
    return mem_id


@pytest.mark.asyncio
async def test_memory_get_bump_preserves_citation_contribution(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``memory_get(bump_access=True)`` must include refs in recomputed salience.

    Pre-fix (Phase 1 / pre slice 1e-b'): the access-bump code path
    constructed ``SalienceInputs`` without ``reference_count_*`` →
    citation contribution dropped to zero on every read.
    """
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _seed_env_and_agent(factory)

    # Two memories: identical except one carries citations.
    cited_id = await _seed_memory_with_citations(factory, env_id, rl=20, ln=2, tk=5, pb=3, initial_salience=0.0)
    uncited_id = await _seed_memory_with_citations(factory, env_id, rl=0, ln=0, tk=0, pb=0, initial_salience=0.0)

    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        cited = await memory_get(cited_id, ctx=ctx, bump_access=True)
        uncited = await memory_get(uncited_id, ctx=ctx, bump_access=True)
    finally:
        reset_session_factory(token)

    # Both took the bump path; counters survived. The cited memory's
    # recomputed salience must reflect the references-term contribution.
    # Margin floor 0.05 — comfortably below w_references=0.15 envelope
    # but well above numerical noise.
    delta = float(cited.salience) - float(uncited.salience)
    assert delta >= 0.05, (
        "R-B4 regression: memory_get access-bump dropped citation "
        f"contribution. cited.salience={cited.salience}, "
        f"uncited.salience={uncited.salience}, delta={delta:.4f}"
    )
    # Sanity: the API should also expose the live counter on the response.
    assert cited.reference_count == 30  # 20 + 2 + 5 + 3
    assert uncited.reference_count == 0


@pytest.mark.asyncio
async def test_memory_get_many_bump_preserves_citation_contribution(
    monkeypatch: pytest.MonkeyPatch,
    postgres_session_factories: SessionPairFactory,
    clean_db: None,
) -> None:
    """``memory_get_many(bump_access=True)`` must include refs in recomputed salience.

    Same invariant as the single-get path, exercising the bulk loop at
    ``memories.py:1348``.
    """
    monkeypatch.setattr(memories_mod, "session_scope", routed_session_scope)
    factory, _ = postgres_session_factories()
    env_id, agent_id = await _seed_env_and_agent(factory)

    cited_id = await _seed_memory_with_citations(factory, env_id, rl=20, ln=2, tk=5, pb=3, initial_salience=0.0)
    uncited_id = await _seed_memory_with_citations(factory, env_id, rl=0, ln=0, tk=0, pb=0, initial_salience=0.0)

    token = use_session_factory(factory)
    try:
        ctx = AgentContext(agent_id=agent_id, attached_env_ids=[env_id])
        resp = await memory_get_many([cited_id, uncited_id], ctx=ctx, bump_access=True)
    finally:
        reset_session_factory(token)

    by_id = {m.id: m for m in resp}
    cited = by_id[cited_id]
    uncited = by_id[uncited_id]
    delta = float(cited.salience) - float(uncited.salience)
    assert delta >= 0.05, (
        "R-B4 regression: memory_get_many bulk access-bump dropped "
        f"citation contribution. cited.salience={cited.salience}, "
        f"uncited.salience={uncited.salience}, delta={delta:.4f}"
    )
    assert cited.reference_count == 30
    assert uncited.reference_count == 0
