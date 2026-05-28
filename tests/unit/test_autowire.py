"""Unit tests for ``memory_mcp.autowire`` (Phase 4 D6a).

Pure-function tests that don't require a live Postgres session — we
exercise the early-skip logic + Stage-A ranking by mocking the session
+ embedder + vector store. Full end-to-end coverage lands in
``tests/integration/test_autowire_compose.py`` (D6b).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from memory_mcp.autowire import (
    AUTO_WIRE_PREDICATE,
    _should_skip_target,
    autowire_fetch_candidates,
)
from memory_mcp.config import Settings
from memory_mcp.db.types import MemoryKind


# ---------------------------------------------------------------------------
# Predicate constant
# ---------------------------------------------------------------------------


def test_predicate_value_locked_in():
    """The literal ``related_to_popular`` is referenced by migration
    0017's trigger guard + recount.py + top.py + the workspace vocab
    in §10.3. Renaming requires migrations + an instructions edit."""
    assert AUTO_WIRE_PREDICATE == "related_to_popular"


# ---------------------------------------------------------------------------
# Early-skip filter
# ---------------------------------------------------------------------------


def test_skip_empty_body():
    assert _should_skip_target(kind=MemoryKind.fact, tags=None, body="") is True


def test_skip_whitespace_body():
    assert _should_skip_target(kind=MemoryKind.fact, tags=None, body="   \n\t") is True


def test_skip_kind_playbook():
    assert _should_skip_target(
        kind=MemoryKind.playbook, tags=None, body="hello world"
    ) is True


def test_skip_kind_playbook_via_string_value():
    assert _should_skip_target(
        kind="playbook", tags=None, body="hello world"
    ) is True


def test_skip_tag_directive_active():
    assert _should_skip_target(
        kind=MemoryKind.fact,
        tags=["topic:foo", "directive:active"],
        body="hello world",
    ) is True


def test_skip_tag_directive_active_prefix_match():
    """``directive:active:some-slug`` also triggers skip."""
    assert _should_skip_target(
        kind=MemoryKind.fact,
        tags=["directive:active:cosmos"],
        body="hello world",
    ) is True


def test_no_skip_normal_fact():
    assert _should_skip_target(
        kind=MemoryKind.fact, tags=["topic:foo"], body="hello world"
    ) is False


def test_no_skip_directive_retired_unrelated_prefix():
    assert _should_skip_target(
        kind=MemoryKind.fact,
        tags=["directive:retired"],
        body="hello world",
    ) is False


def test_no_skip_empty_tags():
    assert _should_skip_target(
        kind=MemoryKind.fact, tags=[], body="hello world"
    ) is False


def test_no_skip_none_tags():
    assert _should_skip_target(
        kind=MemoryKind.fact, tags=None, body="hello world"
    ) is False


# ---------------------------------------------------------------------------
# Stage A — feature OFF / early skip
# ---------------------------------------------------------------------------


def _settings(*, enabled: bool = True, top_k: int = 3, threshold: float = 0.5):
    """Minimal settings override that bypasses env-loading."""
    return Settings(
        autowire_enabled=enabled,
        autowire_top_k=top_k,
        autowire_sim_threshold=threshold,
        autowire_candidate_limit=20,
    )


async def test_stage_a_returns_empty_when_feature_disabled():
    s = AsyncMock()
    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[uuid4(), uuid4()],
        body="some body",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(enabled=False),
    )
    assert out == []
    s.execute.assert_not_called()


async def test_stage_a_returns_empty_when_skip_filter_matches():
    s = AsyncMock()
    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[],
        body="",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(),
    )
    assert out == []
    s.execute.assert_not_called()


async def test_stage_a_returns_empty_when_no_pg_candidates():
    s = AsyncMock()
    # First execute (PG candidate query) returns no rows.
    result = MagicMock()
    result.all.return_value = []
    s.execute.return_value = result
    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(),
    )
    assert out == []


async def test_stage_a_embedder_failure_degrades_to_empty():
    """Embedder raising an exception must NEVER block compose."""
    s = AsyncMock()
    result = MagicMock()
    result.all.return_value = [(uuid4(), 0.9)]
    s.execute.return_value = result

    bad_embedder = MagicMock()
    bad_embedder.embed_texts = MagicMock(side_effect=RuntimeError("embedder down"))

    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(),
        embedder=bad_embedder,
        vector_store=MagicMock(),
    )
    assert out == []


async def test_stage_a_vector_store_failure_degrades_to_empty():
    s = AsyncMock()
    result = MagicMock()
    result.all.return_value = [(uuid4(), 0.9)]
    s.execute.return_value = result

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3]])

    vector_store = MagicMock()
    vector_store.search = AsyncMock(side_effect=RuntimeError("qdrant down"))

    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert out == []


async def test_stage_a_threshold_cutoff_drops_low_similarity():
    """Candidates below ``autowire_sim_threshold`` must be excluded."""
    s = AsyncMock()
    candidate_id = uuid4()
    result = MagicMock()
    result.all.return_value = [(candidate_id, 0.9)]
    s.execute.return_value = result

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3]])

    vector_store = MagicMock()
    vector_store.search = AsyncMock(
        return_value=[{"id": str(candidate_id), "score": 0.30}]
    )

    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(threshold=0.70),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert out == []


async def test_stage_a_combined_ranking_picks_top_k():
    """``salience * sim_score`` ordering with deterministic tie-break."""
    s = AsyncMock()
    ids = [uuid4() for _ in range(5)]

    # All candidates pulled from PG with the salience values below.
    pg_rows = [
        (ids[0], 0.9),
        (ids[1], 0.8),
        (ids[2], 0.5),
        (ids[3], 0.4),
        (ids[4], 0.1),
    ]
    result = MagicMock()
    result.all.return_value = pg_rows
    s.execute.return_value = result

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3]])

    # Qdrant scores — chosen so the combined-score ordering is
    # well-defined and the cutoff at top_k=3 admits exactly ids[0..2].
    sem_scores = {
        ids[0]: 0.90,  # combined = 0.81
        ids[1]: 0.80,  # combined = 0.64
        ids[2]: 0.95,  # combined = 0.475
        ids[3]: 0.95,  # combined = 0.38
        ids[4]: 0.99,  # combined = 0.099
    }
    vector_store = MagicMock()
    vector_store.search = AsyncMock(
        return_value=[
            {"id": str(mid), "score": score}
            for mid, score in sem_scores.items()
        ]
    )

    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(top_k=3, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    out_ids = [mid for mid, _ in out]
    assert out_ids == [ids[0], ids[1], ids[2]]
    # Combined-score values returned to the caller for forensics.
    assert [round(score, 4) for _, score in out] == [0.81, 0.64, 0.475]


async def test_stage_a_excludes_source_ids():
    """Even if a source memory is itself popular, it must not be wired."""
    src_id = uuid4()
    other_id = uuid4()

    pg_result = MagicMock()
    pg_result.all.return_value = [(src_id, 0.99), (other_id, 0.5)]
    # Second execute is the lineage-ancestors CTE; src_id seeds it but
    # no parents exist, so only src_id comes back (it's always in the
    # seed set per the CTE definition).
    lineage_result = MagicMock()
    lineage_result.all.return_value = [(src_id,)]

    s = AsyncMock()
    s.execute.side_effect = [pg_result, lineage_result]

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1]])

    vector_store = MagicMock()
    vector_store.search = AsyncMock(
        return_value=[
            {"id": str(src_id), "score": 0.99},
            {"id": str(other_id), "score": 0.99},
        ]
    )

    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[src_id],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(top_k=3, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    out_ids = [mid for mid, _ in out]
    assert src_id not in out_ids
    assert other_id in out_ids


async def test_stage_a_excludes_lineage_ancestors():
    """A popular memory that is an ancestor of a source must be skipped."""
    ancestor_id = uuid4()
    src_id = uuid4()
    other_id = uuid4()

    pg_result = MagicMock()
    pg_result.all.return_value = [(ancestor_id, 0.9), (other_id, 0.5)]
    # Lineage CTE: src_id seeded, ancestor_id is its parent.
    lineage_result = MagicMock()
    lineage_result.all.return_value = [(src_id,), (ancestor_id,)]

    s = AsyncMock()
    s.execute.side_effect = [pg_result, lineage_result]

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1]])

    vector_store = MagicMock()
    vector_store.search = AsyncMock(
        return_value=[
            {"id": str(ancestor_id), "score": 0.99},
            {"id": str(other_id), "score": 0.99},
        ]
    )

    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[src_id],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(top_k=3, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    out_ids = [mid for mid, _ in out]
    assert ancestor_id not in out_ids
    assert other_id in out_ids


async def test_stage_a_invalid_qdrant_id_silently_skipped():
    """A malformed hit must not crash the helper."""
    s = AsyncMock()
    good_id = uuid4()
    result = MagicMock()
    result.all.return_value = [(good_id, 0.5)]
    s.execute.return_value = result

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1]])

    vector_store = MagicMock()
    vector_store.search = AsyncMock(
        return_value=[
            {"id": "not-a-uuid", "score": 0.99},
            {"id": str(good_id), "score": 0.99},
        ]
    )

    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(top_k=3, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    out_ids = [mid for mid, _ in out]
    assert out_ids == [good_id]


async def test_stage_a_qdrant_empty_response_returns_empty():
    s = AsyncMock()
    result = MagicMock()
    result.all.return_value = [(uuid4(), 0.9)]
    s.execute.return_value = result

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1]])

    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[])

    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert out == []


async def test_stage_a_embedder_returns_empty_vectors_returns_empty():
    s = AsyncMock()
    result = MagicMock()
    result.all.return_value = [(uuid4(), 0.9)]
    s.execute.return_value = result

    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[]])  # empty vector

    out = await autowire_fetch_candidates(
        s=s,
        env_id=uuid4(),
        source_ids=[],
        body="hello",
        new_kind=MemoryKind.fact,
        new_tags=None,
        settings=_settings(),
        embedder=embedder,
        vector_store=MagicMock(),
    )
    assert out == []
