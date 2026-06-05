"""Unit tests for v0.16 decompose Stage A — ``autowire_fetch_candidates_decompose``.

Pure-function tests with mocked session + embedder + vector store, mirroring
``tests/unit/test_autowire.py`` (Phase 4 D6a). End-to-end coverage with real
Postgres + Qdrant testcontainers lands in
``tests/integration/test_autowire_decompose.py`` (Stage H5.3).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from memory_mcp.autowire import (
    autowire_fetch_candidates_decompose,
    reconstruct_auto_wired_by_child,
)
from memory_mcp.config import Settings
from memory_mcp.db.types import MemoryKind

# ---------------------------------------------------------------------------
# Settings helper
# ---------------------------------------------------------------------------


def _settings(
    *,
    master: bool = True,
    decompose: bool = True,
    per_child_k: int = 3,
    total_cap: int = 30,
    threshold: float = 0.5,
    candidate_limit: int = 20,
) -> Settings:
    return Settings(
        autowire_enabled=master,
        autowire_top_k=1,  # unused by decompose path; tightest valid
        autowire_sim_threshold=threshold,
        autowire_candidate_limit=candidate_limit,
        autowire_decompose_enabled=decompose,
        autowire_decompose_per_child_top_k=per_child_k,
        autowire_decompose_total_cap=total_cap,
    )


def _child(idx: int, body: str = "child body", kind=MemoryKind.fact, tags=None):
    return {
        "index": idx,
        "body": body,
        "kind": kind,
        "tags": tags if tags is not None else [],
    }


def _pg_result(rows):
    r = MagicMock()
    r.all.return_value = rows
    return r


def _lineage_result(rows):
    r = MagicMock()
    r.all.return_value = rows
    return r


# ---------------------------------------------------------------------------
# 1, 2 — Feature OFF (master or per-decompose)
# ---------------------------------------------------------------------------


async def test_decompose_feature_off_master_returns_empty_dict() -> None:
    """Master switch OFF short-circuits before any DB / embedder call."""
    s = AsyncMock()
    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=uuid4(),
        children=[_child(0), _child(1)],
        # Both off to satisfy the cross-knob invariant
        # (master OFF + decompose ON is rejected by Settings validator).
        settings=_settings(master=False, decompose=False),
    )
    assert out == {}
    s.execute.assert_not_called()


async def test_decompose_feature_off_decompose_only_returns_empty_dict() -> None:
    """Master ON, decompose OFF → also short-circuits."""
    s = AsyncMock()
    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=uuid4(),
        children=[_child(0), _child(1)],
        settings=_settings(master=True, decompose=False),
    )
    assert out == {}
    s.execute.assert_not_called()


# ---------------------------------------------------------------------------
# 3 — Batched embed called once for N children
# ---------------------------------------------------------------------------


async def test_decompose_batched_embed_called_once_for_n_children() -> None:
    src_id = uuid4()
    s = AsyncMock()
    # PG candidate query then lineage CTE; both empty so we can isolate
    # the embed-call assertion without needing Qdrant scores.
    s.execute.side_effect = [
        _pg_result([(uuid4(), 0.9), (uuid4(), 0.5)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[])

    await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[
            _child(0, body="c0"),
            _child(1, body="c1"),
            _child(2, body="c2"),
        ],
        settings=_settings(),
        embedder=embedder,
        vector_store=vector_store,
    )

    # ONE embed call carrying ALL three bodies.
    assert embedder.embed_texts.call_count == 1
    args, _ = embedder.embed_texts.call_args
    assert list(args[0]) == ["c0", "c1", "c2"]


# ---------------------------------------------------------------------------
# 4 — Parallel Qdrant searches via asyncio.gather
# ---------------------------------------------------------------------------


async def test_decompose_parallel_qdrant_searches_with_gather() -> None:
    src_id = uuid4()
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(uuid4(), 0.9)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1], [0.2], [0.3]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[])

    await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[_child(0), _child(1), _child(2)],
        settings=_settings(),
        embedder=embedder,
        vector_store=vector_store,
    )

    # ONE search call PER child (3 total) — Stage A5 fans out via gather.
    assert vector_store.search.await_count == 3


# ---------------------------------------------------------------------------
# 5 — Shared PG pull: one SELECT for many children
# ---------------------------------------------------------------------------


async def test_decompose_shared_pg_pull_one_query() -> None:
    """The candidate set is identical for every child, so we should
    issue exactly ONE PG SELECT (plus one lineage CTE)."""
    src_id = uuid4()
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(uuid4(), 0.9)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1], [0.2], [0.3]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[])

    await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[_child(0), _child(1), _child(2)],
        settings=_settings(),
        embedder=embedder,
        vector_store=vector_store,
    )

    # Exactly 2 calls: 1 PG candidate pull + 1 lineage CTE.
    assert s.execute.await_count == 2


# ---------------------------------------------------------------------------
# 6 — Lineage seed includes ONLY the source, not children
# ---------------------------------------------------------------------------


async def test_decompose_lineage_seed_includes_source_only() -> None:
    """The lineage-ancestor exclusion CTE is seeded with [source_id]
    (not with child ids, because children don't exist yet pre-txn)."""
    ancestor_id = uuid4()
    src_id = uuid4()
    other_id = uuid4()

    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(ancestor_id, 0.9), (other_id, 0.5)]),
        _lineage_result([(src_id,), (ancestor_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1], [0.2]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(
        return_value=[
            {"id": str(ancestor_id), "score": 0.99},
            {"id": str(other_id), "score": 0.99},
        ]
    )

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[_child(0), _child(1)],
        settings=_settings(per_child_k=3, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )

    # Source's ancestor must be excluded from EVERY child's result.
    for idx in (0, 1):
        out_ids = [mid for mid, _ in out[idx]]
        assert ancestor_id not in out_ids
        assert other_id in out_ids


# ---------------------------------------------------------------------------
# 7 — Per-child top-K respected
# ---------------------------------------------------------------------------


async def test_decompose_per_child_top_k_respected() -> None:
    """5 candidates, per_child_k=2, single child → exactly 2 returned."""
    src_id = uuid4()
    ids = [uuid4() for _ in range(5)]
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(mid, 0.9 - i * 0.05) for i, mid in enumerate(ids)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[{"id": str(mid), "score": 0.95} for mid in ids])

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[_child(0)],
        settings=_settings(per_child_k=2, total_cap=10, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert len(out[0]) == 2


# ---------------------------------------------------------------------------
# 8 — Total cap downsamples globally
# ---------------------------------------------------------------------------


async def test_decompose_total_cap_downsamples_globally() -> None:
    """per_child_k=5, total_cap=8, 3 children, all 15 candidates above
    threshold → 5+5+5 pre-cap, 8 post-cap (global downsample)."""
    src_id = uuid4()
    ids = [uuid4() for _ in range(15)]
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(mid, 0.9 - i * 0.01) for i, mid in enumerate(ids)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1], [0.2], [0.3]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[{"id": str(mid), "score": 0.95} for mid in ids])

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[_child(0), _child(1), _child(2)],
        settings=_settings(per_child_k=5, total_cap=8, threshold=0.5, candidate_limit=15),
        embedder=embedder,
        vector_store=vector_store,
    )
    total = sum(len(v) for v in out.values())
    assert total == 8


# ---------------------------------------------------------------------------
# 9 — Skip kind=playbook per-child (others succeed)
# ---------------------------------------------------------------------------


async def test_decompose_skip_kind_playbook_per_child() -> None:
    """One child with kind=playbook → its entry is []; others process."""
    src_id = uuid4()
    cid = uuid4()
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(cid, 0.9)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    # Only 1 surviving body → 1 vector returned.
    embedder.embed_texts = MagicMock(return_value=[[0.1]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[{"id": str(cid), "score": 0.95}])

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[
            _child(0, kind=MemoryKind.playbook, body="pb"),
            _child(1, body="ok"),
        ],
        settings=_settings(per_child_k=3, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert out[0] == []
    assert len(out[1]) == 1
    # Embedder receives only the surviving child's body.
    args, _ = embedder.embed_texts.call_args
    assert list(args[0]) == ["ok"]


# ---------------------------------------------------------------------------
# 10 — Skip tag=directive:active per-child
# ---------------------------------------------------------------------------


async def test_decompose_skip_tag_directive_active_per_child() -> None:
    src_id = uuid4()
    cid = uuid4()
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(cid, 0.9)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[{"id": str(cid), "score": 0.95}])

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[
            _child(0, tags=["directive:active:my-slug"], body="dir"),
            _child(1, body="ok"),
        ],
        settings=_settings(per_child_k=3, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert out[0] == []
    assert len(out[1]) == 1


# ---------------------------------------------------------------------------
# 11 — Skip empty body per-child
# ---------------------------------------------------------------------------


async def test_decompose_skip_empty_body_per_child() -> None:
    src_id = uuid4()
    cid = uuid4()
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(cid, 0.9)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[{"id": str(cid), "score": 0.95}])

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[
            _child(0, body="   "),
            _child(1, body="ok"),
        ],
        settings=_settings(per_child_k=3, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert out[0] == []
    assert len(out[1]) == 1


# ---------------------------------------------------------------------------
# 12 — Embedder failure degrades batch to empty dict
# ---------------------------------------------------------------------------


async def test_decompose_embedder_failure_degrades_to_empty_dict_top_level() -> None:
    src_id = uuid4()
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(uuid4(), 0.9)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(side_effect=RuntimeError("embedder down"))

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[_child(0), _child(1)],
        settings=_settings(),
        embedder=embedder,
        vector_store=MagicMock(),
    )
    assert out == {}


# ---------------------------------------------------------------------------
# 13 — Per-child Qdrant failure isolated to that child
# ---------------------------------------------------------------------------


async def test_decompose_qdrant_per_child_failure_degrades_that_child_only() -> None:
    """One Qdrant search raises; that child gets []; others succeed."""
    src_id = uuid4()
    cid = uuid4()
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(cid, 0.9)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1], [0.2]])

    call_count = {"n": 0}

    async def _flaky_search(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient qdrant error")
        return [{"id": str(cid), "score": 0.95}]

    vector_store = MagicMock()
    vector_store.search = AsyncMock(side_effect=_flaky_search)

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[_child(0), _child(1)],
        settings=_settings(per_child_k=3, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert out[0] == []
    assert len(out[1]) == 1


# ---------------------------------------------------------------------------
# 14 — Combined-score ranking deterministic
# ---------------------------------------------------------------------------


async def test_decompose_combined_score_ranking_deterministic() -> None:
    src_id = uuid4()
    ids = [uuid4() for _ in range(3)]
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result(
            [
                (ids[0], 0.9),
                (ids[1], 0.5),
                (ids[2], 0.3),
            ]
        ),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(
        return_value=[
            {"id": str(ids[0]), "score": 0.90},
            {"id": str(ids[1]), "score": 0.80},
            {"id": str(ids[2]), "score": 0.95},
        ]
    )

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[_child(0)],
        settings=_settings(per_child_k=3, total_cap=10, threshold=0.5),
        embedder=embedder,
        vector_store=vector_store,
    )
    # combined: ids[0] 0.81, ids[1] 0.40, ids[2] 0.285
    out_ids = [mid for mid, _ in out[0]]
    assert out_ids == [ids[0], ids[1], ids[2]]


# ---------------------------------------------------------------------------
# 15 — Threshold cutoff per-child
# ---------------------------------------------------------------------------


async def test_decompose_threshold_cutoff_per_child() -> None:
    """Per-child threshold filter — candidates below sim_threshold drop."""
    src_id = uuid4()
    keep_id = uuid4()
    drop_id = uuid4()
    s = AsyncMock()
    s.execute.side_effect = [
        _pg_result([(keep_id, 0.9), (drop_id, 0.9)]),
        _lineage_result([(src_id,)]),
    ]
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1]])
    vector_store = MagicMock()
    vector_store.search = AsyncMock(
        return_value=[
            {"id": str(keep_id), "score": 0.90},
            {"id": str(drop_id), "score": 0.30},
        ]
    )

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=src_id,
        children=[_child(0)],
        settings=_settings(per_child_k=3, threshold=0.70),
        embedder=embedder,
        vector_store=vector_store,
    )
    out_ids = [mid for mid, _ in out[0]]
    assert keep_id in out_ids
    assert drop_id not in out_ids


# ---------------------------------------------------------------------------
# Bonus — Empty children list
# ---------------------------------------------------------------------------


async def test_decompose_empty_children_returns_empty_dict() -> None:
    """Edge case: caller passes empty children list."""
    s = AsyncMock()
    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=uuid4(),
        children=[],
        settings=_settings(),
    )
    assert out == {}
    s.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Bonus — All children skipped at A1 (no embed / no Qdrant)
# ---------------------------------------------------------------------------


async def test_decompose_all_children_skipped_returns_empty_lists() -> None:
    """Every child triggers skip → no embed, no Qdrant. Result has []
    entries for every child (NOT empty dict — that signals failure)."""
    s = AsyncMock()
    embedder = MagicMock()
    embedder.embed_texts = MagicMock()
    vector_store = MagicMock()
    vector_store.search = AsyncMock()

    out = await autowire_fetch_candidates_decompose(
        s=s,
        env_id=uuid4(),
        source_id=uuid4(),
        children=[
            _child(0, kind=MemoryKind.playbook, body="pb"),
            _child(1, body="   "),
        ],
        settings=_settings(),
        embedder=embedder,
        vector_store=vector_store,
    )
    assert out == {0: [], 1: []}
    embedder.embed_texts.assert_not_called()
    vector_store.search.assert_not_called()
    # No PG candidate pull either when all children are pre-filtered.
    s.execute.assert_not_called()


# ---------------------------------------------------------------------------
# reconstruct_auto_wired_by_child — replay-side state-current
# ---------------------------------------------------------------------------


async def test_reconstruct_empty_child_ids_returns_empty_dict() -> None:
    s = AsyncMock()
    out = await reconstruct_auto_wired_by_child(s=s, child_ids=[])
    assert out == {}
    s.execute.assert_not_called()


async def test_reconstruct_groups_rows_per_child() -> None:
    c1, c2 = uuid4(), uuid4()
    d1, d2, d3 = uuid4(), uuid4(), uuid4()
    s = AsyncMock()
    rows = [(c1, d1), (c1, d2), (c2, d3)]
    res = MagicMock()
    res.all.return_value = rows
    s.execute.return_value = res

    out = await reconstruct_auto_wired_by_child(s=s, child_ids=[c1, c2])
    assert out == {c1: [d1, d2], c2: [d3]}


async def test_reconstruct_omits_child_with_no_edges() -> None:
    """Children with zero edges are absent from the result — the
    decompose response builder fills missing entries with []."""
    c1, c2 = uuid4(), uuid4()
    d1 = uuid4()
    s = AsyncMock()
    res = MagicMock()
    res.all.return_value = [(c1, d1)]
    s.execute.return_value = res

    out = await reconstruct_auto_wired_by_child(s=s, child_ids=[c1, c2])
    assert out == {c1: [d1]}
    assert c2 not in out


async def test_reconstruct_normalises_string_uuids() -> None:
    """Some drivers return UUID columns as strings; helper coerces."""
    c1 = uuid4()
    d1 = uuid4()
    s = AsyncMock()
    res = MagicMock()
    res.all.return_value = [(str(c1), str(d1))]
    s.execute.return_value = res

    out = await reconstruct_auto_wired_by_child(s=s, child_ids=[c1])
    assert out == {c1: [d1]}
