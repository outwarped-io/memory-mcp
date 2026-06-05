"""Unit tests for ``memory_mcp.search.ranking`` (pure-math, no DB)."""

from __future__ import annotations

import datetime as dt
from uuid import uuid4

from memory_mcp.search.ranking import (
    FusedHit,
    RankedHit,
    apply_salience_boost,
    reciprocal_rank_fuse,
    sort_hits,
)


def test_rrf_single_list_decreasing_scores() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()
    lex = [
        RankedHit(memory_id=a, rank=1, raw_score=0.9, source="lex"),
        RankedHit(memory_id=b, rank=2, raw_score=0.5, source="lex"),
        RankedHit(memory_id=c, rank=3, raw_score=0.1, source="lex"),
    ]
    fused = reciprocal_rank_fuse(lists=[lex])
    assert fused[a].score > fused[b].score > fused[c].score
    for mid in (a, b, c):
        assert fused[mid].sources == ["lex"]
        assert "lex" in fused[mid].raw_scores


def test_rrf_dual_list_dual_source_wins() -> None:
    """A document in BOTH lists should outrank one in only ONE list."""
    a, b, c = uuid4(), uuid4(), uuid4()
    lex = [
        RankedHit(memory_id=a, rank=1, raw_score=0.9, source="lex"),
        RankedHit(memory_id=c, rank=2, raw_score=0.5, source="lex"),
    ]
    sem = [
        RankedHit(memory_id=a, rank=2, raw_score=0.7, source="sem"),
        RankedHit(memory_id=b, rank=1, raw_score=0.8, source="sem"),
    ]
    fused = reciprocal_rank_fuse(lists=[lex, sem])
    # ``a`` appears in both → score = 1/(60+1) + 1/(60+2)
    # ``b`` only in sem at rank 1 → 1/(60+1)
    # ``c`` only in lex at rank 2 → 1/(60+2)
    assert fused[a].score > fused[b].score
    assert fused[a].score > fused[c].score
    assert set(fused[a].sources) == {"lex", "sem"}
    assert fused[b].sources == ["sem"]


def test_rrf_k_constant_dampens_top_rank() -> None:
    """Increasing k brings ranks closer (lower score gradient)."""
    a, b = uuid4(), uuid4()
    lex = [
        RankedHit(memory_id=a, rank=1, raw_score=1, source="lex"),
        RankedHit(memory_id=b, rank=2, raw_score=1, source="lex"),
    ]
    f1 = reciprocal_rank_fuse(lists=[lex], k=1)
    f60 = reciprocal_rank_fuse(lists=[lex], k=60)
    gradient_1 = f1[a].score - f1[b].score
    gradient_60 = f60[a].score - f60[b].score
    assert gradient_1 > gradient_60


def test_apply_salience_boost_preserves_zero() -> None:
    a = uuid4()
    h = FusedHit(memory_id=a, score=0.5, salience=0.0)
    apply_salience_boost([h])
    assert h.score == 0.5  # 1 + 0.5*0 = 1


def test_apply_salience_boost_multiplies() -> None:
    a = uuid4()
    h = FusedHit(memory_id=a, score=0.4, salience=1.0)
    apply_salience_boost([h], weight=0.5)
    # 0.4 * (1 + 0.5*1) = 0.6
    assert abs(h.score - 0.6) < 1e-9


def test_sort_hits_pinned_wins_over_higher_score() -> None:
    pinned = FusedHit(memory_id=uuid4(), score=0.1, pinned=True)
    unpinned = FusedHit(memory_id=uuid4(), score=0.9, pinned=False)
    sorted_ = sort_hits([unpinned, pinned])
    assert sorted_[0] is pinned
    assert sorted_[1] is unpinned


def test_sort_hits_score_then_recency_breaks_ties() -> None:
    older = FusedHit(
        memory_id=uuid4(),
        score=0.5,
        pinned=False,
        updated_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
    )
    newer = FusedHit(
        memory_id=uuid4(),
        score=0.5,
        pinned=False,
        updated_at=dt.datetime(2026, 5, 1, tzinfo=dt.UTC),
    )
    sorted_ = sort_hits([older, newer])
    assert sorted_[0] is newer


def test_rrf_empty_lists_returns_empty() -> None:
    assert reciprocal_rank_fuse(lists=[]) == {}
    assert reciprocal_rank_fuse(lists=[[], []]) == {}
