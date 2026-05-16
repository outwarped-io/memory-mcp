"""Reciprocal Rank Fusion + tie-break logic.

RRF (Cormack et al. 2009): for each ranked list and each document ``d``,
the document's score is ``Σ 1 / (k + rank_i(d))`` where ``rank_i`` is the
1-indexed rank of ``d`` in list ``i`` (or excluded if absent). The
constant ``k`` (default 60) damps top-rank dominance — it's the standard
value in the literature.

We add two small post-fusion adjustments:

* **Salience boost** — multiply the RRF score by ``(1 + 0.5 * salience)``
  so a salience-1.0 hit edges out a salience-0.0 hit at the same rank.
* **Pinned wins ties** — sort key ``(score, pinned, updated_at)``.

The fusion module keeps the math here; ``api.py`` orchestrates lex/sem
list construction and feeds them in.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class RankedHit:
    """One ranked entry from a single retrieval source (lex / sem / id)."""

    memory_id: UUID
    rank: int  # 1-indexed
    raw_score: float
    source: str  # "lex" | "sem" | "id"


@dataclass
class FusedHit:
    """Output of RRF — re-ranked, with provenance."""

    memory_id: UUID
    score: float
    sources: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)
    pinned: bool = False
    salience: float = 0.0
    updated_at: dt.datetime | None = None


def reciprocal_rank_fuse(
    *,
    lists: Sequence[Iterable[RankedHit]],
    k: int = 60,
) -> dict[UUID, FusedHit]:
    """Fuse N ranked lists into a single ``{memory_id: FusedHit}`` map.

    The result preserves ``raw_scores`` per source for diagnostics.
    Caller fills ``pinned`` / ``salience`` / ``updated_at`` after fusion.
    """
    out: dict[UUID, FusedHit] = {}
    for ranked in lists:
        for hit in ranked:
            entry = out.setdefault(
                hit.memory_id,
                FusedHit(memory_id=hit.memory_id, score=0.0),
            )
            entry.score += 1.0 / (k + hit.rank)
            entry.raw_scores[hit.source] = hit.raw_score
            if hit.source not in entry.sources:
                entry.sources.append(hit.source)
    return out


def apply_salience_boost(hits: Iterable[FusedHit], *, weight: float = 0.5) -> None:
    """Multiply ``hit.score`` by ``(1 + weight*salience)`` in place."""
    for h in hits:
        h.score *= 1.0 + weight * (h.salience or 0.0)


def sort_hits(hits: Iterable[FusedHit]) -> list[FusedHit]:
    """Sort hits descending by (pinned, score, updated_at).

    ``pinned`` is True/False; True wins. Ties on score broken by recency.
    """
    return sorted(
        hits,
        key=lambda h: (
            h.pinned,
            h.score,
            h.updated_at or dt.datetime.min.replace(tzinfo=dt.UTC),
        ),
        reverse=True,
    )


__all__ = [
    "FusedHit",
    "RankedHit",
    "apply_salience_boost",
    "reciprocal_rank_fuse",
    "sort_hits",
]
