"""Salience scoring — pure function over the relevant memory fields.

Salience drives two things:

1. **Search ranking** — fused into the lex/sem/graph result ordering as a
   tie-break and weighting factor.
2. **Decay** — the dream-worker's decay pass uses thresholds on salience to
   walk ``active → stale → archived``.

The function is intentionally a **pure function over inputs**: same row +
same weights + same ``now`` → same number. This makes the formula
unit-testable without a database and lets the on-read access bump path
project a hypothetical post-update row to compute the new salience BEFORE
issuing the UPDATE (so access_count, last_accessed_at, and salience all
land in a single statement).

Formula
-------

::

    salience = clamp01(
          w_access     · log1p(access_count) / log1p(access_window)
        + w_recency    · exp(-Δt_access / τ_recency)
        + w_confidence · confidence
        - w_negative   · log1p(negative_feedback_count)
        + pinned_bonus      if pinned    else 0
        + verified_bonus    · exp(-Δt_verified / τ_verified)   if verified_at else 0
    )

Where ``Δt_access`` is the seconds between ``now`` and ``last_accessed_at``
(``+inf`` if never accessed; the recency term then collapses to 0), and
``Δt_verified`` is similarly the time since manual verification.

Dominance property
------------------

The default weights are tuned so the negative-feedback term can dominate
the formula. With confidence = 0.0 and access_count = 100 (full saturation
of the access term), 5 negative-feedback events drive salience to 0 even
with maximum recency. This is the "low confidence + rising negatives must
stale even if accessed often" invariant from the design plan.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from memory_mcp.config import Settings


@dataclass(frozen=True)
class SalienceInputs:
    """Subset of memory fields the salience formula needs.

    Decoupled from ``db.models.Memory`` so unit tests don't need an ORM
    instance and so the projected-post-update path in ``memories.py`` can
    construct one without round-tripping through SQLAlchemy.
    """

    access_count: int
    last_accessed_at: dt.datetime | None
    confidence: float
    pinned: bool
    negative_feedback_count: int
    verified_at: dt.datetime | None
    created_at: dt.datetime


@dataclass(frozen=True)
class SalienceWeights:
    """Tunables for :func:`compute_salience`.

    Bound to ``Settings.dream_salience_*`` via
    :func:`salience_weights_from_settings`. Tests can construct a
    bespoke instance to exercise edge cases.
    """

    w_access: float = 0.30
    w_recency: float = 0.25
    w_confidence: float = 0.30
    # 0.30 (not 0.50) — tuned so 5 negatives at zero confidence drives
    # salience to ~0 (dominance invariant) while still letting the pinned
    # bonus protect against moderate (1–2) negative-feedback events.
    w_negative: float = 0.30
    pinned_bonus: float = 0.30
    verified_bonus: float = 0.10
    access_window: int = 100
    recency_tau_seconds: int = 7 * 24 * 3600
    verified_tau_seconds: int = 30 * 24 * 3600

    # Fallback used when ``last_accessed_at is None`` and we still want a
    # tiny floor of recency from ``created_at``. Set to 0 to fully ignore.
    _floor_recency_from_created: bool = field(default=True, repr=False)


def salience_weights_from_settings(settings: Settings) -> SalienceWeights:
    return SalienceWeights(
        w_access=settings.dream_salience_w_access,
        w_recency=settings.dream_salience_w_recency,
        w_confidence=settings.dream_salience_w_confidence,
        w_negative=settings.dream_salience_w_negative,
        pinned_bonus=settings.dream_salience_pinned_bonus,
        verified_bonus=settings.dream_salience_verified_bonus,
        access_window=settings.dream_salience_access_window,
        recency_tau_seconds=settings.dream_salience_recency_tau_seconds,
        verified_tau_seconds=settings.dream_salience_verified_tau_seconds,
    )


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _seconds_since(then: dt.datetime | None, now: dt.datetime) -> float:
    """``now - then`` in seconds; ``inf`` when ``then is None``.

    Both timestamps are expected to be timezone-aware. Naive timestamps are
    treated as UTC (matches Postgres ``timestamptz`` ingest semantics).
    """
    if then is None:
        return math.inf
    if then.tzinfo is None:
        then = then.replace(tzinfo=dt.UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.UTC)
    delta = (now - then).total_seconds()
    # Future timestamps (e.g. clock skew, manual SET) are treated as zero
    # so a moved-into-the-future ``last_accessed_at`` doesn't grant infinite
    # recency boost.
    return max(0.0, delta)


def compute_salience(
    row: SalienceInputs,
    *,
    now: dt.datetime,
    weights: SalienceWeights | None = None,
) -> float:
    """Return a salience score in ``[0.0, 1.0]``.

    Pure function — never reads from or writes to the database. Callers
    project hypothetical post-update field values (e.g. ``access_count + 1``,
    ``last_accessed_at = now``) and re-call this function to compute the
    salience that should land in the same ``UPDATE`` statement.
    """
    w = weights or SalienceWeights()

    # --- Access term ---------------------------------------------------------
    # Saturates at ``access_window``: above that count the term hits the
    # weight ceiling exactly. Without the ``min(1.0, ...)`` cap the ratio
    # ``log1p(N)/log1p(W)`` would keep growing past 1 for ``N > W``, which
    # would let very high-traffic memories overpower the negative-feedback
    # term and break the dominance invariant ("low confidence + rising
    # negatives must stale even if accessed often"). Without the cap,
    # access_count=10_000 would contribute 0.60 instead of 0.30.
    access_norm_denom = math.log1p(max(1, w.access_window))
    access_norm = min(1.0, math.log1p(max(0, row.access_count)) / access_norm_denom)
    access_term = w.w_access * access_norm

    # --- Recency term --------------------------------------------------------
    # exp(-Δt / τ); collapses to 0 when ``last_accessed_at is None`` (Δt = ∞)
    # unless the floor-from-created flag is set, in which case we use
    # ``created_at`` as a tiny baseline so brand-new memories aren't
    # immediately salience-zero.
    delta_access = _seconds_since(row.last_accessed_at, now)
    if row.last_accessed_at is None and w._floor_recency_from_created:
        delta_access = _seconds_since(row.created_at, now)
    recency_factor = (
        0.0
        if math.isinf(delta_access)
        else math.exp(-delta_access / max(1, w.recency_tau_seconds))
    )
    recency_term = w.w_recency * recency_factor

    # --- Confidence term -----------------------------------------------------
    confidence_term = w.w_confidence * _clamp01(row.confidence)

    # --- Negative-feedback term ---------------------------------------------
    # Subtractive on purpose so it can dominate other contributions.
    negative_term = w.w_negative * math.log1p(max(0, row.negative_feedback_count))

    # --- Pinned bonus --------------------------------------------------------
    pinned_term = w.pinned_bonus if row.pinned else 0.0

    # --- Verified bonus ------------------------------------------------------
    if row.verified_at is None:
        verified_term = 0.0
    else:
        delta_verified = _seconds_since(row.verified_at, now)
        if math.isinf(delta_verified):
            verified_factor = 0.0
        else:
            verified_factor = math.exp(-delta_verified / max(1, w.verified_tau_seconds))
        verified_term = w.verified_bonus * verified_factor

    raw = (
        access_term
        + recency_term
        + confidence_term
        - negative_term
        + pinned_term
        + verified_term
    )
    return _clamp01(raw)


__all__ = [
    "SalienceInputs",
    "SalienceWeights",
    "compute_salience",
    "salience_weights_from_settings",
]
