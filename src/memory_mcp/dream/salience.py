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
        + w_references · references_term(N_rl, N_ln, N_tk, N_pb)
        + w_authority  · clamp01(log1p(reference_authority) / log1p(authority_window))
        + pinned_bonus      if pinned    else 0
        + verified_bonus    · exp(-Δt_verified / τ_verified)   if verified_at else 0
    )

Where ``Δt_access`` is the seconds between ``now`` and ``last_accessed_at``
(``+inf`` if never accessed; the recency term then collapses to 0), and
``Δt_verified`` is similarly the time since manual verification.

References term — per-kind independent normalization
----------------------------------------------------

::

    per_kind[k] = clamp01(log1p(N_k) / log1p(window_k))
    combined    = Σ(w_k · per_kind[k]) / Σ(w_k)            # weighted average in [0, 1]
    references_term = combined                              # multiplied by w_references upstream

This shape is **bounded in [0, 1] by construction**, regardless of how
hot or cold any single citation kind happens to be. A v1 design that
summed ``w_k · log1p(N_k) / log1p(window_k)`` directly was rejected — it
saturated too fast (5 playbook embeds consumed 91 % of the ``w_references``
envelope) and made the dominance invariant (below) impossible to preserve.

Dominance property
------------------

The default weights are tuned so the negative-feedback term can dominate
the formula. With ``confidence = 0.0`` and ``access_count = 100`` (full
saturation of the access term), 5 negative-feedback events drive salience
to 0 even with maximum recency AND maximum references contribution.

Phase 1 (memory-mcp v0.14) raises ``w_negative`` from ``0.30`` to ``0.40``
to absorb the new ``w_references = 0.15`` positive term while preserving
the invariant. Phase 1e (memory-mcp v0.14.1) raises it again from
``0.40`` to ``0.46`` to absorb the new ``w_authority = 0.10`` positive
term **and** to narrow the dominance scope.

**Narrowed scope (Phase 1e — R-B2).** The 5-negative dominance
invariant applies only to rows where ``confidence = 0.0``,
``pinned = False``, and ``verified_at = None``. Pinned, verified, or
high-confidence memories are *intentionally* non-dominable by 5
negatives — that's the documented design intent of the bonus terms.

Re-derivation at ``w_negative = 0.46`` under the narrowed scope:

* Positives saturated: ``access(0.30) + recency(0.25) +
  references(0.15) + authority(0.10) = 0.80``.
* Required: ``w_negative · log1p(5) ≥ 0.80`` →
  ``w_negative ≥ 0.80 / 1.7918 = 0.4465``.
* Picked ``w_negative = 0.46``: ``0.46 · 1.7918 ≈ 0.8242``.
* Net: ``0.80 - 0.8242 = -0.0242`` → clamp01 → 0. ✓

Margin ``-0.0242`` is 4× the Phase 1 margin (``-0.006`` at the original
plan target) and comfortable headroom against float drift.

Side effect: ``w_negative = 0.46`` makes a single negative subtract
``0.46 · log1p(1) ≈ 0.319`` (vs ``0.277`` at v0.14, ``0.208`` at the
pre-Phase-1 baseline) — roughly a 15 % heavier hit per negative
relative to v0.14. Tests with hard-coded thresholds were updated.

Formula version invariant (Phase 1e-d)
--------------------------------------

``Memory.salience_formula_version`` stamps the formula version each row's
stored ``salience`` was computed under.
``Settings.dream_salience_formula_version`` declares the current
version. The recount pass picks up any row whose stamp lags the current
version and recomputes + re-stamps.

**ANY change to ``compute_salience`` math — new term, retuned weight,
new clamp behavior — MUST bump
``Settings.dream_salience_formula_version``.** Otherwise existing rows
keep their pre-change salience forever (the recount pass uses set-union
to decide what to recompute; without a version bump, undrifted rows
never enter the set). This includes the obvious "added a new term"
case but also subtler changes like flipping a default weight in
``SalienceWeights`` if that weight is the path callers go through.
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
    # Phase 1 (v0.14): per-kind graph-citation counts. Default 0 so callers
    # that never read the four counter columns (e.g. unit tests authored
    # against pre-Phase-1 fixtures) continue to compute a valid salience.
    reference_count_rel_link: int = 0
    reference_count_lineage: int = 0
    reference_count_task: int = 0
    reference_count_playbook: int = 0
    # Phase 1e-d (v0.14.1): authority = Σ source.salience over inbound
    # citations. Stored in ``Memory.reference_authority`` (GENERATED total
    # of four ``ref_authority_*`` per-kind columns). The salience term is
    # gated through ``SalienceWeights.w_authority`` — when the knob is
    # OFF, ``salience_weights_from_settings`` returns ``w_authority=0.0``
    # and this field has no effect even at saturation. Default 0.0 so
    # callers / unit tests that pre-date 1e-d still compute valid
    # salience.
    reference_authority: float = 0.0


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
    # 0.46 (raised from 0.40 in Phase 1e v0.14.1; from 0.30 in pre-Phase-1
    # baseline). Tuned so the narrowed-scope dominance invariant
    # (``confidence=0, pinned=False, verified_at=None``) survives the
    # addition of both the references term (Phase 1) AND the authority
    # term (Phase 1e, ``w_authority=0.10``). See module docstring's
    # "Dominance property" section.
    w_negative: float = 0.46
    pinned_bonus: float = 0.30
    verified_bonus: float = 0.10
    access_window: int = 100
    recency_tau_seconds: int = 7 * 24 * 3600
    verified_tau_seconds: int = 30 * 24 * 3600

    # ---- Phase 1: references term ----------------------------------------
    # ``w_references`` is the *envelope* — the maximum contribution the
    # references term can make to salience. The per-kind sub-weights below
    # are relative within that envelope.
    w_references: float = 0.15
    # Per-kind sub-weights (relative). Set ratios reflect that lineage and
    # playbook embeds carry stronger signal than ad-hoc rel_link / task
    # references. ``Σ w_k`` is the divisor in the weighted-average step;
    # ratios are what matters, not absolute magnitudes.
    w_references_rl: float = 1.0
    w_references_ln: float = 1.5
    w_references_tk: float = 1.2
    w_references_pb: float = 2.0
    # Saturation windows (per-kind). At ``N_k == window_k`` the kind's
    # per-kind term hits 1.0. Smaller windows saturate faster — lineage and
    # playbook are *rare* signals; one or two embeds already mean a lot.
    window_rl: int = 50
    window_ln: int = 5
    window_tk: int = 20
    window_pb: int = 10

    # ---- Phase 1e: authority term ----------------------------------------
    # Authority = Σ source.salience over inbound citations. Stored as
    # ``Memory.reference_authority`` (sum of four per-kind authority
    # columns). The salience term is gated by
    # ``Settings.dream_popularity_authority_weighted`` — when the knob is
    # OFF, :func:`salience_weights_from_settings` returns
    # ``w_authority=0.0`` and the term zeros without needing a branch
    # inside :func:`compute_salience`. Default ``0.0`` here so direct-ctor
    # callers (unit tests building ``SalienceWeights()`` without a
    # Settings instance) get knob-OFF semantics by default — the
    # ``0.10`` knob-ON value lives in
    # ``Settings.dream_salience_w_authority``.
    #
    # Wired into :func:`compute_salience` in slice 1e-d.
    w_authority: float = 0.0
    authority_window: float = 25.0

    # Fallback used when ``last_accessed_at is None`` and we still want a
    # tiny floor of recency from ``created_at``. Set to 0 to fully ignore.
    _floor_recency_from_created: bool = field(default=True, repr=False)


def salience_weights_from_settings(settings: Settings) -> SalienceWeights:
    # Phase 1e knob gating: when the operator hasn't opted in to authority
    # weighting, return ``w_authority=0.0`` so the term in
    # :func:`compute_salience` zeroes out regardless of the
    # ``reference_authority`` value on the row. Keeps ``compute_salience``
    # pure (no Settings dependency) while still honoring the knob.
    w_authority = settings.dream_salience_w_authority if settings.dream_popularity_authority_weighted else 0.0

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
        w_references=settings.dream_salience_w_references,
        w_references_rl=settings.dream_salience_w_references_rl,
        w_references_ln=settings.dream_salience_w_references_ln,
        w_references_tk=settings.dream_salience_w_references_tk,
        w_references_pb=settings.dream_salience_w_references_pb,
        window_rl=settings.dream_salience_window_rl,
        window_ln=settings.dream_salience_window_ln,
        window_tk=settings.dream_salience_window_tk,
        window_pb=settings.dream_salience_window_pb,
        # Phase 1e-d — gated via ``w_authority`` above (see comment).
        w_authority=w_authority,
        authority_window=settings.dream_salience_authority_window,
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


def _references_term(row: SalienceInputs, w: SalienceWeights) -> float:
    """Compute the references contribution, normalised independently per kind.

    Returns the *unscaled* per-kind weighted average in ``[0, 1]``; the
    caller multiplies by ``w.w_references`` to obtain the salience-space
    contribution.
    """

    def _per_kind(n: int, window: int) -> float:
        if window <= 0:
            return 0.0
        return _clamp01(math.log1p(max(0, n)) / math.log1p(max(1, window)))

    per_rl = _per_kind(row.reference_count_rel_link, w.window_rl)
    per_ln = _per_kind(row.reference_count_lineage, w.window_ln)
    per_tk = _per_kind(row.reference_count_task, w.window_tk)
    per_pb = _per_kind(row.reference_count_playbook, w.window_pb)

    sum_w = w.w_references_rl + w.w_references_ln + w.w_references_tk + w.w_references_pb
    if sum_w <= 0.0:
        return 0.0

    combined = (
        w.w_references_rl * per_rl
        + w.w_references_ln * per_ln
        + w.w_references_tk * per_tk
        + w.w_references_pb * per_pb
    ) / sum_w
    return _clamp01(combined)


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

    # --- Access term --------------------------------------------------------
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

    # --- Recency term -------------------------------------------------------
    # exp(-Δt / τ); collapses to 0 when ``last_accessed_at is None`` (Δt = ∞)
    # unless the floor-from-created flag is set, in which case we use
    # ``created_at`` as a tiny baseline so brand-new memories aren't
    # immediately salience-zero.
    delta_access = _seconds_since(row.last_accessed_at, now)
    if row.last_accessed_at is None and w._floor_recency_from_created:
        delta_access = _seconds_since(row.created_at, now)
    recency_factor = 0.0 if math.isinf(delta_access) else math.exp(-delta_access / max(1, w.recency_tau_seconds))
    recency_term = w.w_recency * recency_factor

    # --- Confidence term ----------------------------------------------------
    confidence_term = w.w_confidence * _clamp01(row.confidence)

    # --- Negative-feedback term --------------------------------------------
    # Subtractive on purpose so it can dominate other contributions.
    negative_term = w.w_negative * math.log1p(max(0, row.negative_feedback_count))

    # --- References term ----------------------------------------------------
    references_term = w.w_references * _references_term(row, w)

    # --- Authority term -----------------------------------------------------
    # Phase 1e-d: Σ source.salience over inbound citations, log-clamped
    # against ``authority_window`` so very-cited memories saturate the
    # contribution rather than dominating the formula. Gated via the
    # weight itself — when the operator has not opted in to authority
    # weighting, ``w.w_authority`` is 0.0 (set by
    # :func:`salience_weights_from_settings`) and the term contributes
    # zero regardless of ``row.reference_authority``. The
    # ``max(1.0, w.authority_window)`` guard avoids division by
    # ``log1p(0) == 0`` if the operator sets the window to a degenerate
    # value.
    authority_norm_denom = math.log1p(max(1.0, w.authority_window))
    authority_norm = _clamp01(math.log1p(max(0.0, row.reference_authority)) / authority_norm_denom)
    authority_term = w.w_authority * authority_norm

    # --- Pinned bonus -------------------------------------------------------
    pinned_term = w.pinned_bonus if row.pinned else 0.0

    # --- Verified bonus -----------------------------------------------------
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
        + references_term
        + authority_term
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
