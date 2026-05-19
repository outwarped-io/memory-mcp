"""Unit tests for the salience pure function.

Validates:

* Output is always clamped to ``[0.0, 1.0]``.
* Monotonicity in each contributing variable holding others fixed.
* The "negative-feedback dominates" invariant from the design plan.
* Pinned + verified bonuses are additive and decay cleanly.
* Timezone-naive inputs are treated as UTC.
* Future timestamps don't grant infinite recency boost.
* Settings-bound weights match the dataclass defaults.
"""

from __future__ import annotations

import datetime as dt

from memory_mcp.config import Settings
from memory_mcp.dream.salience import (
    SalienceInputs,
    SalienceWeights,
    compute_salience,
    salience_weights_from_settings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTC = dt.UTC
NOW = dt.datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)


def _row(
    *,
    access_count: int = 0,
    last_accessed_at: dt.datetime | None = None,
    confidence: float = 0.5,
    pinned: bool = False,
    negative_feedback_count: int = 0,
    verified_at: dt.datetime | None = None,
    created_at: dt.datetime | None = None,
    reference_count_rel_link: int = 0,
    reference_count_lineage: int = 0,
    reference_count_task: int = 0,
    reference_count_playbook: int = 0,
) -> SalienceInputs:
    return SalienceInputs(
        access_count=access_count,
        last_accessed_at=last_accessed_at,
        confidence=confidence,
        pinned=pinned,
        negative_feedback_count=negative_feedback_count,
        verified_at=verified_at,
        created_at=created_at if created_at is not None else NOW - dt.timedelta(days=1),
        reference_count_rel_link=reference_count_rel_link,
        reference_count_lineage=reference_count_lineage,
        reference_count_task=reference_count_task,
        reference_count_playbook=reference_count_playbook,
    )


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------

def test_salience_is_always_in_unit_interval() -> None:
    # Pathological inputs: all maxima, all minima, mixed
    rows = [
        _row(access_count=10**9, confidence=10.0, pinned=True, verified_at=NOW),
        _row(access_count=0, confidence=-5.0, negative_feedback_count=10**6),
        _row(access_count=50, confidence=0.5, negative_feedback_count=3),
    ]
    for r in rows:
        s = compute_salience(r, now=NOW)
        assert 0.0 <= s <= 1.0, f"salience out of range: {s}"


def test_neutral_defaults_yield_mid_range_salience() -> None:
    # A freshly-accessed memory at default confidence should be solidly
    # above zero but well below the pinned/verified ceiling.
    s = compute_salience(_row(last_accessed_at=NOW), now=NOW)
    assert 0.3 < s < 0.7, f"neutral salience expected mid-range, got {s}"


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

def test_access_count_monotonically_increases_salience() -> None:
    base = _row(last_accessed_at=NOW, confidence=0.5)
    s_low = compute_salience(base, now=NOW)
    s_med = compute_salience(_row(last_accessed_at=NOW, confidence=0.5, access_count=10), now=NOW)
    s_high = compute_salience(_row(last_accessed_at=NOW, confidence=0.5, access_count=100), now=NOW)
    assert s_low < s_med < s_high


def test_confidence_monotonically_increases_salience() -> None:
    s_low = compute_salience(_row(last_accessed_at=NOW, confidence=0.0), now=NOW)
    s_high = compute_salience(_row(last_accessed_at=NOW, confidence=1.0), now=NOW)
    assert s_low < s_high


def test_negative_feedback_monotonically_decreases_salience() -> None:
    # Use ``access_count=100`` + ``pinned=True`` to lift the base above the
    # clamp floor, so the curve is monotonic across multiple negative
    # counts under the v0.14 ``w_negative=0.40`` tuning.
    s_zero = compute_salience(
        _row(last_accessed_at=NOW, access_count=100, pinned=True, negative_feedback_count=0),
        now=NOW,
    )
    s_two = compute_salience(
        _row(last_accessed_at=NOW, access_count=100, pinned=True, negative_feedback_count=2),
        now=NOW,
    )
    s_ten = compute_salience(
        _row(last_accessed_at=NOW, access_count=100, pinned=True, negative_feedback_count=10),
        now=NOW,
    )
    assert s_zero > s_two > s_ten


def test_recency_monotonically_decreases_salience() -> None:
    fresh = compute_salience(_row(last_accessed_at=NOW), now=NOW)
    day = compute_salience(_row(last_accessed_at=NOW - dt.timedelta(days=1)), now=NOW)
    week = compute_salience(_row(last_accessed_at=NOW - dt.timedelta(days=7)), now=NOW)
    month = compute_salience(_row(last_accessed_at=NOW - dt.timedelta(days=30)), now=NOW)
    assert fresh > day > week > month


# ---------------------------------------------------------------------------
# Bonuses
# ---------------------------------------------------------------------------

def test_pinned_bonus_strictly_increases_salience() -> None:
    base = _row(last_accessed_at=NOW, confidence=0.3)
    s_unpinned = compute_salience(base, now=NOW)
    s_pinned = compute_salience(
        _row(last_accessed_at=NOW, confidence=0.3, pinned=True),
        now=NOW,
    )
    assert s_pinned > s_unpinned
    assert s_pinned - s_unpinned >= 0.25  # default pinned_bonus is 0.30 (clamped)


def test_verified_bonus_decays_over_time() -> None:
    fresh = compute_salience(
        _row(last_accessed_at=NOW, verified_at=NOW), now=NOW
    )
    old = compute_salience(
        _row(last_accessed_at=NOW, verified_at=NOW - dt.timedelta(days=90)),
        now=NOW,
    )
    none = compute_salience(_row(last_accessed_at=NOW), now=NOW)
    # Fresh verification > old verification ≈ no verification (within rounding).
    assert fresh > old
    assert old <= none + 0.02  # at 90 days τ=30, exp(-3) ≈ 0.05 contribution


# ---------------------------------------------------------------------------
# Negative-feedback dominance — design plan invariant
# ---------------------------------------------------------------------------

def test_negative_feedback_dominates_high_access_low_confidence() -> None:
    """The canonical invariant: low confidence + rising negatives must stale
    even when access_count is at saturation and recency is maximal."""
    bad = compute_salience(
        _row(
            last_accessed_at=NOW,
            access_count=100,
            confidence=0.0,
            negative_feedback_count=5,
        ),
        now=NOW,
    )
    assert bad < 0.10, (
        f"5 negatives + zero confidence + max access must score very low, got {bad}"
    )


def test_negative_feedback_dominates_at_extreme_access_counts() -> None:
    """Dominance must also hold for memories with access_count >>
    ``access_window`` — the access term is capped, so 1000+ accesses
    can't outpower 5 negatives at zero confidence."""
    for access in (1_000, 10_000, 1_000_000):
        bad = compute_salience(
            _row(
                last_accessed_at=NOW,
                access_count=access,
                confidence=0.0,
                negative_feedback_count=5,
            ),
            now=NOW,
        )
        assert bad < 0.10, (
            f"access_count={access}: dominance failed, got {bad}"
        )


def test_access_term_saturates_at_access_window() -> None:
    """The access term hits its weight ceiling at ``access_count ==
    access_window`` and does not grow beyond. Without this cap, a
    high-traffic memory would overpower the negative-feedback term."""
    base_inputs = dict(  # noqa: C408
        last_accessed_at=NOW, confidence=0.0, negative_feedback_count=0,
    )
    s_at_cap = compute_salience(
        _row(access_count=100, **base_inputs), now=NOW,
    )
    s_above_cap = compute_salience(
        _row(access_count=10_000, **base_inputs), now=NOW,
    )
    s_far_above = compute_salience(
        _row(access_count=1_000_000, **base_inputs), now=NOW,
    )
    # All three must be equal: access_count beyond access_window contributes
    # the same (capped) amount.
    assert abs(s_at_cap - s_above_cap) < 1e-9
    assert abs(s_at_cap - s_far_above) < 1e-9


def test_pinned_protects_against_moderate_negatives_but_not_extreme() -> None:
    moderate = compute_salience(
        _row(
            last_accessed_at=NOW, confidence=0.5,
            negative_feedback_count=1, pinned=True,
        ),
        now=NOW,
    )
    extreme = compute_salience(
        _row(
            last_accessed_at=NOW, confidence=0.0,
            negative_feedback_count=20, pinned=True,
        ),
        now=NOW,
    )
    # Pinned + 1 negative: stays above the typical stale threshold (0.30)
    # so the decay pass won't archive it. With the v0.14 ``w_negative=0.40``
    # tuning, 1 negative subtracts ~0.28 from the positive 0.70 base
    # (recency + confidence + pinned) → ~0.42 > 0.30.
    assert moderate > 0.30
    # Pinned can't outweigh 20 negatives at zero confidence.
    assert extreme < 0.10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_naive_timestamps_treated_as_utc() -> None:
    naive = NOW.replace(tzinfo=None)
    s_naive = compute_salience(_row(last_accessed_at=naive), now=NOW)
    s_aware = compute_salience(_row(last_accessed_at=NOW), now=NOW)
    assert abs(s_naive - s_aware) < 1e-9


def test_future_last_accessed_at_does_not_grant_extra_boost() -> None:
    """Clock skew safety: a future ``last_accessed_at`` is clamped to "now"
    so it can't grant > the maximum recency contribution."""
    future = NOW + dt.timedelta(days=10)
    s_future = compute_salience(_row(last_accessed_at=future), now=NOW)
    s_now = compute_salience(_row(last_accessed_at=NOW), now=NOW)
    assert abs(s_future - s_now) < 1e-9


def test_never_accessed_falls_back_to_created_at() -> None:
    # Default weights have ``_floor_recency_from_created=True`` so a
    # brand-new never-accessed memory still gets a sliver of recency.
    s_just_created = compute_salience(
        _row(last_accessed_at=None, created_at=NOW), now=NOW
    )
    s_old_never_accessed = compute_salience(
        _row(last_accessed_at=None, created_at=NOW - dt.timedelta(days=365)),
        now=NOW,
    )
    assert s_just_created > s_old_never_accessed


def test_floor_recency_from_created_can_be_disabled() -> None:
    weights = SalienceWeights(_floor_recency_from_created=False)
    s = compute_salience(
        _row(last_accessed_at=None, created_at=NOW),
        now=NOW,
        weights=weights,
    )
    # No access term, no recency term, default 0.5 confidence → 0.30 · 0.5 = 0.15
    # (small access-count fallback also contributes 0 since access_count=0).
    assert s < 0.20


# ---------------------------------------------------------------------------
# Settings binding
# ---------------------------------------------------------------------------

def test_salience_weights_from_settings_match_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    bound = salience_weights_from_settings(s)
    default = SalienceWeights()
    assert bound.w_access == default.w_access
    assert bound.w_recency == default.w_recency
    assert bound.w_confidence == default.w_confidence
    assert bound.w_negative == default.w_negative
    assert bound.pinned_bonus == default.pinned_bonus
    assert bound.verified_bonus == default.verified_bonus
    assert bound.access_window == default.access_window
    assert bound.recency_tau_seconds == default.recency_tau_seconds
    assert bound.verified_tau_seconds == default.verified_tau_seconds


def test_settings_overrides_propagate_to_weights() -> None:
    s = Settings(_env_file=None, dream_salience_w_negative=0.99)  # type: ignore[call-arg]
    bound = salience_weights_from_settings(s)
    assert bound.w_negative == 0.99


# ---------------------------------------------------------------------------
# Phase 1 (v0.14): references term
# ---------------------------------------------------------------------------

def test_references_term_zero_when_no_citations() -> None:
    no_refs = compute_salience(_row(last_accessed_at=NOW), now=NOW)
    with_refs = compute_salience(
        _row(last_accessed_at=NOW, reference_count_rel_link=10),
        now=NOW,
    )
    assert with_refs > no_refs


def test_each_reference_kind_independently_lifts_salience() -> None:
    base = compute_salience(_row(last_accessed_at=NOW), now=NOW)
    rl = compute_salience(_row(last_accessed_at=NOW, reference_count_rel_link=50), now=NOW)
    ln = compute_salience(_row(last_accessed_at=NOW, reference_count_lineage=5), now=NOW)
    tk = compute_salience(_row(last_accessed_at=NOW, reference_count_task=20), now=NOW)
    pb = compute_salience(_row(last_accessed_at=NOW, reference_count_playbook=10), now=NOW)
    assert rl > base
    assert ln > base
    assert tk > base
    assert pb > base


def test_playbook_carries_more_weight_per_edge_than_rel_link() -> None:
    one_playbook = compute_salience(
        _row(last_accessed_at=NOW, reference_count_playbook=1), now=NOW
    )
    one_rel_link = compute_salience(
        _row(last_accessed_at=NOW, reference_count_rel_link=1), now=NOW
    )
    # Same edge count; per-kind sub-weight ordering (pb=2.0 vs rl=1.0) plus
    # tighter playbook window (10 vs 50) should make a single playbook edge
    # contribute more than a single rel_link edge.
    assert one_playbook > one_rel_link


def test_references_term_saturates_at_envelope_ceiling() -> None:
    # All four kinds saturated past their windows; references term should
    # approach (but not exceed) ``w_references`` (default 0.15).
    high = _row(
        last_accessed_at=NOW,
        confidence=0.0,  # neutralize confidence term
        reference_count_rel_link=500,
        reference_count_lineage=50,
        reference_count_task=200,
        reference_count_playbook=100,
    )
    bare = _row(last_accessed_at=NOW, confidence=0.0)
    s_high = compute_salience(high, now=NOW)
    s_bare = compute_salience(bare, now=NOW)
    delta = s_high - s_bare
    # Ceiling is ``w_references`` = 0.15; saturated kinds approach 1.0 each
    # so weighted-average → 1.0 and contribution → 0.15. Allow small
    # numerical slack for log1p-near-window rounding.
    assert delta <= SalienceWeights().w_references + 1e-9
    assert delta >= 0.14  # Near ceiling at saturation.


def test_dominance_invariant_with_references_at_max() -> None:
    """v0.14 dominance check: even with ALL positive terms at saturation
    AND references at full envelope, 5 negatives at zero confidence drive
    salience to ~0. Reverifies the ``w_negative=0.40`` retune (B1)."""
    saturated = _row(
        access_count=1000,
        last_accessed_at=NOW,
        confidence=0.0,
        negative_feedback_count=5,
        reference_count_rel_link=500,
        reference_count_lineage=50,
        reference_count_task=200,
        reference_count_playbook=100,
    )
    s = compute_salience(saturated, now=NOW)
    assert s <= 0.001


def test_settings_propagate_references_weights() -> None:
    s = Settings(  # type: ignore[call-arg]
        _env_file=None,
        dream_salience_w_references=0.25,
        dream_salience_w_references_rl=2.0,
        dream_salience_window_pb=20,
    )
    bound = salience_weights_from_settings(s)
    assert bound.w_references == 0.25
    assert bound.w_references_rl == 2.0
    assert bound.window_pb == 20


# ---------------------------------------------------------------------------
# R-B4 — access-bump preserves citation contribution
# ---------------------------------------------------------------------------

def test_access_bump_preserves_citation_contribution() -> None:
    """R-B4 regression (Phase 1e plan §A10 / slice 1e-b').

    On the read-path access bump, ``memories.py`` constructs a
    ``SalienceInputs`` reflecting the post-bump row and recomputes
    salience. The pre-1e-b' code path passed only the access/recency/
    feedback fields and omitted the four ``reference_count_*`` fields,
    silently letting them default to 0. Effect: every read on a cited
    memory recomputed salience as if the citations didn't exist —
    erasing the references-term contribution from the stored salience.

    This test pins the post-fix behavior: with non-zero citation
    counts, the recomputed salience must include the references term.
    The simplest assertion is monotone — bumping the access count of a
    cited memory yields a salience strictly greater than what bumping
    an uncited memory with identical other fields would produce, AND
    strictly greater than what the buggy "omit citations" path would
    have computed.
    """
    weights = SalienceWeights()
    # Same row: bumped access, fixed recency, cited.
    cited_post_bump = compute_salience(
        _row(
            access_count=10,
            last_accessed_at=NOW,
            confidence=0.0,  # neutralize confidence so references is visible
            reference_count_rel_link=20,
            reference_count_lineage=2,
            reference_count_task=5,
            reference_count_playbook=3,
        ),
        now=NOW,
        weights=weights,
    )
    # Same row sans citations — the broken pre-1e-b' constructor that
    # forgot to pass the counts.
    uncited_post_bump = compute_salience(
        _row(
            access_count=10,
            last_accessed_at=NOW,
            confidence=0.0,
        ),
        now=NOW,
        weights=weights,
    )
    # The fixed path must produce strictly larger salience than the
    # broken path would have. Margin floor at 0.05 (well below the
    # w_references=0.15 envelope) to keep the test robust against
    # window/weight tweaks.
    assert cited_post_bump - uncited_post_bump >= 0.05, (
        f"access-bump path lost citation contribution: "
        f"cited={cited_post_bump:.4f}, uncited={uncited_post_bump:.4f}"
    )
