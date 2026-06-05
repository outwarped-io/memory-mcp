"""Lifecycle transition invariants.

Encodes the matrix from ``design.md`` → "Lifecycle". Each row is asserted
twice: once as ``is_valid_transition`` returning True for an allowed move,
and once for every disallowed move. This catches accidental table edits
without requiring a database round-trip.

The matrix mirrors the `_LIFECYCLE_TRANSITIONS` declaration in
``memory_mcp.db.types`` but is hand-typed here so a typo on either side
fails the test rather than silently agreeing.
"""

from __future__ import annotations

import pytest

from memory_mcp.db.types import MemoryStatus, is_valid_transition

# (src, allowed_dsts) — copied from design.md, NOT imported from the same
# private constant we're verifying against. Idempotent self-transitions are
# implicit (always allowed) and not enumerated here.
EXPECTED_ALLOWED: dict[MemoryStatus, set[MemoryStatus]] = {
    MemoryStatus.proposed: {MemoryStatus.active, MemoryStatus.retired},
    MemoryStatus.active: {
        MemoryStatus.stale,
        MemoryStatus.archived,
        MemoryStatus.superseded,
        MemoryStatus.retired,
    },
    MemoryStatus.stale: {
        MemoryStatus.active,
        MemoryStatus.archived,
        MemoryStatus.superseded,
        MemoryStatus.retired,
    },
    MemoryStatus.archived: {
        MemoryStatus.active,
        MemoryStatus.superseded,
        MemoryStatus.retired,
    },
    MemoryStatus.superseded: {MemoryStatus.retired},
    MemoryStatus.retired: set(),
}

ALL_STATUSES = list(MemoryStatus)


@pytest.mark.parametrize("src", ALL_STATUSES)
@pytest.mark.parametrize("dst", ALL_STATUSES)
def test_transition_matrix_matches_design(
    src: MemoryStatus,
    dst: MemoryStatus,
) -> None:
    if src == dst:
        # Self-transitions are idempotent (re-application is a no-op).
        assert is_valid_transition(src, dst), f"self-transition {src.value} → {src.value} must be allowed"
        return

    expected = dst in EXPECTED_ALLOWED[src]
    actual = is_valid_transition(src, dst)
    assert actual == expected, f"transition {src.value} → {dst.value}: expected={expected!r} actual={actual!r}"


def test_retired_is_terminal() -> None:
    """Retired never transitions out via the public table.

    Recovery from retired requires an admin restore op, which writes a new
    row rather than transitioning. Verify the table reflects that.
    """
    assert EXPECTED_ALLOWED[MemoryStatus.retired] == set()
    for dst in ALL_STATUSES:
        if dst == MemoryStatus.retired:
            continue
        assert not is_valid_transition(MemoryStatus.retired, dst), f"retired → {dst.value} must NOT be allowed"


def test_proposed_does_not_short_circuit_to_archived() -> None:
    """Proposed memories must go through ``active`` or be ``rejected``.

    Direct ``proposed → archived`` would muddle the dream-review workflow
    and is explicitly excluded.
    """
    assert not is_valid_transition(MemoryStatus.proposed, MemoryStatus.archived)
    assert not is_valid_transition(MemoryStatus.proposed, MemoryStatus.stale)
    assert not is_valid_transition(MemoryStatus.proposed, MemoryStatus.superseded)


def test_archived_can_be_reactivated_or_superseded() -> None:
    """Rubber-duck gate-3 finding: archived → superseded is required so
    later writes can replace stale-archived facts without an admin restore.
    """
    assert is_valid_transition(MemoryStatus.archived, MemoryStatus.superseded)
    assert is_valid_transition(MemoryStatus.archived, MemoryStatus.active)
    assert is_valid_transition(MemoryStatus.archived, MemoryStatus.retired)


def test_superseded_only_path_is_retire() -> None:
    """A superseded memory cannot become active or stale again — the
    successor row owns those states. Only ``retired`` is a permitted
    onward move, used during hard-delete or admin cleanup.
    """
    expected = {MemoryStatus.retired}
    actual = {
        dst
        for dst in ALL_STATUSES
        if dst != MemoryStatus.superseded and is_valid_transition(MemoryStatus.superseded, dst)
    }
    assert actual == expected
