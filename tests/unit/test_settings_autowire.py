"""Unit tests for the Phase 4 ``autowire_*`` settings knobs.

The auto-wire pass (compose-only in v0.15.0) is OFF by default. These
tests lock in the defaults, range validators, and the cross-knob
invariant that ``autowire_candidate_limit >= autowire_top_k`` (otherwise
the candidate pre-pull can't saturate ``top_k`` and the pass silently
under-emits).
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from memory_mcp.config import Settings


# ---- defaults ----------------------------------------------------------------


def test_autowire_defaults_disabled() -> None:
    """OFF by default — the pass adds latency to every compose, so
    operators must opt in explicitly."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_enabled is False


def test_autowire_default_top_k_three() -> None:
    """K=3 bounds outbox + Neo4j projection pressure per compose."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_top_k == 3


def test_autowire_default_sim_threshold_provisional() -> None:
    """Provisional 0.70 floor — calibration pending; safe because
    autowire_enabled defaults OFF."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_sim_threshold == pytest.approx(0.70)


def test_autowire_default_candidate_limit() -> None:
    """20 = comfortable headroom above default top_k=3 while bounding
    Postgres pre-pull cost."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_candidate_limit == 20


# ---- env overrides -----------------------------------------------------------


def test_autowire_enabled_env_override() -> None:
    with mock.patch.dict(os.environ, {"AUTOWIRE_ENABLED": "true"}, clear=False):
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_enabled is True


def test_autowire_top_k_env_override() -> None:
    with mock.patch.dict(os.environ, {"AUTOWIRE_TOP_K": "5"}, clear=False):
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_top_k == 5


def test_autowire_sim_threshold_env_override() -> None:
    with mock.patch.dict(
        os.environ, {"AUTOWIRE_SIM_THRESHOLD": "0.85"}, clear=False
    ):
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_sim_threshold == pytest.approx(0.85)


def test_autowire_candidate_limit_env_override() -> None:
    with mock.patch.dict(
        os.environ, {"AUTOWIRE_CANDIDATE_LIMIT": "50"}, clear=False
    ):
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_candidate_limit == 50


# ---- range validators --------------------------------------------------------


def test_autowire_top_k_rejects_zero() -> None:
    with (
        mock.patch.dict(os.environ, {"AUTOWIRE_TOP_K": "0"}, clear=False),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_autowire_top_k_rejects_eleven() -> None:
    """Hard cap at 10 — more would defeat the bounded-edges goal."""
    with (
        mock.patch.dict(os.environ, {"AUTOWIRE_TOP_K": "11"}, clear=False),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_autowire_top_k_accepts_one_and_ten_boundaries() -> None:
    for val in ("1", "10"):
        with mock.patch.dict(os.environ, {"AUTOWIRE_TOP_K": val, "AUTOWIRE_CANDIDATE_LIMIT": "200"}, clear=False):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.autowire_top_k == int(val)


def test_autowire_sim_threshold_rejects_negative() -> None:
    with (
        mock.patch.dict(
            os.environ, {"AUTOWIRE_SIM_THRESHOLD": "-0.01"}, clear=False
        ),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_autowire_sim_threshold_rejects_above_one() -> None:
    with (
        mock.patch.dict(
            os.environ, {"AUTOWIRE_SIM_THRESHOLD": "1.01"}, clear=False
        ),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_autowire_sim_threshold_accepts_zero_and_one_boundaries() -> None:
    for val in ("0.0", "1.0"):
        with mock.patch.dict(
            os.environ, {"AUTOWIRE_SIM_THRESHOLD": val}, clear=False
        ):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.autowire_sim_threshold == pytest.approx(float(val))


def test_autowire_candidate_limit_rejects_zero() -> None:
    with (
        mock.patch.dict(
            os.environ, {"AUTOWIRE_CANDIDATE_LIMIT": "0"}, clear=False
        ),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_autowire_candidate_limit_rejects_above_200() -> None:
    with (
        mock.patch.dict(
            os.environ, {"AUTOWIRE_CANDIDATE_LIMIT": "201"}, clear=False
        ),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


# ---- cross-knob invariant ----------------------------------------------------


def test_autowire_candidate_limit_must_be_at_least_top_k() -> None:
    """``autowire_candidate_limit`` < ``autowire_top_k`` is misconfiguration —
    the candidate pre-pull cannot saturate top_k and the pass silently
    under-emits. ``_autowire_invariants`` catches it at config load."""
    with (
        mock.patch.dict(
            os.environ,
            {"AUTOWIRE_TOP_K": "5", "AUTOWIRE_CANDIDATE_LIMIT": "3"},
            clear=False,
        ),
        pytest.raises(ValueError, match="candidate_limit"),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_autowire_candidate_limit_equal_to_top_k_allowed() -> None:
    """Edge case: ``candidate_limit == top_k`` is valid (just tight)."""
    with mock.patch.dict(
        os.environ,
        {"AUTOWIRE_TOP_K": "5", "AUTOWIRE_CANDIDATE_LIMIT": "5"},
        clear=False,
    ):
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_top_k == 5
    assert s.autowire_candidate_limit == 5


# ---- v0.16 decompose knobs --------------------------------------------------


def test_decompose_autowire_defaults_disabled() -> None:
    """Decompose auto-wire OFF by default (independent of master switch);
    fan-out is N× compose risk so operators must opt in explicitly."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_decompose_enabled is False
    assert s.autowire_decompose_per_child_top_k == 3
    assert s.autowire_decompose_total_cap == 30


def test_decompose_autowire_env_overrides() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "AUTOWIRE_ENABLED": "true",
            "AUTOWIRE_DECOMPOSE_ENABLED": "true",
            "AUTOWIRE_DECOMPOSE_PER_CHILD_TOP_K": "5",
            "AUTOWIRE_DECOMPOSE_TOTAL_CAP": "50",
        },
        clear=False,
    ):
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.autowire_decompose_enabled is True
    assert s.autowire_decompose_per_child_top_k == 5
    assert s.autowire_decompose_total_cap == 50


def test_decompose_per_child_top_k_range_validators() -> None:
    """``per_child_top_k`` must be in 1..10 (mirrors compose top_k)."""
    with (
        mock.patch.dict(
            os.environ,
            {"AUTOWIRE_DECOMPOSE_PER_CHILD_TOP_K": "0"},
            clear=False,
        ),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]
    with (
        mock.patch.dict(
            os.environ,
            {"AUTOWIRE_DECOMPOSE_PER_CHILD_TOP_K": "11"},
            clear=False,
        ),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_decompose_total_cap_range_validators() -> None:
    """``total_cap`` must be in 1..100 (bounds worst-case 20×10 = 200)."""
    with (
        mock.patch.dict(
            os.environ,
            {"AUTOWIRE_DECOMPOSE_TOTAL_CAP": "0"},
            clear=False,
        ),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]
    with (
        mock.patch.dict(
            os.environ,
            {"AUTOWIRE_DECOMPOSE_TOTAL_CAP": "101"},
            clear=False,
        ),
        pytest.raises(ValueError),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_decompose_enabled_requires_master_switch() -> None:
    """``autowire_decompose_enabled`` without master ``autowire_enabled``
    is misconfiguration. Master OFF disables ALL auto-wire."""
    with (
        mock.patch.dict(
            os.environ,
            {
                "AUTOWIRE_ENABLED": "false",
                "AUTOWIRE_DECOMPOSE_ENABLED": "true",
            },
            clear=False,
        ),
        pytest.raises(ValueError, match="requires autowire_enabled"),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_decompose_total_cap_must_be_at_least_per_child_top_k() -> None:
    """Global cap below per-child cap is incoherent — every child
    would silently clip before per-child K is applied."""
    with (
        mock.patch.dict(
            os.environ,
            {
                "AUTOWIRE_DECOMPOSE_PER_CHILD_TOP_K": "5",
                "AUTOWIRE_DECOMPOSE_TOTAL_CAP": "3",
            },
            clear=False,
        ),
        pytest.raises(ValueError, match="total_cap"),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_decompose_candidate_limit_must_accommodate_per_child_top_k() -> None:
    """Shared candidate pre-pull serves all children; if it can't even
    saturate per-child K, the pass silently under-emits."""
    with (
        mock.patch.dict(
            os.environ,
            {
                "AUTOWIRE_TOP_K": "1",
                "AUTOWIRE_CANDIDATE_LIMIT": "2",
                "AUTOWIRE_DECOMPOSE_PER_CHILD_TOP_K": "5",
            },
            clear=False,
        ),
        pytest.raises(ValueError, match="candidate_limit"),
    ):
        Settings(_env_file=None)  # type: ignore[call-arg]
