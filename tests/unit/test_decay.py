"""Unit tests for the dream-decay pass.

Tests run without a real Postgres — the SELECT loaders and
``memory_update`` are monkeypatched to controlled fakes. This keeps the
suite hermetic and catches state-machine regressions in milliseconds.

Coverage matrix:

* Active leg transitions: below-threshold ⇒ stale; above-threshold ⇒ no-op.
* Stale leg transitions: below-threshold ⇒ archived; above-threshold ⇒ no-op.
* Skips: pinned never enters either leg's candidate set (loader filter);
  retired / superseded never enter either leg.
* Idempotency: re-running with no DB-state change emits no new transitions.
* Concurrency: ``VersionConflictError`` from ``memory_update`` is silently
  skipped and counted.
* Cap: hitting ``DREAM_DECAY_BATCH_CAP`` flips ``items_capped_*`` flags.
* Empty env: clean zeros everywhere, no exceptions.
* Validation: env not in actor_ctx.attached_env_ids ⇒ ValueError early.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from memory_mcp.config import Settings
from memory_mcp.db.types import MemoryStatus
from memory_mcp.dream.passes.decay import (
    DecayCandidateRow,
    DecayPassResult,
    run_decay,
)
from memory_mcp.dream.salience import SalienceInputs
from memory_mcp.errors import VersionConflictError
from memory_mcp.identity import AgentContext

UTC = dt.UTC
NOW = dt.datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(env_id: UUID) -> AgentContext:
    return AgentContext(agent_id=uuid4(), attached_env_ids=[env_id])


def _settings() -> Settings:
    return Settings(  # type: ignore[arg-type]
        dream_decay_inactive_days=30,
        dream_decay_stale_threshold=0.30,
        dream_decay_archive_threshold=0.10,
        dream_decay_batch_cap=500,
    )


def _candidate(
    *,
    status: MemoryStatus = MemoryStatus.active,
    version: int = 1,
    access_count: int = 0,
    last_accessed_at: dt.datetime | None = None,
    confidence: float = 0.5,
    pinned: bool = False,
    negative_feedback_count: int = 0,
    verified_at: dt.datetime | None = None,
    created_at: dt.datetime | None = None,
    reference_count: int = 0,
) -> DecayCandidateRow:
    return DecayCandidateRow(
        id=uuid4(),
        version=version,
        status=status,
        salience_inputs=SalienceInputs(
            access_count=access_count,
            last_accessed_at=last_accessed_at,
            confidence=confidence,
            pinned=pinned,
            negative_feedback_count=negative_feedback_count,
            verified_at=verified_at,
            created_at=created_at or (NOW - dt.timedelta(days=180)),
        ),
        reference_count=reference_count,
    )


def _patch_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active: list[DecayCandidateRow],
    stale: list[DecayCandidateRow],
) -> tuple[AsyncMock, AsyncMock]:
    """Replace SELECT loaders with AsyncMocks returning canned rows."""
    active_loader = AsyncMock(return_value=active)
    stale_loader = AsyncMock(return_value=stale)
    monkeypatch.setattr(
        "memory_mcp.dream.passes.decay._load_active_candidates", active_loader,
    )
    monkeypatch.setattr(
        "memory_mcp.dream.passes.decay._load_stale_candidates", stale_loader,
    )
    return active_loader, stale_loader


def _patch_memory_update(
    monkeypatch: pytest.MonkeyPatch,
    *,
    side_effect: list[Any] | None = None,
) -> AsyncMock:
    """Replace memory_update with an AsyncMock; default returns ``None``."""
    update = AsyncMock(side_effect=side_effect) if side_effect else AsyncMock(return_value=None)
    monkeypatch.setattr("memory_mcp.dream.passes.decay.memory_update", update)
    return update


# ---------------------------------------------------------------------------
# Active leg (active → stale)
# ---------------------------------------------------------------------------


class TestActiveLeg:
    @pytest.mark.asyncio
    async def test_below_threshold_transitions_to_stale(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        # Cold + low-confidence + negative feedback ⇒ salience well below 0.30.
        cold = _candidate(
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=180),
            confidence=0.1,
            negative_feedback_count=3,
        )
        _patch_loaders(monkeypatch, active=[cold], stale=[])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert isinstance(result, DecayPassResult)
        assert result.examined_active == 1
        assert result.transitioned_to_stale == 1
        assert result.transitioned_to_archived == 0
        assert result.skipped_above_threshold == 0
        assert update.await_count == 1
        # Verify the patch
        call_kwargs = update.await_args.kwargs
        patch = update.await_args.args[1]
        assert patch.status == MemoryStatus.stale
        assert patch.expected_version == cold.version
        assert call_kwargs["ctx"].attached_env_ids == [env_id]

    @pytest.mark.asyncio
    async def test_above_threshold_no_op(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        # Hot + high-confidence ⇒ salience well above 0.30.
        hot = _candidate(
            access_count=50,
            last_accessed_at=NOW - dt.timedelta(hours=1),
            confidence=0.9,
        )
        _patch_loaders(monkeypatch, active=[hot], stale=[])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert result.examined_active == 1
        assert result.transitioned_to_stale == 0
        assert result.skipped_above_threshold == 1
        assert update.await_count == 0

    @pytest.mark.asyncio
    async def test_version_conflict_silently_skipped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        cold = _candidate(
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=180),
            confidence=0.1,
        )
        _patch_loaders(monkeypatch, active=[cold], stale=[])
        update = _patch_memory_update(
            monkeypatch,
            side_effect=[VersionConflictError(expected=1, actual=2)],
        )

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert result.examined_active == 1
        assert result.transitioned_to_stale == 0
        assert result.skipped_version_conflicts == 1
        assert update.await_count == 1


# ---------------------------------------------------------------------------
# Stale leg (stale → archived)
# ---------------------------------------------------------------------------


class TestStaleLeg:
    @pytest.mark.asyncio
    async def test_below_archive_threshold_transitions(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        # Very cold + heavy negatives ⇒ salience below 0.10.
        ghost = _candidate(
            status=MemoryStatus.stale,
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=365),
            confidence=0.0,
            negative_feedback_count=20,
        )
        _patch_loaders(monkeypatch, active=[], stale=[ghost])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert result.examined_stale == 1
        assert result.transitioned_to_archived == 1
        assert update.await_count == 1
        patch = update.await_args.args[1]
        assert patch.status == MemoryStatus.archived
        assert patch.expected_version == ghost.version

    @pytest.mark.asyncio
    async def test_above_archive_threshold_no_op(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        # Stale but middling salience (0.10 < s < 0.30): stays stale.
        warm_stale = _candidate(
            status=MemoryStatus.stale,
            access_count=2,
            last_accessed_at=NOW - dt.timedelta(days=10),
            confidence=0.5,
        )
        _patch_loaders(monkeypatch, active=[], stale=[warm_stale])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert result.examined_stale == 1
        assert result.transitioned_to_archived == 0
        assert result.skipped_above_threshold == 1
        assert update.await_count == 0


# ---------------------------------------------------------------------------
# Two-leg interaction
# ---------------------------------------------------------------------------


class TestTwoLegInteraction:
    @pytest.mark.asyncio
    async def test_active_and_stale_in_same_run(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        cold_active = _candidate(
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=180),
            confidence=0.1,
        )
        ghost_stale = _candidate(
            status=MemoryStatus.stale,
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=365),
            confidence=0.0,
            negative_feedback_count=20,
        )
        _patch_loaders(monkeypatch, active=[cold_active], stale=[ghost_stale])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert result.transitioned_to_stale == 1
        assert result.transitioned_to_archived == 1
        assert update.await_count == 2
        # Verify both targets used the right status patch.
        statuses = {call.args[1].status for call in update.await_args_list}
        assert statuses == {MemoryStatus.stale, MemoryStatus.archived}

    @pytest.mark.asyncio
    async def test_empty_env_clean_zeros(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        _patch_loaders(monkeypatch, active=[], stale=[])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert result.examined_active == 0
        assert result.examined_stale == 0
        assert result.transitioned_to_stale == 0
        assert result.transitioned_to_archived == 0
        assert result.skipped_version_conflicts == 0
        assert result.skipped_above_threshold == 0
        assert result.items_capped_active_leg is False
        assert result.items_capped_stale_leg is False
        assert update.await_count == 0


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_rerun_after_unchanged_state(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two consecutive runs with the same SELECT result + healthy
        ``memory_update`` should produce the same transitions on the
        first run and zero transitions on the second run (because the
        rows would no longer match the active leg's filter — we simulate
        this by clearing the loader between runs)."""
        env_id = uuid4()
        cold = _candidate(
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=180),
            confidence=0.1,
        )
        _patch_loaders(monkeypatch, active=[cold], stale=[])
        update = _patch_memory_update(monkeypatch)

        first = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )
        # Simulate the SQL filter (status=active) pruning the now-stale
        # row out of the next run's candidate set.
        _patch_loaders(monkeypatch, active=[], stale=[])

        second = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert first.transitioned_to_stale == 1
        assert second.transitioned_to_stale == 0
        assert second.transitioned_to_archived == 0
        # Single round-trip across two runs.
        assert update.await_count == 1


# ---------------------------------------------------------------------------
# Cap behavior
# ---------------------------------------------------------------------------


class TestBatchCap:
    @pytest.mark.asyncio
    async def test_active_leg_cap_flag(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        # Cap=2; loader returns exactly 2 ⇒ flag flips True.
        settings = Settings(  # type: ignore[arg-type]
            dream_decay_batch_cap=2,
        )
        rows = [
            _candidate(
                access_count=0,
                last_accessed_at=NOW - dt.timedelta(days=180),
                confidence=0.1,
            )
            for _ in range(2)
        ]
        _patch_loaders(monkeypatch, active=rows, stale=[])
        _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=settings, now=NOW,
        )

        assert result.items_capped_active_leg is True
        assert result.items_capped_stale_leg is False
        assert result.transitioned_to_stale == 2

    @pytest.mark.asyncio
    async def test_below_cap_flag_stays_false(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        settings = Settings(dream_decay_batch_cap=10)  # type: ignore[arg-type]
        rows = [
            _candidate(
                access_count=0,
                last_accessed_at=NOW - dt.timedelta(days=180),
                confidence=0.1,
            )
            for _ in range(3)
        ]
        _patch_loaders(monkeypatch, active=rows, stale=[])
        _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=settings, now=NOW,
        )

        assert result.items_capped_active_leg is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.asyncio
    async def test_env_not_attached_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        other_env = uuid4()
        ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[other_env])
        # Loaders shouldn't even be called — exception fires first.
        active_loader = AsyncMock(return_value=[])
        monkeypatch.setattr(
            "memory_mcp.dream.passes.decay._load_active_candidates", active_loader,
        )

        with pytest.raises(ValueError, match="attached_env_ids"):
            await run_decay(
                env_id, actor_ctx=ctx, settings=_settings(), now=NOW,
            )

        active_loader.assert_not_called()


# ---------------------------------------------------------------------------
# Settings validation (decay knobs)
# ---------------------------------------------------------------------------


class TestDecaySettingsValidation:
    """Reject obviously-broken decay knob values at settings construction."""

    def test_inactive_days_must_be_positive(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011 — pydantic surfaces ValidationError(ValueError)
            Settings(dream_decay_inactive_days=0)  # type: ignore[arg-type]

    def test_thresholds_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            Settings(dream_decay_stale_threshold=1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError):  # noqa: PT011
            Settings(dream_decay_archive_threshold=-0.1)  # type: ignore[arg-type]

    def test_batch_cap_must_be_positive(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            Settings(dream_decay_batch_cap=0)  # type: ignore[arg-type]

    def test_reference_floor_rejects_negative(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            Settings(dream_decay_reference_floor=-1)  # type: ignore[arg-type]

    def test_reference_floor_zero_is_allowed(self) -> None:
        # Zero is the documented "disable the gate" sentinel.
        s = Settings(dream_decay_reference_floor=0)  # type: ignore[arg-type]
        assert s.dream_decay_reference_floor == 0


# ---------------------------------------------------------------------------
# Reference-floor gate (Phase 1 — graph-citation popularity)
# ---------------------------------------------------------------------------


class TestReferenceFloorGate:
    """The active-leg gate that protects highly-cited memories from staling.

    Contract under test (decay.py ``_run_leg``):

    * Active candidates with ``reference_count >= dream_decay_reference_floor``
      are skipped regardless of salience — they survive the active→stale leg
      and surface in ``DecayPassResult.skipped_reference_floor``.
    * The gate is leg-specific: stale→archived ignores the floor.
    * The gate honors the threshold: zero candidates with insufficient
      references still transition normally.
    * ``floor=0`` disables the gate entirely.
    """

    @pytest.mark.asyncio
    async def test_active_at_or_above_floor_skipped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        # Would-otherwise-decay candidate: cold + low-confidence + negative
        # feedback. Salience well below 0.30.
        cold_but_cited = _candidate(
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=180),
            confidence=0.1,
            negative_feedback_count=3,
            reference_count=5,  # >= default floor of 3
        )
        _patch_loaders(monkeypatch, active=[cold_but_cited], stale=[])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert result.examined_active == 1
        assert result.transitioned_to_stale == 0
        assert result.skipped_above_threshold == 0
        assert result.skipped_reference_floor == 1
        assert update.await_count == 0

    @pytest.mark.asyncio
    async def test_active_below_floor_still_transitions(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        cold = _candidate(
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=180),
            confidence=0.1,
            negative_feedback_count=3,
            reference_count=2,  # < default floor of 3
        )
        _patch_loaders(monkeypatch, active=[cold], stale=[])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert result.transitioned_to_stale == 1
        assert result.skipped_reference_floor == 0
        assert update.await_count == 1

    @pytest.mark.asyncio
    async def test_stale_leg_ignores_floor(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stale → archived is structural decay; cited stale rows still go."""
        env_id = uuid4()
        cited_stale = _candidate(
            status=MemoryStatus.stale,
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=365),
            confidence=0.0,
            negative_feedback_count=5,
            reference_count=10,  # well above the active-leg floor
        )
        _patch_loaders(monkeypatch, active=[], stale=[cited_stale])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        assert result.transitioned_to_archived == 1
        assert result.skipped_reference_floor == 0
        assert update.await_count == 1

    @pytest.mark.asyncio
    async def test_floor_zero_disables_gate(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        cold_but_cited = _candidate(
            access_count=0,
            last_accessed_at=NOW - dt.timedelta(days=180),
            confidence=0.1,
            negative_feedback_count=3,
            reference_count=999,
        )
        _patch_loaders(monkeypatch, active=[cold_but_cited], stale=[])
        update = _patch_memory_update(monkeypatch)

        settings = Settings(  # type: ignore[arg-type]
            dream_decay_inactive_days=30,
            dream_decay_stale_threshold=0.30,
            dream_decay_archive_threshold=0.10,
            dream_decay_batch_cap=500,
            dream_decay_reference_floor=0,
        )
        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=settings, now=NOW,
        )

        assert result.transitioned_to_stale == 1
        assert result.skipped_reference_floor == 0
        assert update.await_count == 1

    @pytest.mark.asyncio
    async def test_above_threshold_takes_precedence_over_floor_counter(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Floor gate fires first — above-threshold counter only counts rows
        whose salience genuinely cleared the threshold, not gate-saved ones.
        """
        env_id = uuid4()
        hot_and_cited = _candidate(
            access_count=50,
            last_accessed_at=NOW - dt.timedelta(days=180),
            confidence=0.9,
            negative_feedback_count=0,
            reference_count=10,
        )
        _patch_loaders(monkeypatch, active=[hot_and_cited], stale=[])
        update = _patch_memory_update(monkeypatch)

        result = await run_decay(
            env_id, actor_ctx=_ctx(env_id), settings=_settings(), now=NOW,
        )

        # The candidate is hot (would survive on salience), but the floor
        # gate evaluates first → counted in skipped_reference_floor, NOT
        # skipped_above_threshold. The two counters stay individually
        # observable.
        assert result.skipped_reference_floor == 1
        assert result.skipped_above_threshold == 0
        assert result.transitioned_to_stale == 0
        assert update.await_count == 0


# Marker so unused-import doesn't complain when test bodies are simplified.
_ = (Awaitable, Callable)
