"""Unit tests for the dream-worker job orchestration.

Covers the lifecycle wrapper (:func:`run_dream_pass`), the per-mode
dispatcher, advisory-lock helpers, the heartbeat upsert, and the env
discovery loader. Tests use ``monkeypatch`` to substitute the seam
helpers (``_insert_dream_run``, ``_finalize_dream_run``,
``update_heartbeat``, the lock helpers, and the per-mode pass functions
imported into :mod:`dream_worker.jobs`) so no test hits a real DB.

Patterns:

* Lock helpers are patched to return ``True``/``False`` to exercise the
  contended path and the happy path.
* Pass functions are replaced with ``AsyncMock``s returning ``None`` or
  raising a chosen exception — the dispatcher's job is to route + wrap,
  not to run real work.
* ``session_scope`` is patched at the *call sites*' import names
  (``dream_worker.jobs.session_scope``) for the whole-pass outer scope.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from dream_worker.jobs import (
    _MODE_LOCK_KEY,
    HEARTBEAT_SINK,
    DreamMode,
    DreamPassOutcome,
    DreamPassReport,
    _coerce_value,
    _env_lock_key,
    _result_to_dict,
    run_dream_pass,
)
from memory_mcp.identity import AgentContext

NOW = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


def _make_actor_ctx(env_id: UUID) -> AgentContext:
    return AgentContext(
        agent_id=uuid4(),
        agent_name="dream-worker",
        attached_env_ids=[env_id],
        is_default_agent=False,
    )


@dataclass(frozen=True)
class _FakePassResult:
    """Stand-in result dataclass — covers UUID + datetime coercion."""
    env_id: UUID
    proposals_emitted: int = 0
    duration_seconds: float = 0.0
    nested: dict[str, Any] | None = None


def _make_summarizer(kind_value: str = "template") -> MagicMock:
    summarizer = MagicMock()
    summarizer.kind = MagicMock(value=kind_value)
    return summarizer


class _FakeSession:
    """Minimal AsyncSession look-alike for advisory-lock SQL."""
    def __init__(self) -> None:
        self.executed: list[Any] = []

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        self.executed.append((stmt, params))
        # pg_try_advisory_lock returns a row with .locked=True
        result = MagicMock()
        result.first.return_value = MagicMock(locked=True)
        return result


class _FakeContextManager:
    """Async context manager wrapping a single _FakeSession."""
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *_args: Any) -> None:
        return None


def _patch_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lock_acquired: bool = True,
) -> dict[str, AsyncMock]:
    """Patch the seam helpers and return mocks for assertion."""
    mocks: dict[str, AsyncMock] = {
        "try_lock": AsyncMock(return_value=lock_acquired),
        "release_lock": AsyncMock(return_value=None),
        "insert_run": AsyncMock(return_value=uuid4()),
        "finalize_run": AsyncMock(return_value=None),
        "heartbeat": AsyncMock(return_value=None),
    }
    monkeypatch.setattr(
        "dream_worker.jobs._try_acquire_lock_in_session", mocks["try_lock"],
    )
    monkeypatch.setattr(
        "dream_worker.jobs._release_lock_in_session", mocks["release_lock"],
    )
    monkeypatch.setattr(
        "dream_worker.jobs._insert_dream_run", mocks["insert_run"],
    )
    monkeypatch.setattr(
        "dream_worker.jobs._finalize_dream_run", mocks["finalize_run"],
    )
    monkeypatch.setattr(
        "dream_worker.jobs.update_heartbeat", mocks["heartbeat"],
    )
    # Outer session_scope is replaced with a fake CM so the lock helpers
    # have something to work with.
    fake_session = _FakeSession()
    monkeypatch.setattr(
        "dream_worker.jobs.session_scope",
        lambda: _FakeContextManager(fake_session),
    )
    return mocks


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestLockKeyDerivation:
    def test_env_lock_key_deterministic(self) -> None:
        e = UUID("12345678-1234-1234-1234-123456789abc")
        assert _env_lock_key(e) == _env_lock_key(e)

    def test_env_lock_key_fits_in_int31(self) -> None:
        e = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
        k = _env_lock_key(e)
        assert 0 <= k < 2**31

    def test_different_envs_different_keys(self) -> None:
        e1 = UUID("00000000-0000-0000-0000-000000000001")
        e2 = UUID("00000000-0000-0000-0000-000000000002")
        # NB: the XOR-fold can collide in principle but not for these.
        assert _env_lock_key(e1) != _env_lock_key(e2)

    def test_each_mode_has_distinct_lock_key(self) -> None:
        keys = [_MODE_LOCK_KEY[m] for m in DreamMode]
        assert len(set(keys)) == len(keys)


class TestResultCoercion:
    def test_result_to_dict_stringifies_uuid(self) -> None:
        env_id = UUID("12345678-1234-1234-1234-123456789abc")
        r = _FakePassResult(env_id=env_id, proposals_emitted=3)
        out = _result_to_dict(r)
        assert out["env_id"] == str(env_id)
        assert out["proposals_emitted"] == 3
        assert out["duration_seconds"] == 0.0

    def test_coerce_handles_nested_dict_with_uuid(self) -> None:
        env_id = UUID("12345678-1234-1234-1234-123456789abc")
        v = {"id": env_id, "items": [env_id, "x", 1]}
        out = _coerce_value(v)
        assert out == {"id": str(env_id), "items": [str(env_id), "x", 1]}

    def test_coerce_handles_datetime(self) -> None:
        out = _coerce_value(NOW)
        assert out == NOW.isoformat()

    def test_coerce_passes_primitives_through(self) -> None:
        assert _coerce_value(None) is None
        assert _coerce_value(True) is True
        assert _coerce_value(0.5) == 0.5
        assert _coerce_value("hi") == "hi"


# ---------------------------------------------------------------------------
# run_dream_pass: skipped path
# ---------------------------------------------------------------------------


class TestRunDreamPassSkipped:
    @pytest.mark.asyncio
    async def test_lock_not_acquired_returns_skipped_no_dream_run(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        ctx = _make_actor_ctx(env_id)
        mocks = _patch_lifecycle(monkeypatch, lock_acquired=False)

        result = await run_dream_pass(
            env_id, DreamMode.decay,
            actor_ctx=ctx,
            summarizer=_make_summarizer(),
            now=NOW,
        )

        assert result.outcome == DreamPassOutcome.skipped
        assert result.dream_run_id is None
        assert result.summary is None
        # Skipped path: no dream_runs INSERT, no heartbeat, no release.
        mocks["insert_run"].assert_not_called()
        mocks["finalize_run"].assert_not_called()
        mocks["heartbeat"].assert_not_called()
        mocks["release_lock"].assert_not_called()


# ---------------------------------------------------------------------------
# run_dream_pass: success path
# ---------------------------------------------------------------------------


class TestRunDreamPassSuccess:
    @pytest.mark.asyncio
    async def test_decay_dispatches_to_run_decay(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        ctx = _make_actor_ctx(env_id)
        mocks = _patch_lifecycle(monkeypatch)

        fake_result = _FakePassResult(env_id=env_id, proposals_emitted=5)
        decay_mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr("dream_worker.jobs.run_decay", decay_mock)

        result = await run_dream_pass(
            env_id, DreamMode.decay,
            actor_ctx=ctx,
            summarizer=_make_summarizer(),
            now=NOW,
        )

        assert result.outcome == DreamPassOutcome.done
        assert result.dream_run_id is not None
        assert result.summary == {
            "env_id": str(env_id),
            "proposals_emitted": 5,
            "duration_seconds": 0.0,
            "nested": None,
        }
        decay_mock.assert_called_once()
        kwargs = decay_mock.call_args.kwargs
        assert kwargs["actor_ctx"] is ctx
        # decay does NOT receive dream_run_id (it doesn't write proposals).
        assert "dream_run_id" not in kwargs

        # Lifecycle hooks all fired in order.
        mocks["insert_run"].assert_awaited_once()
        mocks["finalize_run"].assert_awaited_once()
        mocks["heartbeat"].assert_awaited_once()
        mocks["release_lock"].assert_awaited_once()

        # Heartbeat marked success.
        hb_kwargs = mocks["heartbeat"].await_args.kwargs
        assert hb_kwargs["success"] is True
        assert hb_kwargs["last_error"] is None

    @pytest.mark.asyncio
    async def test_dedupe_dispatches_with_qdrant_and_embedder(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        ctx = _make_actor_ctx(env_id)
        _patch_lifecycle(monkeypatch)

        fake_result = _FakePassResult(env_id=env_id)
        dedupe_mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr("dream_worker.jobs.run_dedupe", dedupe_mock)

        embedder = MagicMock()
        vector_store = MagicMock()

        result = await run_dream_pass(
            env_id, DreamMode.dedupe,
            actor_ctx=ctx,
            summarizer=_make_summarizer(),
            embedder=embedder,
            vector_store=vector_store,
            now=NOW,
        )

        assert result.outcome == DreamPassOutcome.done
        dedupe_mock.assert_called_once()
        kwargs = dedupe_mock.call_args.kwargs
        assert kwargs["qdrant"] is vector_store
        assert kwargs["embedder"] is embedder
        assert kwargs["dream_run_id"] is not None

    @pytest.mark.asyncio
    async def test_promote_dispatches_with_summarizer(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        ctx = _make_actor_ctx(env_id)
        _patch_lifecycle(monkeypatch)

        fake_result = _FakePassResult(env_id=env_id, proposals_emitted=2)
        promote_mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr("dream_worker.jobs.run_promote", promote_mock)

        summarizer = _make_summarizer("llm")

        result = await run_dream_pass(
            env_id, DreamMode.promote,
            actor_ctx=ctx,
            summarizer=summarizer,
            now=NOW,
        )

        assert result.outcome == DreamPassOutcome.done
        promote_mock.assert_called_once()
        kwargs = promote_mock.call_args.kwargs
        assert kwargs["summarizer"] is summarizer
        assert kwargs["dream_run_id"] is not None


# ---------------------------------------------------------------------------
# run_dream_pass: failure path
# ---------------------------------------------------------------------------


class TestRunDreamPassFailure:
    @pytest.mark.asyncio
    async def test_pass_exception_marks_run_failed_and_lock_released(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        ctx = _make_actor_ctx(env_id)
        mocks = _patch_lifecycle(monkeypatch)

        decay_mock = AsyncMock(side_effect=RuntimeError("kaboom"))
        monkeypatch.setattr("dream_worker.jobs.run_decay", decay_mock)

        result = await run_dream_pass(
            env_id, DreamMode.decay,
            actor_ctx=ctx,
            summarizer=_make_summarizer(),
            now=NOW,
        )

        assert result.outcome == DreamPassOutcome.failed
        assert result.last_error is not None
        assert "RuntimeError" in result.last_error
        assert "kaboom" in result.last_error

        # finalize_dream_run called with status='failed' and last_error set.
        mocks["finalize_run"].assert_awaited_once()
        finalize_kwargs = mocks["finalize_run"].await_args.kwargs
        assert finalize_kwargs["status"] == "failed"
        assert "kaboom" in finalize_kwargs["last_error"]

        # Heartbeat marked error, but last_error preserved.
        hb_kwargs = mocks["heartbeat"].await_args.kwargs
        assert hb_kwargs["success"] is False
        assert hb_kwargs["last_error"] is not None

        # Lock released even on exception.
        mocks["release_lock"].assert_awaited_once()


# ---------------------------------------------------------------------------
# run_dream_pass: invariant checks
# ---------------------------------------------------------------------------


class TestRunDreamPassInvariants:
    @pytest.mark.asyncio
    async def test_env_not_in_attached_envs_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        # AgentContext does NOT include env_id.
        ctx = AgentContext(
            agent_id=uuid4(),
            attached_env_ids=[uuid4()],  # different env
            is_default_agent=False,
        )

        with pytest.raises(ValueError, match="attached_env_ids"):
            await run_dream_pass(
                env_id, DreamMode.decay,
                actor_ctx=ctx,
                summarizer=_make_summarizer(),
                now=NOW,
            )

    @pytest.mark.asyncio
    async def test_dedupe_without_vector_store_raises_inside_dispatch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dedupe NEEDS qdrant + embedder; if missing, the dispatch
        layer raises and the run is marked failed (NOT crashing the
        worker)."""
        env_id = uuid4()
        ctx = _make_actor_ctx(env_id)
        mocks = _patch_lifecycle(monkeypatch)

        result = await run_dream_pass(
            env_id, DreamMode.dedupe,
            actor_ctx=ctx,
            summarizer=_make_summarizer(),
            # vector_store + embedder intentionally missing
            now=NOW,
        )

        assert result.outcome == DreamPassOutcome.failed
        assert "vector_store" in (result.last_error or "")
        # Run lifecycle was honored — finalize + heartbeat + release all called.
        mocks["finalize_run"].assert_awaited_once()
        mocks["heartbeat"].assert_awaited_once()
        mocks["release_lock"].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_summarizer_kind_recorded_on_dream_run(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        ctx = _make_actor_ctx(env_id)
        mocks = _patch_lifecycle(monkeypatch)
        monkeypatch.setattr(
            "dream_worker.jobs.run_promote",
            AsyncMock(return_value=_FakePassResult(env_id=env_id)),
        )

        await run_dream_pass(
            env_id, DreamMode.promote,
            actor_ctx=ctx,
            summarizer=_make_summarizer("llm"),
            now=NOW,
        )

        insert_kwargs = mocks["insert_run"].await_args.kwargs
        assert insert_kwargs["summarizer_kind"] == "llm"
        assert insert_kwargs["mode"] == DreamMode.promote
        assert insert_kwargs["env_id"] == env_id

    @pytest.mark.asyncio
    async def test_triggered_by_threaded_through(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        ctx = _make_actor_ctx(env_id)
        mocks = _patch_lifecycle(monkeypatch)
        monkeypatch.setattr(
            "dream_worker.jobs.run_decay",
            AsyncMock(return_value=_FakePassResult(env_id=env_id)),
        )

        await run_dream_pass(
            env_id, DreamMode.decay,
            actor_ctx=ctx,
            summarizer=_make_summarizer(),
            triggered_by="manual",
            now=NOW,
        )

        assert mocks["insert_run"].await_args.kwargs["triggered_by"] == "manual"


# ---------------------------------------------------------------------------
# Heartbeat sink name format
# ---------------------------------------------------------------------------


class TestHeartbeatSinkName:
    def test_heartbeat_sink_constant(self) -> None:
        assert HEARTBEAT_SINK == "dream_worker"

    def test_each_mode_has_distinct_sink(self) -> None:
        # Sink names are constructed as "dream_worker:<mode>".
        sinks = [f"{HEARTBEAT_SINK}:{m.value}" for m in DreamMode]
        assert sinks == [
            "dream_worker:decay",
            "dream_worker:dedupe",
            "dream_worker:promote",
            "dream_worker:decision_conflicts",
        ]
        assert len(set(sinks)) == len(sinks)


# ---------------------------------------------------------------------------
# DreamPassReport
# ---------------------------------------------------------------------------


class TestDreamPassReport:
    def test_default_values(self) -> None:
        env_id = uuid4()
        r = DreamPassReport(
            env_id=env_id, mode=DreamMode.decay, outcome=DreamPassOutcome.skipped,
        )
        assert r.dream_run_id is None
        assert r.summary is None
        assert r.last_error is None
        assert r.duration_seconds == 0.0
