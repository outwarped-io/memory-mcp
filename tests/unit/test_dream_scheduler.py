"""Unit tests for the dream-worker APScheduler integration.

Covers :class:`dream_worker.scheduler.DreamScheduler`:

* Job registration — three jobs with stable ids, correct trigger
  interval, ``coalesce=True``, ``max_instances`` from settings,
  ``misfire_grace_time = cadence_seconds``.
* Tick wrappers — fresh env discovery per tick, per-env dispatch,
  per-env failure isolation, cooperative shutdown halts loop between
  envs.
* Mode-specific dispatch kwargs — ``decay`` and ``promote`` get
  ``embedder=None`` + ``vector_store=None``; ``dedupe`` gets both.
* Manual trigger — ``trigger_now()`` invokes ``run_dream_pass`` with
  ``triggered_by="manual"`` and the same per-env ctx contract.
* Shutdown semantics — ``shutdown()`` is **synchronous** (does not
  ``await`` APScheduler), sets ``stopping`` flag first, and is safe
  to call before ``start()``.

All tests stub out :func:`run_dream_pass` and :func:`list_active_envs`
so no DB I/O happens.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from dream_worker.jobs import DreamMode, DreamPassOutcome, DreamPassReport
from dream_worker.scheduler import (
    JOB_ID_DECAY,
    JOB_ID_DECISION_CONFLICTS,
    JOB_ID_DEDUPE,
    JOB_ID_METRICS_REFRESH,
    JOB_ID_PROMOTE,
    JOB_ID_RECOUNT,
    DreamScheduler,
)
from memory_mcp.identity import AgentContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(
    *,
    decay_seconds: int = 3600,
    dedupe_seconds: int = 1800,
    promote_seconds: int = 7200,
    decision_conflicts_seconds: int = 3600,
    recount_seconds: int = 3600,
    max_instances: int = 1,
    metrics_refresh_seconds: int = 0,
) -> MagicMock:
    """Cheap settings stub. The scheduler only reads cadence + max_instances."""
    s = MagicMock()
    s.dream_decay_cadence_seconds = decay_seconds
    s.dream_dedupe_cadence_seconds = dedupe_seconds
    s.dream_promote_cadence_seconds = promote_seconds
    s.dream_decision_conflicts_cadence_seconds = decision_conflicts_seconds
    s.dream_recount_cadence_seconds = recount_seconds
    s.dream_scheduler_max_instances = max_instances
    s.dream_metrics_refresh_seconds = metrics_refresh_seconds
    return s


def _make_scheduler(
    settings: MagicMock | None = None,
    *,
    agent_id: UUID | None = None,
) -> DreamScheduler:
    settings = settings or _make_settings()
    return DreamScheduler(
        settings,
        summarizer=MagicMock(name="summarizer"),
        embedder=MagicMock(name="embedder"),
        vector_store=MagicMock(name="vector_store"),
        agent_id=agent_id or uuid4(),
        agent_name="dream-worker",
    )


def _ok_report(env_id: UUID, mode: DreamMode) -> DreamPassReport:
    return DreamPassReport(
        env_id=env_id,
        mode=mode,
        outcome=DreamPassOutcome.done,
        duration_seconds=0.01,
    )


# ---------------------------------------------------------------------------
# Job registration
# ---------------------------------------------------------------------------


class TestRegisterJobs:
    @pytest.mark.asyncio
    async def test_jobs_registered_with_stable_ids(self) -> None:
        s = _make_scheduler()
        s.start()
        try:
            ids = {job.id for job in s.scheduler.get_jobs()}
            assert ids == {
                JOB_ID_DECAY,
                JOB_ID_DEDUPE,
                JOB_ID_PROMOTE,
                JOB_ID_DECISION_CONFLICTS,
                JOB_ID_RECOUNT,
            }
        finally:
            s.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_intervals_match_settings(self) -> None:
        settings = _make_settings(
            decay_seconds=120,
            dedupe_seconds=240,
            promote_seconds=360,
            decision_conflicts_seconds=480,
            recount_seconds=600,
        )
        s = _make_scheduler(settings)
        s.start()
        try:
            jobs = {job.id: job for job in s.scheduler.get_jobs()}
            assert int(jobs[JOB_ID_DECAY].trigger.interval.total_seconds()) == 120
            assert int(jobs[JOB_ID_DEDUPE].trigger.interval.total_seconds()) == 240
            assert int(jobs[JOB_ID_PROMOTE].trigger.interval.total_seconds()) == 360
            assert int(jobs[JOB_ID_DECISION_CONFLICTS].trigger.interval.total_seconds()) == 480
            assert int(jobs[JOB_ID_RECOUNT].trigger.interval.total_seconds()) == 600
        finally:
            s.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_jobs_have_coalesce_true(self) -> None:
        s = _make_scheduler()
        s.start()
        try:
            for job in s.scheduler.get_jobs():
                assert job.coalesce is True
        finally:
            s.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_jobs_max_instances_from_settings(self) -> None:
        settings = _make_settings(max_instances=3)
        s = _make_scheduler(settings)
        s.start()
        try:
            for job in s.scheduler.get_jobs():
                assert job.max_instances == 3
        finally:
            s.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_misfire_grace_time_equals_cadence(self) -> None:
        """Plan: ``misfire_grace_time = cadence_seconds`` per RD #11."""
        settings = _make_settings(
            decay_seconds=60,
            dedupe_seconds=120,
            promote_seconds=180,
            decision_conflicts_seconds=240,
            recount_seconds=300,
        )
        s = _make_scheduler(settings)
        s.start()
        try:
            jobs = {job.id: job for job in s.scheduler.get_jobs()}
            assert jobs[JOB_ID_DECAY].misfire_grace_time == 60
            assert jobs[JOB_ID_DEDUPE].misfire_grace_time == 120
            assert jobs[JOB_ID_PROMOTE].misfire_grace_time == 180
            assert jobs[JOB_ID_DECISION_CONFLICTS].misfire_grace_time == 240
            assert jobs[JOB_ID_RECOUNT].misfire_grace_time == 300
        finally:
            s.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_start_twice_raises(self) -> None:
        s = _make_scheduler()
        s.start()
        try:
            with pytest.raises(RuntimeError, match="already called"):
                s.start()
        finally:
            s.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_metrics_refresh_job_registers_when_cadence_positive(
        self,
    ) -> None:
        settings = _make_settings(metrics_refresh_seconds=30)
        s = _make_scheduler(settings)
        s.start()
        try:
            jobs = {job.id: job for job in s.scheduler.get_jobs()}
            assert JOB_ID_METRICS_REFRESH in jobs
            assert (
                int(
                    jobs[JOB_ID_METRICS_REFRESH].trigger.interval.total_seconds(),
                )
                == 30
            )
        finally:
            s.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_metrics_refresh_job_omitted_when_cadence_zero(self) -> None:
        settings = _make_settings(metrics_refresh_seconds=0)
        s = _make_scheduler(settings)
        s.start()
        try:
            ids = {job.id for job in s.scheduler.get_jobs()}
            assert JOB_ID_METRICS_REFRESH not in ids
        finally:
            s.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Tick wrapper — env discovery + dispatch
# ---------------------------------------------------------------------------


class TestRunModeTick:
    @pytest.mark.asyncio
    async def test_no_envs_dispatches_nothing(self, monkeypatch) -> None:
        s = _make_scheduler()
        list_envs = AsyncMock(return_value=[])
        run_pass = AsyncMock()
        monkeypatch.setattr("dream_worker.scheduler.list_active_envs", list_envs)
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s._run_mode_tick(DreamMode.decay)

        assert list_envs.await_count == 1
        assert run_pass.await_count == 0

    @pytest.mark.asyncio
    async def test_iterates_all_envs_in_order(self, monkeypatch) -> None:
        env_ids = [uuid4() for _ in range(3)]
        s = _make_scheduler()
        run_pass = AsyncMock(
            side_effect=lambda env_id, mode, **_: _ok_report(env_id, mode),
        )
        monkeypatch.setattr(
            "dream_worker.scheduler.list_active_envs",
            AsyncMock(return_value=env_ids),
        )
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s._run_mode_tick(DreamMode.decay)

        assert run_pass.await_count == 3
        called_envs = [call.args[0] for call in run_pass.await_args_list]
        assert called_envs == env_ids

    @pytest.mark.asyncio
    async def test_env_discovery_is_per_tick(self, monkeypatch) -> None:
        """list_active_envs called once per tick — fresh each time."""
        s = _make_scheduler()
        list_envs = AsyncMock(
            side_effect=[
                [uuid4()],
                [uuid4(), uuid4()],
            ]
        )
        monkeypatch.setattr("dream_worker.scheduler.list_active_envs", list_envs)
        monkeypatch.setattr(
            "dream_worker.scheduler.run_dream_pass",
            AsyncMock(side_effect=lambda env_id, mode, **_: _ok_report(env_id, mode)),
        )

        await s._run_mode_tick(DreamMode.decay)
        await s._run_mode_tick(DreamMode.decay)

        assert list_envs.await_count == 2

    @pytest.mark.asyncio
    async def test_one_env_failure_does_not_break_loop(self, monkeypatch) -> None:
        """Per-env failure isolation: bad env doesn't poison later envs."""
        env_ids = [uuid4(), uuid4(), uuid4()]
        s = _make_scheduler()

        async def _maybe_fail(env_id, mode, **_):  # noqa: ANN001
            if env_id == env_ids[1]:
                raise RuntimeError("simulated transient")
            return _ok_report(env_id, mode)

        run_pass = AsyncMock(side_effect=_maybe_fail)
        monkeypatch.setattr(
            "dream_worker.scheduler.list_active_envs",
            AsyncMock(return_value=env_ids),
        )
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s._run_mode_tick(DreamMode.decay)

        # All three were attempted despite the middle one raising.
        assert run_pass.await_count == 3

    @pytest.mark.asyncio
    async def test_list_envs_failure_skips_tick(self, monkeypatch) -> None:
        """If env discovery raises, tick exits cleanly without dispatching."""
        s = _make_scheduler()
        run_pass = AsyncMock()
        monkeypatch.setattr(
            "dream_worker.scheduler.list_active_envs",
            AsyncMock(side_effect=RuntimeError("DB pause")),
        )
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        # Must not raise — tick should swallow + log.
        await s._run_mode_tick(DreamMode.decay)

        assert run_pass.await_count == 0

    @pytest.mark.asyncio
    async def test_stop_flag_halts_loop_between_envs(self, monkeypatch) -> None:
        """Set ``stopping`` after first env — remaining envs must skip."""
        env_ids = [uuid4() for _ in range(4)]
        s = _make_scheduler()

        call_count = {"n": 0}

        async def _flip_stop_after_first(env_id, mode, **_):  # noqa: ANN001
            call_count["n"] += 1
            if call_count["n"] == 1:
                s._stopping = True  # simulate SIGTERM
            return _ok_report(env_id, mode)

        run_pass = AsyncMock(side_effect=_flip_stop_after_first)
        monkeypatch.setattr(
            "dream_worker.scheduler.list_active_envs",
            AsyncMock(return_value=env_ids),
        )
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s._run_mode_tick(DreamMode.decay)

        # Only the first env ran; the loop bailed before envs 2..4.
        assert run_pass.await_count == 1


# ---------------------------------------------------------------------------
# Mode-specific dispatch kwargs
# ---------------------------------------------------------------------------


class TestDispatchKwargs:
    @pytest.mark.asyncio
    async def test_decay_passes_no_embedder_no_vector_store(
        self,
        monkeypatch,
    ) -> None:
        env_id = uuid4()
        s = _make_scheduler()
        run_pass = AsyncMock(return_value=_ok_report(env_id, DreamMode.decay))
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s._dispatch_one(env_id, DreamMode.decay)

        kwargs = run_pass.await_args.kwargs
        assert kwargs["embedder"] is None
        assert kwargs["vector_store"] is None
        assert kwargs["triggered_by"] == "scheduler"

    @pytest.mark.asyncio
    async def test_promote_passes_no_embedder_no_vector_store(
        self,
        monkeypatch,
    ) -> None:
        env_id = uuid4()
        s = _make_scheduler()
        run_pass = AsyncMock(return_value=_ok_report(env_id, DreamMode.promote))
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s._dispatch_one(env_id, DreamMode.promote)

        kwargs = run_pass.await_args.kwargs
        assert kwargs["embedder"] is None
        assert kwargs["vector_store"] is None

    @pytest.mark.asyncio
    async def test_dedupe_passes_embedder_and_vector_store(
        self,
        monkeypatch,
    ) -> None:
        env_id = uuid4()
        embedder = MagicMock(name="embedder")
        vector_store = MagicMock(name="vector_store")
        settings = _make_settings()
        s = DreamScheduler(
            settings,
            summarizer=MagicMock(name="summarizer"),
            embedder=embedder,
            vector_store=vector_store,
            agent_id=uuid4(),
            agent_name="dream-worker",
        )
        run_pass = AsyncMock(return_value=_ok_report(env_id, DreamMode.dedupe))
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s._dispatch_one(env_id, DreamMode.dedupe)

        kwargs = run_pass.await_args.kwargs
        assert kwargs["embedder"] is embedder
        assert kwargs["vector_store"] is vector_store

    @pytest.mark.asyncio
    async def test_decision_conflicts_passes_vector_store_only(
        self,
        monkeypatch,
    ) -> None:
        env_id = uuid4()
        vector_store = MagicMock(name="vector_store")
        settings = _make_settings()
        s = DreamScheduler(
            settings,
            summarizer=MagicMock(name="summarizer"),
            embedder=MagicMock(name="embedder"),
            vector_store=vector_store,
            agent_id=uuid4(),
            agent_name="dream-worker",
        )
        run_pass = AsyncMock(return_value=_ok_report(env_id, DreamMode.decision_conflicts))
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s._dispatch_one(env_id, DreamMode.decision_conflicts)

        kwargs = run_pass.await_args.kwargs
        assert kwargs["embedder"] is None
        assert kwargs["vector_store"] is vector_store

    @pytest.mark.asyncio
    async def test_actor_ctx_attached_to_only_one_env(
        self,
        monkeypatch,
    ) -> None:
        env_id = uuid4()
        agent_id = uuid4()
        s = _make_scheduler(agent_id=agent_id)
        run_pass = AsyncMock(return_value=_ok_report(env_id, DreamMode.decay))
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s._dispatch_one(env_id, DreamMode.decay)

        ctx: AgentContext = run_pass.await_args.kwargs["actor_ctx"]
        assert ctx.agent_id == agent_id
        # Critical: only THIS env, not all envs (least privilege).
        assert ctx.attached_env_ids == [env_id]
        assert ctx.is_default_agent is True


# ---------------------------------------------------------------------------
# Manual trigger
# ---------------------------------------------------------------------------


class TestTriggerNow:
    @pytest.mark.asyncio
    async def test_trigger_now_dispatches_immediately(
        self,
        monkeypatch,
    ) -> None:
        env_id = uuid4()
        s = _make_scheduler()
        report = _ok_report(env_id, DreamMode.decay)
        run_pass = AsyncMock(return_value=report)
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        out = await s.trigger_now(env_id, DreamMode.decay)

        assert out is report
        kwargs = run_pass.await_args.kwargs
        assert kwargs["triggered_by"] == "manual"

    @pytest.mark.asyncio
    async def test_trigger_now_dedupe_passes_embedder_vector_store(
        self,
        monkeypatch,
    ) -> None:
        env_id = uuid4()
        embedder = MagicMock(name="embedder")
        vector_store = MagicMock(name="vector_store")
        settings = _make_settings()
        s = DreamScheduler(
            settings,
            summarizer=MagicMock(),
            embedder=embedder,
            vector_store=vector_store,
            agent_id=uuid4(),
        )
        run_pass = AsyncMock(return_value=_ok_report(env_id, DreamMode.dedupe))
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s.trigger_now(env_id, DreamMode.dedupe)

        kwargs = run_pass.await_args.kwargs
        assert kwargs["embedder"] is embedder
        assert kwargs["vector_store"] is vector_store

    @pytest.mark.asyncio
    async def test_trigger_now_decay_omits_embedder_and_vector_store(
        self,
        monkeypatch,
    ) -> None:
        env_id = uuid4()
        s = _make_scheduler()
        run_pass = AsyncMock(return_value=_ok_report(env_id, DreamMode.decay))
        monkeypatch.setattr("dream_worker.scheduler.run_dream_pass", run_pass)

        await s.trigger_now(env_id, DreamMode.decay)

        kwargs = run_pass.await_args.kwargs
        assert kwargs["embedder"] is None
        assert kwargs["vector_store"] is None


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_before_start_is_safe(self) -> None:
        """Calling shutdown() without start() must not raise."""
        s = _make_scheduler()
        s.shutdown(wait=False)
        assert s.stopping is True

    @pytest.mark.asyncio
    async def test_shutdown_sets_stopping_flag(self) -> None:
        s = _make_scheduler()
        assert s.stopping is False
        s.start()
        try:
            s.shutdown(wait=False)
            assert s.stopping is True
        finally:
            # idempotent — already shut down
            pass

    @pytest.mark.asyncio
    async def test_shutdown_is_synchronous(self) -> None:
        """RD blocker #2: APScheduler 3.x .shutdown is sync; we must not await it.

        Asserts ``shutdown`` returns ``None`` (not a coroutine). If a future
        refactor accidentally returned ``self._scheduler.shutdown(...)`` and
        APScheduler returned a coroutine, this test would catch it.
        """
        s = _make_scheduler()
        s.start()
        result = s.shutdown(wait=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_shutdown_calls_apscheduler_with_wait_flag(
        self,
        monkeypatch,
    ) -> None:
        """Forward the wait kwarg to APScheduler verbatim."""
        s = _make_scheduler()
        s.start()
        try:
            calls: list[Any] = []
            real = s._scheduler.shutdown

            def _fake(wait: bool = True) -> None:
                calls.append(wait)
                real(wait=False)  # actually drain, but record the kwarg

            monkeypatch.setattr(s._scheduler, "shutdown", _fake)
            s.shutdown(wait=True)
            assert calls == [True]
        finally:
            # ensure the loop teardown is clean — already called real shutdown
            pass


# ---------------------------------------------------------------------------
# Mode tick + tick callable factory
# ---------------------------------------------------------------------------


class TestTickFactory:
    @pytest.mark.asyncio
    async def test_make_tick_func_names_uniquely_per_mode(self) -> None:
        s = _make_scheduler()
        f_decay = s._make_tick_func(DreamMode.decay)
        f_dedupe = s._make_tick_func(DreamMode.dedupe)
        f_promote = s._make_tick_func(DreamMode.promote)
        assert f_decay.__name__ != f_dedupe.__name__
        assert f_decay.__name__ != f_promote.__name__
        assert f_dedupe.__name__ != f_promote.__name__
        assert "decay" in f_decay.__name__
        assert "dedupe" in f_dedupe.__name__
        assert "promote" in f_promote.__name__

    @pytest.mark.asyncio
    async def test_make_tick_func_invokes_run_mode_tick(
        self,
        monkeypatch,
    ) -> None:
        s = _make_scheduler()
        called: list[DreamMode] = []

        async def _spy(mode):  # noqa: ANN001
            called.append(mode)

        monkeypatch.setattr(s, "_run_mode_tick", _spy)
        f = s._make_tick_func(DreamMode.dedupe)
        await f()
        assert called == [DreamMode.dedupe]
