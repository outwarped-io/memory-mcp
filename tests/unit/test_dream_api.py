"""Unit tests for ``memory_mcp.dream.api`` — flow-control coverage.

These tests cover the parts of the dream tool surface that are
testable without a real database:

* ``dream_run`` — env list resolution, wait=True/False paths, the
  background-task registry lifecycle, per-pair failure isolation.
* ``dream_status`` — query construction, count aggregation, LLM probe
  timeout, heartbeat shape coercion.
* ``dream_proposals_list`` — cursor encoding/decoding, filter
  fingerprint mismatch rejection, keyset ordering.
* ``dream_review`` — action validation (`amend` is rejected),
  not-found handling, status validation (only `open` proposals can be
  reviewed), accept-handler dispatch by ``kind``.

The full SQL-mutating accept paths (``_accept_merge`` /
``_accept_promotion``) are covered by integration / smoke tests
against real Postgres; here we only assert that ``dream_review``
correctly dispatches to them and routes their return values.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from dream_worker.jobs import DreamMode, DreamPassOutcome, DreamPassReport
from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.dream import api as dream_api
from memory_mcp.dream.api import (
    DreamProposalsListRequest,
    DreamReviewPatch,
    DreamReviewRequest,
    DreamRunRequest,
    DreamStatusRequest,
    _decode_cursor,
    _encode_cursor,
    _filters_hash,
    dream_proposals_list,
    dream_review,
    dream_run,
    dream_status,
    get_active_background_tasks,
)
from memory_mcp.errors import (
    InvalidInputError,
    InvalidTransitionError,
    NotFoundError,
    VersionConflictError,
)
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryResponse


def _make_memory_response(env_id: UUID | None = None) -> MemoryResponse:
    """Build a valid :class:`MemoryResponse` for tests that don't care about content."""
    now = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
    return MemoryResponse(
        id=uuid4(),
        env_id=env_id or uuid4(),
        kind=MemoryKind.fact,
        status=MemoryStatus.active,
        title="merged",
        body="merged body",
        tags=[],
        metadata={},
        salience=0.5,
        confidence=0.5,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _ctx(*envs: UUID, agent_id: UUID | None = None) -> AgentContext:
    return AgentContext(
        agent_id=agent_id or uuid4(),
        agent_name="test-agent",
        attached_env_ids=list(envs),
    )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProposal:
    """Stand-in for a ``DreamProposal`` ORM row."""

    def __init__(
        self,
        *,
        proposal_id: UUID | None = None,
        env_id: UUID | None = None,
        kind: str = "merge_candidate",
        status: str = "open",
        payload: dict[str, Any] | None = None,
        summarizer_kind: str | None = "template",
        llm_failed: bool = False,
        dream_run_id: UUID | None = None,
    ) -> None:
        self.id = proposal_id or uuid4()
        self.env_id = env_id or uuid4()
        self.kind = kind
        self.status = status
        self.payload = payload or {}
        self.summarizer_kind = summarizer_kind
        self.llm_failed = llm_failed
        self.dream_run_id = dream_run_id
        now = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
        self.created_at = now
        self.updated_at = now
        self.reviewed_at: dt.datetime | None = None
        self.reviewed_by_agent_id: UUID | None = None
        self.review_action: str | None = None
        self.review_notes: str | None = None


class _FakeSession:
    """Minimal ``AsyncSession`` stand-in.

    Records ``execute`` calls and returns canned results in order.
    Caller wires the result queue per test by appending to
    :attr:`results`. ``add`` / ``flush`` / ``refresh`` are no-ops.
    """

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.results: list[Any] = []

    async def execute(self, stmt: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(stmt)
        if not self.results:
            r = MagicMock()
            r.scalar_one_or_none.return_value = None
            r.scalars.return_value.all.return_value = []
            r.all.return_value = []
            return r
        return self.results.pop(0)

    def add(self, *_: Any, **__: Any) -> None:
        pass

    async def flush(self) -> None:
        pass

    async def refresh(self, *_: Any, **__: Any) -> None:
        pass


@asynccontextmanager
async def _session_scope_returning(session: _FakeSession) -> Any:
    yield session


# ---------------------------------------------------------------------------
# `dream_run`
# ---------------------------------------------------------------------------


class TestDreamRun:
    """Verify env resolution, mode selection, and wait/no-wait paths."""

    @pytest.fixture(autouse=True)
    def _patch_resources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stub heavy resource constructors so a unit run never tries to
        # talk to Qdrant, an embedder model, or an LLM.
        monkeypatch.setattr(dream_api, "build_summarizer", lambda *_a, **_kw: object())
        monkeypatch.setattr(dream_api, "get_embedder", lambda *_a, **_kw: object())
        fake_store = MagicMock()
        fake_store.close = AsyncMock(return_value=None)
        monkeypatch.setattr(
            dream_api,
            "QdrantVectorStore",
            lambda *_a, **_kw: fake_store,
        )

    @pytest.mark.asyncio
    async def test_no_envs_returns_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(dream_api, "list_active_envs", AsyncMock(return_value=[]))
        out = await dream_run(DreamRunRequest(), ctx=_ctx())
        assert out.scheduled == []
        assert out.reports == []

    def test_all_modes_registry_has_four_modes(self) -> None:
        assert (
            DreamMode.decay,
            DreamMode.dedupe,
            DreamMode.promote,
            DreamMode.decision_conflicts,
            DreamMode.recount,
        ) == dream_api._ALL_MODES

    @pytest.mark.asyncio
    async def test_single_env_default_modes_include_decision_conflicts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = uuid4()
        called: list[tuple[UUID, DreamMode]] = []

        async def fake_pass(eid: UUID, mode: DreamMode, **_: Any) -> DreamPassReport:
            called.append((eid, mode))
            return DreamPassReport(
                env_id=eid,
                mode=mode,
                outcome=DreamPassOutcome.done,
            )

        monkeypatch.setattr(dream_api, "run_dream_pass", fake_pass)
        out = await dream_run(
            DreamRunRequest(env_id=env, wait=True),
            ctx=_ctx(env),
        )
        assert {(c[0], c[1]) for c in called} == {
            (env, DreamMode.decay),
            (env, DreamMode.dedupe),
            (env, DreamMode.promote),
            (env, DreamMode.decision_conflicts),
            (env, DreamMode.recount),
        }
        assert len(out.reports) == 5

    @pytest.mark.asyncio
    async def test_explicit_decision_conflicts_receives_vector_store(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = uuid4()
        fake_store = MagicMock(name="vector_store")
        fake_store.close = AsyncMock(return_value=None)
        monkeypatch.setattr(
            dream_api,
            "QdrantVectorStore",
            lambda *_a, **_kw: fake_store,
        )
        run_pass = AsyncMock(
            return_value=DreamPassReport(
                env_id=env,
                mode=DreamMode.decision_conflicts,
                outcome=DreamPassOutcome.done,
            ),
        )
        monkeypatch.setattr(dream_api, "run_dream_pass", run_pass)

        await dream_run(
            DreamRunRequest(
                env_id=env,
                modes=[DreamMode.decision_conflicts],
                wait=True,
            ),
            ctx=_ctx(env),
        )

        kwargs = run_pass.await_args.kwargs
        assert kwargs["embedder"] is None
        assert kwargs["vector_store"] is fake_store

    @pytest.mark.asyncio
    async def test_attached_envs_used_when_no_explicit_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a, b = uuid4(), uuid4()
        called: list[UUID] = []

        async def fake_pass(eid: UUID, _mode: DreamMode, **_: Any) -> DreamPassReport:
            called.append(eid)
            return DreamPassReport(
                env_id=eid,
                mode=DreamMode.decay,
                outcome=DreamPassOutcome.done,
            )

        monkeypatch.setattr(dream_api, "run_dream_pass", fake_pass)
        await dream_run(
            DreamRunRequest(modes=[DreamMode.decay], wait=True),
            ctx=_ctx(a, b),
        )
        assert set(called) == {a, b}

    @pytest.mark.asyncio
    async def test_per_pair_failure_isolated(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = uuid4()
        modes_seen: list[DreamMode] = []

        async def fake_pass(eid: UUID, mode: DreamMode, **_: Any) -> DreamPassReport:
            modes_seen.append(mode)
            if mode is DreamMode.dedupe:
                raise RuntimeError("synthetic dedupe failure")
            return DreamPassReport(
                env_id=eid,
                mode=mode,
                outcome=DreamPassOutcome.done,
            )

        monkeypatch.setattr(dream_api, "run_dream_pass", fake_pass)
        out = await dream_run(
            DreamRunRequest(env_id=env, wait=True),
            ctx=_ctx(env),
        )
        # Subsequent modes still ran:
        assert {
            DreamMode.decay,
            DreamMode.dedupe,
            DreamMode.promote,
            DreamMode.decision_conflicts,
            DreamMode.recount,
        } == set(modes_seen)
        # The failed pass appears as a failed report rather than aborting:
        outcomes = {r.mode: r.outcome for r in out.reports}
        assert outcomes[DreamMode.dedupe] == DreamPassOutcome.failed
        assert outcomes[DreamMode.decay] == DreamPassOutcome.done
        assert outcomes[DreamMode.promote] == DreamPassOutcome.done
        assert outcomes[DreamMode.decision_conflicts] == DreamPassOutcome.done
        assert outcomes[DreamMode.recount] == DreamPassOutcome.done

    @pytest.mark.asyncio
    async def test_wait_false_returns_schedule_and_tracks_task(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = uuid4()
        coordinator_started = asyncio.Event()
        coordinator_release = asyncio.Event()

        async def fake_pass(eid: UUID, mode: DreamMode, **_: Any) -> DreamPassReport:
            coordinator_started.set()
            await coordinator_release.wait()
            return DreamPassReport(
                env_id=eid,
                mode=mode,
                outcome=DreamPassOutcome.done,
            )

        monkeypatch.setattr(dream_api, "run_dream_pass", fake_pass)

        out = await dream_run(
            DreamRunRequest(env_id=env, modes=[DreamMode.decay], wait=False),
            ctx=_ctx(env),
        )
        try:
            assert out.scheduled == [
                dream_api.DreamRunScheduledItem(env_id=env, mode=DreamMode.decay),
            ]
            assert out.reports == []
            await asyncio.wait_for(coordinator_started.wait(), timeout=1.0)
            assert get_active_background_tasks(), "background coordinator must be tracked while in flight"
        finally:
            coordinator_release.set()
            # Drain background work so subsequent tests don't see lingering tasks.
            for task in list(get_active_background_tasks()):
                await asyncio.wait_for(task, timeout=2.0)
        assert not get_active_background_tasks(), "completed background tasks must be removed from the registry"


# ---------------------------------------------------------------------------
# `dream_status`
# ---------------------------------------------------------------------------


class TestDreamStatus:
    @pytest.mark.asyncio
    async def test_aggregates_runs_counts_and_heartbeats(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = uuid4()
        # Stub the three loaders directly so we don't have to hand-build
        # SQLAlchemy result objects for every query.
        monkeypatch.setattr(
            dream_api,
            "_load_last_runs_per_mode",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            dream_api,
            "_load_open_proposal_counts",
            AsyncMock(
                return_value={
                    "merge_candidate": 3,
                    "promotion_candidate": 1,
                    "decay_candidate": 0,
                }
            ),
        )
        monkeypatch.setattr(
            dream_api,
            "_load_dream_heartbeats",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            dream_api,
            "_bounded_llm_probe",
            AsyncMock(return_value={"status": "ok"}),
        )
        monkeypatch.setattr(
            dream_api,
            "session_scope",
            lambda: _session_scope_returning(_FakeSession()),
        )

        out = await dream_status(
            DreamStatusRequest(env_id=env),
            ctx=_ctx(env),
        )
        assert out.open_proposal_counts["merge_candidate"] == 3
        assert out.llm_status == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_llm_probe_timeout_reports_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def slow_probe(*_a: Any, **_kw: Any) -> dict[str, Any]:
            await asyncio.sleep(10.0)
            return {"status": "ok"}

        # Patch the imported probe_llm symbol used inside _bounded_llm_probe.
        # The function imports lazily, so we patch the module attribute.
        from memory_mcp.llm import base as llm_base

        monkeypatch.setattr(llm_base, "probe_llm", slow_probe)

        # Use a tiny timeout via monkeypatch on asyncio.wait_for so
        # the test doesn't need to actually wait 2 seconds.
        original_wait_for = asyncio.wait_for

        async def fast_wait_for(coro: Any, timeout: float) -> Any:  # noqa: ARG001, ASYNC109
            return await original_wait_for(coro, timeout=0.01)

        monkeypatch.setattr(dream_api.asyncio, "wait_for", fast_wait_for)

        from memory_mcp.config import Settings

        result = await dream_api._bounded_llm_probe(
            Settings(_env_file=None),  # type: ignore[call-arg]
        )
        assert result["status"] == "error"
        assert "timed out" in result["error"]


# ---------------------------------------------------------------------------
# `dream_proposals_list` — cursor + filters
# ---------------------------------------------------------------------------


class TestDreamProposalsList:
    def test_cursor_roundtrip(self) -> None:
        state = {
            "created_at": "2026-05-10T12:00:00+00:00",
            "id": str(uuid4()),
            "filters_hash": "x|open||",
        }
        token = _encode_cursor(state)
        assert _decode_cursor(token) == state

    def test_decode_rejects_garbage(self) -> None:
        with pytest.raises(InvalidInputError):
            _decode_cursor("not-json")

    def test_decode_rejects_missing_fields(self) -> None:
        with pytest.raises(InvalidInputError):
            _decode_cursor('{"created_at": "x"}')

    def test_filters_hash_is_stable_across_equal_filters(self) -> None:
        a = DreamProposalsListRequest(status="open")
        b = DreamProposalsListRequest(status="open")
        assert _filters_hash(a) == _filters_hash(b)

    def test_filters_hash_changes_with_status(self) -> None:
        a = DreamProposalsListRequest(status="open")
        b = DreamProposalsListRequest(status="accepted")
        assert _filters_hash(a) != _filters_hash(b)

    @pytest.mark.asyncio
    async def test_reused_cursor_with_changed_filters_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Build a cursor under one filter shape, then call with a different one.
        old_hash = _filters_hash(DreamProposalsListRequest(status="open"))
        token = _encode_cursor(
            {
                "created_at": "2026-05-10T12:00:00+00:00",
                "id": str(uuid4()),
                "filters_hash": old_hash,
            }
        )
        monkeypatch.setattr(
            dream_api,
            "session_scope",
            lambda: _session_scope_returning(_FakeSession()),
        )
        with pytest.raises(InvalidInputError, match="different filter"):
            await dream_proposals_list(
                DreamProposalsListRequest(status="accepted", cursor=token),
                ctx=_ctx(),
            )

    @pytest.mark.asyncio
    async def test_returns_items_and_next_cursor_when_more_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Seed 3 fake rows for limit=2 → expect 2 items + a cursor.
        rows = [_FakeProposal() for _ in range(3)]
        # _FakeProposal default created_at is identical, so vary them so
        # the cursor encoding picks up a stable last id.
        for i, r in enumerate(rows):
            r.created_at = dt.datetime(
                2026,
                5,
                10,
                12,
                0,
                i,
                tzinfo=dt.UTC,
            )

        s = _FakeSession()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = list(rows)
        s.results.append(result_mock)

        monkeypatch.setattr(
            dream_api,
            "session_scope",
            lambda: _session_scope_returning(s),
        )

        out = await dream_proposals_list(
            DreamProposalsListRequest(limit=2),
            ctx=_ctx(),
        )
        assert len(out.items) == 2
        assert out.next_cursor is not None
        decoded = _decode_cursor(out.next_cursor)
        assert decoded["id"] == str(rows[1].id)


# ---------------------------------------------------------------------------
# `dream_review` — flow control
# ---------------------------------------------------------------------------


class TestDreamReview:
    @pytest.fixture
    def _patched_session_scope(self, monkeypatch: pytest.MonkeyPatch):
        s = _FakeSession()

        def _scope() -> Any:
            return _session_scope_returning(s)

        monkeypatch.setattr(dream_api, "session_scope", _scope)
        return s

    @pytest.mark.asyncio
    async def test_amend_action_rejected_with_invalid_input(self) -> None:
        with pytest.raises(InvalidInputError, match="amend"):
            await dream_review(
                DreamReviewRequest(proposal_id=uuid4(), action="amend"),
                ctx=_ctx(),
            )

    @pytest.mark.asyncio
    async def test_proposal_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        # _lock_proposal returns None when row missing.
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(
                side_effect=NotFoundError(
                    "dream_proposal not found",
                )
            ),
        )
        with pytest.raises(NotFoundError):
            await dream_review(
                DreamReviewRequest(proposal_id=uuid4(), action="reject"),
                ctx=_ctx(),
            )

    @pytest.mark.asyncio
    async def test_already_terminal_proposal_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        proposal = _FakeProposal(status="accepted")
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(return_value=proposal),
        )
        with pytest.raises(InvalidTransitionError):
            await dream_review(
                DreamReviewRequest(proposal_id=proposal.id, action="reject"),
                ctx=_ctx(),
            )

    @pytest.mark.asyncio
    async def test_reject_does_not_mutate_memories(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        proposal = _FakeProposal(status="open")
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(return_value=proposal),
        )
        # Spy on the accept handlers — they must NOT be called for reject.
        accept_merge = AsyncMock()
        accept_promo = AsyncMock()
        monkeypatch.setattr(dream_api, "_accept_merge", accept_merge)
        monkeypatch.setattr(dream_api, "_accept_promotion", accept_promo)
        monkeypatch.setattr(
            dream_api,
            "_finalize_proposal_status",
            AsyncMock(return_value=None),
        )

        out = await dream_review(
            DreamReviewRequest(
                proposal_id=proposal.id,
                action="reject",
                notes="not useful",
            ),
            ctx=_ctx(),
        )
        accept_merge.assert_not_called()
        accept_promo.assert_not_called()
        assert out.accepted_memory is None
        assert out.superseded_memory_ids == []

    @pytest.mark.asyncio
    async def test_accept_merge_dispatches_to_merge_handler(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        proposal = _FakeProposal(kind="merge_candidate", status="open")
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(return_value=proposal),
        )

        # Synthesize a merged Memory ORM-like object the handler "would" return.
        merged = MagicMock()
        merged.id = uuid4()
        merged.env_id = proposal.env_id
        merged.kind = MemoryKind.fact.value
        merged.status = MemoryStatus.active.value
        merged.title = "merged"
        merged.body = "merged body"
        merged.version = 1
        merged.created_at = proposal.created_at
        merged.updated_at = proposal.created_at
        merged.salience = 0.5
        merged.confidence = 0.5
        merged.access_count = 0
        merged.last_accessed_at = None
        merged.pinned = False
        merged.negative_feedback_count = 0
        merged.verified_at = None
        merged.expires_at = None
        merged.superseded_by = None
        merged.metadata_ = {}

        superseded_ids = [uuid4(), uuid4()]
        accept_merge = AsyncMock(
            return_value=(merged, ["tag-a", "tag-b"], superseded_ids),
        )
        accept_promo = AsyncMock(side_effect=AssertionError("must not be called"))
        monkeypatch.setattr(dream_api, "_accept_merge", accept_merge)
        monkeypatch.setattr(dream_api, "_accept_promotion", accept_promo)
        monkeypatch.setattr(
            dream_api,
            "_finalize_proposal_status",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            dream_api,
            "_to_response",
            lambda mem, tag_names: _make_memory_response(env_id=proposal.env_id),
        )

        out = await dream_review(
            DreamReviewRequest(proposal_id=proposal.id, action="accept"),
            ctx=_ctx(),
        )
        accept_merge.assert_awaited_once()
        accept_promo.assert_not_called()
        assert out.superseded_memory_ids == superseded_ids
        assert out.accepted_memory is not None

    @pytest.mark.asyncio
    async def test_accept_promotion_dispatches_to_promotion_handler(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        proposal = _FakeProposal(kind="promotion_candidate", status="open")
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(return_value=proposal),
        )

        new_memory = MagicMock()
        new_memory.id = uuid4()
        new_memory.env_id = proposal.env_id

        accept_merge = AsyncMock(side_effect=AssertionError("must not be called"))
        accept_promo = AsyncMock(return_value=(new_memory, []))
        monkeypatch.setattr(dream_api, "_accept_merge", accept_merge)
        monkeypatch.setattr(dream_api, "_accept_promotion", accept_promo)
        monkeypatch.setattr(
            dream_api,
            "_finalize_proposal_status",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            dream_api,
            "_to_response",
            lambda *a, **kw: _make_memory_response(env_id=proposal.env_id),
        )

        out = await dream_review(
            DreamReviewRequest(proposal_id=proposal.id, action="accept"),
            ctx=_ctx(),
        )
        accept_promo.assert_awaited_once()
        accept_merge.assert_not_called()
        assert out.accepted_memory is not None
        assert out.superseded_memory_ids == []

    @pytest.mark.asyncio
    async def test_accept_merge_propagates_version_conflict(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        proposal = _FakeProposal(kind="merge_candidate", status="open")
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(return_value=proposal),
        )
        accept_merge = AsyncMock(
            side_effect=VersionConflictError(expected=1, actual=2),
        )
        monkeypatch.setattr(dream_api, "_accept_merge", accept_merge)
        monkeypatch.setattr(
            dream_api,
            "_finalize_proposal_status",
            AsyncMock(side_effect=AssertionError("must not be called on failure")),
        )

        with pytest.raises(VersionConflictError):
            await dream_review(
                DreamReviewRequest(proposal_id=proposal.id, action="accept"),
                ctx=_ctx(),
            )

    @pytest.mark.asyncio
    async def test_unknown_proposal_kind_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        # Schema enum forbids this in real DB; defense-in-depth here.
        proposal = _FakeProposal(kind="unknown_kind", status="open")
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(return_value=proposal),
        )
        monkeypatch.setattr(
            dream_api,
            "_finalize_proposal_status",
            AsyncMock(return_value=None),
        )
        with pytest.raises(InvalidInputError, match="unknown proposal kind"):
            await dream_review(
                DreamReviewRequest(proposal_id=proposal.id, action="accept"),
                ctx=_ctx(),
            )

    @pytest.mark.asyncio
    async def test_decay_candidate_accept_is_noop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        proposal = _FakeProposal(kind="decay_candidate", status="open")
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(return_value=proposal),
        )
        accept_merge = AsyncMock(side_effect=AssertionError("must not be called"))
        accept_promo = AsyncMock(side_effect=AssertionError("must not be called"))
        monkeypatch.setattr(dream_api, "_accept_merge", accept_merge)
        monkeypatch.setattr(dream_api, "_accept_promotion", accept_promo)
        monkeypatch.setattr(
            dream_api,
            "_finalize_proposal_status",
            AsyncMock(return_value=None),
        )

        out = await dream_review(
            DreamReviewRequest(proposal_id=proposal.id, action="accept"),
            ctx=_ctx(),
        )
        assert out.accepted_memory is None
        assert out.superseded_memory_ids == []


# ---------------------------------------------------------------------------
# Validation in the merge accept payload parser (independent of DB)
# ---------------------------------------------------------------------------


class TestAcceptMergePayloadValidation:
    """The early-validation path of ``_accept_merge`` runs before any
    DB locking. We exercise the malformed-payload branches without
    needing a real session."""

    @pytest.mark.asyncio
    async def test_missing_primary_id_raises_invalid_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        proposal = _FakeProposal(
            kind="merge_candidate",
            payload={"candidate_ids": [str(uuid4())]},
        )
        with pytest.raises(InvalidInputError, match="missing or malformed"):
            await dream_api._accept_merge(
                _FakeSession(),  # type: ignore[arg-type]
                proposal=proposal,  # type: ignore[arg-type]
                ctx=_ctx(),
                patch=None,
                expected_versions={},
                settings=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_empty_candidate_ids_raises_invalid_input(self) -> None:
        proposal = _FakeProposal(
            kind="merge_candidate",
            payload={"primary_id": str(uuid4()), "candidate_ids": []},
        )
        with pytest.raises(InvalidInputError, match="no candidate_ids"):
            await dream_api._accept_merge(
                _FakeSession(),  # type: ignore[arg-type]
                proposal=proposal,  # type: ignore[arg-type]
                ctx=_ctx(),
                patch=None,
                expected_versions={},
                settings=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_primary_in_candidates_raises_invalid_input(self) -> None:
        primary = uuid4()
        proposal = _FakeProposal(
            kind="merge_candidate",
            payload={
                "primary_id": str(primary),
                "candidate_ids": [str(primary), str(uuid4())],
            },
        )
        with pytest.raises(InvalidInputError, match="includes primary"):
            await dream_api._accept_merge(
                _FakeSession(),  # type: ignore[arg-type]
                proposal=proposal,  # type: ignore[arg-type]
                ctx=_ctx(),
                patch=None,
                expected_versions={},
                settings=MagicMock(),
            )


class TestAcceptPromotionPayloadValidation:
    @pytest.mark.asyncio
    async def test_empty_observation_ids_raises_invalid_input(self) -> None:
        proposal = _FakeProposal(
            kind="promotion_candidate",
            payload={"observation_ids": [], "target_kind": "fact"},
        )
        with pytest.raises(InvalidInputError, match="no observation_ids"):
            await dream_api._accept_promotion(
                _FakeSession(),  # type: ignore[arg-type]
                proposal=proposal,  # type: ignore[arg-type]
                ctx=_ctx(),
                patch=None,
                expected_versions={},
                settings=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_malformed_uuid_raises_invalid_input(self) -> None:
        proposal = _FakeProposal(
            kind="promotion_candidate",
            payload={"observation_ids": ["not-a-uuid"], "target_kind": "fact"},
        )
        with pytest.raises(InvalidInputError, match="malformed"):
            await dream_api._accept_promotion(
                _FakeSession(),  # type: ignore[arg-type]
                proposal=proposal,  # type: ignore[arg-type]
                ctx=_ctx(),
                patch=None,
                expected_versions={},
                settings=MagicMock(),
            )


# ---------------------------------------------------------------------------
# Patch model — exercise structural invariants
# ---------------------------------------------------------------------------


class TestDreamReviewPatch:
    def test_confidence_must_be_in_range(self) -> None:
        with pytest.raises(ValueError):
            DreamReviewPatch(confidence=1.5)

    def test_all_fields_optional(self) -> None:
        # An empty patch is legal — just means "use payload + fallbacks".
        patch = DreamReviewPatch()
        assert patch.title is None and patch.body is None and patch.confidence is None
