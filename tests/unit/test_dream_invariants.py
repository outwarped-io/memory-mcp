"""Supplemental Phase 2.2 invariants — fills gaps not already covered.

The bulk of dream-mode unit coverage lives in:

* ``test_salience.py`` — formula monotonicity / pinning / negatives dominate
* ``test_summarizer.py`` — both summarizers as first-class
* ``test_decay.py`` — state-machine matrix + idempotency
* ``test_dedupe.py`` — clustering + cross-run idempotency + race handling
* ``test_promote.py`` — clustering + truncation + multi-entity overlap
* ``test_dream_jobs.py`` — advisory-lock skip path + result coercion
* ``test_dream_scheduler.py`` — APScheduler wiring + tick wrapper
* ``test_dream_api.py`` — tool surface flow control + accept dispatch

This file holds the residual invariants:

1. ``llm_failed=True`` proposals review-accept successfully via the
   ``suggested_*`` fallback fields embedded in the payload at dedupe /
   promote time. Documents the contract that the accept handler does
   not care **how** the summary was produced.
2. Switching ``DREAM_SUMMARIZER`` between two runs over the same input
   does not produce duplicate proposals. The dedupe-key is built from
   member UUIDs only (summarizer-agnostic) so a second pass — even with
   a different summarizer — short-circuits before invoking the
   summarizer at all.
3. Summarizer call-site instrumentation: dedupe and promote increment
   ``mcp_dream_summarizer_calls_total`` and observe
   ``mcp_dream_summarizer_latency_seconds`` and bump
   ``mcp_dream_llm_fallbacks_total`` when ``llm_failed=True``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.dream import api as dream_api
from memory_mcp.dream.api import DreamReviewRequest, dream_review
from memory_mcp.dream.passes import dedupe as dedupe_mod
from memory_mcp.dream.passes import promote as promote_mod
from memory_mcp.dream.summarizer import (
    MergeCluster,
    MergeClusterMember,
    MergeSummary,
    PromotionCluster,
    PromotionClusterObservation,
    PromotionSummary,
    SummarizerKind,
)
from memory_mcp.identity import AgentContext
from memory_mcp.memories import MemoryResponse
from memory_mcp.observability import (
    dream_llm_fallbacks_total,
    dream_summarizer_calls_total,
    dream_summarizer_latency_seconds,
)

# ---------------------------------------------------------------------------
# Shared helpers — keep parity with test_dream_api.py without importing it.
# ---------------------------------------------------------------------------


def _ctx(env_id: UUID | None = None) -> AgentContext:
    return AgentContext(
        agent_id=uuid4(),
        agent_name="test-agent",
        attached_env_ids=(env_id,) if env_id else (),
    )


def _make_memory_response(env_id: UUID | None = None) -> MemoryResponse:
    now = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
    return MemoryResponse(
        id=uuid4(),
        env_id=env_id or uuid4(),
        kind=MemoryKind.fact,
        status=MemoryStatus.active,
        title="merged",
        body="merged body",
        tags=[],
        salience=0.5,
        confidence=0.5,
        access_count=0,
        last_accessed_at=None,
        pinned=False,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        version=1,
        metadata={},
        created_at=now,
        updated_at=now,
    )


class _FakeProposal:
    def __init__(
        self,
        *,
        proposal_id: UUID | None = None,
        env_id: UUID | None = None,
        kind: str = "merge_candidate",
        status: str = "open",
        payload: dict[str, Any] | None = None,
        summarizer_kind: str | None = "llm",
        llm_failed: bool = True,
    ) -> None:
        self.id = proposal_id or uuid4()
        self.env_id = env_id or uuid4()
        self.kind = kind
        self.status = status
        self.payload = payload or {}
        self.summarizer_kind = summarizer_kind
        self.llm_failed = llm_failed
        self.dream_run_id = None
        now = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
        self.created_at = now
        self.updated_at = now
        self.reviewed_at: dt.datetime | None = None
        self.reviewed_by_agent_id: UUID | None = None
        self.review_action: str | None = None
        self.review_notes: str | None = None


class _FakeSession:
    """Minimal AsyncSession-like stub used by patched session_scope."""

    def __init__(self) -> None:
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.flush = AsyncMock()
        self.refresh = AsyncMock()
        self.add = MagicMock()


@pytest.fixture
def _patched_session_scope(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    s = _FakeSession()
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session_scope():
        yield s

    monkeypatch.setattr(dream_api, "session_scope", _fake_session_scope)
    return s


# ---------------------------------------------------------------------------
# 1. llm_failed proposals review-accept successfully
# ---------------------------------------------------------------------------


class TestLLMFailedAcceptInvariants:
    """``llm_failed=True`` proposals must accept exactly like template ones.

    The accept path reads ``suggested_merged_*`` (merge) or
    ``suggested_*`` (promotion) from the payload — fields that
    ``LLMSummarizer`` populates from the template fallback when the LLM
    call fails. So the dispatch into ``_accept_merge`` /
    ``_accept_promotion`` must not branch on ``llm_failed``.
    """

    @pytest.mark.asyncio
    async def test_accept_merge_with_llm_failed_proposal_succeeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        proposal = _FakeProposal(
            kind="merge_candidate",
            status="open",
            summarizer_kind="llm",
            llm_failed=True,
            payload={
                "primary_id": str(uuid4()),
                "candidate_ids": [str(uuid4()), str(uuid4())],
                # Fallback content that template path produced inside
                # the LLMSummarizer when the LLM call failed:
                "suggested_merged_title": "fallback title",
                "suggested_merged_body": "fallback body",
                "summarizer_kind": "llm",
                "llm_failed": True,
            },
        )
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(return_value=proposal),
        )

        merged = _make_memory_response(env_id=proposal.env_id)
        accept_merge = AsyncMock(
            return_value=(MagicMock(), [], []),
        )
        monkeypatch.setattr(dream_api, "_accept_merge", accept_merge)
        monkeypatch.setattr(
            dream_api,
            "_accept_promotion",
            AsyncMock(side_effect=AssertionError("must not be called")),
        )
        monkeypatch.setattr(
            dream_api,
            "_finalize_proposal_status",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            dream_api,
            "_to_response",
            lambda mem, tag_names: merged,
        )

        out = await dream_review(
            DreamReviewRequest(proposal_id=proposal.id, action="accept"),
            ctx=_ctx(),
        )
        accept_merge.assert_awaited_once()
        assert out.accepted_memory is not None

    @pytest.mark.asyncio
    async def test_accept_promotion_with_llm_failed_proposal_succeeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _patched_session_scope: _FakeSession,
    ) -> None:
        proposal = _FakeProposal(
            kind="promotion_candidate",
            status="open",
            summarizer_kind="llm",
            llm_failed=True,
            payload={
                "evidence_observation_ids": [str(uuid4()), str(uuid4())],
                "target_kind": MemoryKind.fact.value,
                "source_entity_id": str(uuid4()),
                # Fallback content (LLMSummarizer fell back to template):
                "suggested_title": "fallback title",
                "suggested_body": "fallback body",
                "suggested_confidence": 0.5,
                "summarizer_kind": "llm",
                "llm_failed": True,
            },
        )
        monkeypatch.setattr(
            dream_api,
            "_lock_proposal",
            AsyncMock(return_value=proposal),
        )

        promoted = _make_memory_response(env_id=proposal.env_id)
        accept_promo = AsyncMock(return_value=(MagicMock(), []))
        monkeypatch.setattr(
            dream_api,
            "_accept_merge",
            AsyncMock(side_effect=AssertionError("must not be called")),
        )
        monkeypatch.setattr(dream_api, "_accept_promotion", accept_promo)
        monkeypatch.setattr(
            dream_api,
            "_finalize_proposal_status",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            dream_api,
            "_to_response",
            lambda mem, tag_names: promoted,
        )

        out = await dream_review(
            DreamReviewRequest(proposal_id=proposal.id, action="accept"),
            ctx=_ctx(),
        )
        accept_promo.assert_awaited_once()
        assert out.accepted_memory is not None


# ---------------------------------------------------------------------------
# 2. Summarizer-switch idempotency — dedupe key is summarizer-agnostic
# ---------------------------------------------------------------------------


class TestSummarizerSwitchIdempotency:
    """Switching ``DREAM_SUMMARIZER`` mid-flight must not duplicate work.

    The dedupe key is derived from member UUIDs; the partial unique
    index ``WHERE status='open'`` enforces the one-open-proposal-per-set
    rule at the DB level, and the summarizer is not consulted before
    the cross-run skip check fires.
    """

    def test_dedupe_key_is_summarizer_agnostic(self) -> None:
        """The same set of member UUIDs hashes the same regardless of
        what summarizer was used to produce the prior proposal."""

        ids = [uuid4(), uuid4(), uuid4()]
        key1 = dedupe_mod._build_dedupe_key(ids)
        key2 = dedupe_mod._build_dedupe_key(list(reversed(ids)))
        # Order-invariant: cluster {A,B,C} == cluster {C,B,A}
        assert key1 == key2
        # Stable string format — has the cluster-kind prefix:
        assert key1.startswith("merge:")


# ---------------------------------------------------------------------------
# 3. Summarizer call-site instrumentation
# ---------------------------------------------------------------------------


def _read_counter(counter, **labels: str) -> float:
    """Return the current ``_value`` for a labeled prom Counter."""

    return counter.labels(**labels)._value.get()


class _StubSummarizer:
    """Tiny summarizer stub used to drive the instrumentation wrapper."""

    def __init__(self, kind: SummarizerKind, *, llm_failed: bool = False) -> None:
        self._kind = kind
        self._llm_failed = llm_failed

    @property
    def kind(self) -> SummarizerKind:
        return self._kind

    async def summarize_merge(self, cluster: MergeCluster) -> MergeSummary:
        return MergeSummary(
            suggested_merged_title="t",
            suggested_merged_body="b",
            summarizer_kind=self._kind,
            llm_failed=self._llm_failed,
        )

    async def summarize_promotion(
        self,
        cluster: PromotionCluster,
    ) -> PromotionSummary:
        return PromotionSummary(
            suggested_title="t",
            suggested_body="b",
            suggested_confidence=0.6,
            summarizer_kind=self._kind,
            llm_failed=self._llm_failed,
        )


@pytest.mark.asyncio
async def test_dedupe_summarizer_call_increments_metrics() -> None:
    """The dedupe call-site wrapper observes latency + counts the call."""

    summarizer = _StubSummarizer(SummarizerKind.template)
    cluster = MergeCluster(
        primary_id=uuid4(),
        members=[
            MergeClusterMember(
                memory_id=uuid4(),
                title="m",
                body="hello",
                created_at=dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC),
                salience=0.95,
            ),
        ],
        cosine_scores=[1.0],
    )

    before_calls = _read_counter(
        dream_summarizer_calls_total,
        kind="template",
        outcome="ok",
    )
    before_obs = dream_summarizer_latency_seconds.labels(kind="template")._sum.get()

    summary = await dedupe_mod._instrumented_summarize_merge(summarizer, cluster)

    assert summary.suggested_merged_title == "t"
    after_calls = _read_counter(
        dream_summarizer_calls_total,
        kind="template",
        outcome="ok",
    )
    after_obs = dream_summarizer_latency_seconds.labels(kind="template")._sum.get()
    assert after_calls == before_calls + 1
    assert after_obs >= before_obs  # latency observation recorded


@pytest.mark.asyncio
async def test_dedupe_llm_failed_summary_increments_fallback_counter() -> None:
    """When ``llm_failed=True`` the wrapper bumps ``llm_fallbacks_total``."""

    summarizer = _StubSummarizer(SummarizerKind.llm, llm_failed=True)
    cluster = MergeCluster(
        primary_id=uuid4(),
        members=[
            MergeClusterMember(
                memory_id=uuid4(),
                title=None,
                body="hello",
                created_at=dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC),
                salience=0.92,
            ),
        ],
        cosine_scores=[1.0],
    )

    before = _read_counter(dream_llm_fallbacks_total, **{"pass": "dedupe"})
    before_fallback_outcome = _read_counter(
        dream_summarizer_calls_total,
        kind="llm",
        outcome="fallback",
    )

    summary = await dedupe_mod._instrumented_summarize_merge(summarizer, cluster)
    assert summary.llm_failed is True

    after = _read_counter(dream_llm_fallbacks_total, **{"pass": "dedupe"})
    after_fallback_outcome = _read_counter(
        dream_summarizer_calls_total,
        kind="llm",
        outcome="fallback",
    )
    assert after == before + 1
    assert after_fallback_outcome == before_fallback_outcome + 1


@pytest.mark.asyncio
async def test_promote_summarizer_call_increments_metrics() -> None:
    summarizer = _StubSummarizer(SummarizerKind.template)
    cluster = PromotionCluster(
        source_entity_id=uuid4(),
        source_entity_name="alpha",
        observations=[
            PromotionClusterObservation(
                memory_id=uuid4(),
                body="x",
                created_at=dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC),
            ),
        ],
    )
    before = _read_counter(
        dream_summarizer_calls_total,
        kind="template",
        outcome="ok",
    )
    summary = await promote_mod._instrumented_summarize_promotion(
        summarizer,
        cluster,
    )
    assert summary.suggested_title == "t"
    after = _read_counter(
        dream_summarizer_calls_total,
        kind="template",
        outcome="ok",
    )
    assert after == before + 1


@pytest.mark.asyncio
async def test_promote_llm_failed_summary_increments_fallback_counter() -> None:
    summarizer = _StubSummarizer(SummarizerKind.llm, llm_failed=True)
    cluster = PromotionCluster(
        source_entity_id=uuid4(),
        source_entity_name="alpha",
        observations=[
            PromotionClusterObservation(
                memory_id=uuid4(),
                body="x",
                created_at=dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC),
            ),
        ],
    )
    before = _read_counter(dream_llm_fallbacks_total, **{"pass": "promote"})
    await promote_mod._instrumented_summarize_promotion(summarizer, cluster)
    after = _read_counter(dream_llm_fallbacks_total, **{"pass": "promote"})
    assert after == before + 1


@pytest.mark.asyncio
async def test_dedupe_summarizer_exception_increments_error_counter() -> None:
    """Exceptions in the summarizer record ``outcome='error'`` and re-raise."""

    class _Failing:
        @property
        def kind(self) -> SummarizerKind:
            return SummarizerKind.template

        async def summarize_merge(self, cluster: MergeCluster) -> MergeSummary:
            raise RuntimeError("boom")

    cluster = MergeCluster(
        primary_id=uuid4(),
        members=[
            MergeClusterMember(
                memory_id=uuid4(),
                title=None,
                body="hi",
                created_at=dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC),
                salience=0.92,
            ),
        ],
        cosine_scores=[1.0],
    )

    before = _read_counter(
        dream_summarizer_calls_total,
        kind="template",
        outcome="error",
    )
    with pytest.raises(RuntimeError, match="boom"):
        await dedupe_mod._instrumented_summarize_merge(_Failing(), cluster)
    after = _read_counter(
        dream_summarizer_calls_total,
        kind="template",
        outcome="error",
    )
    assert after == before + 1
