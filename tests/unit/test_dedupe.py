"""Unit tests for the dream-dedupe pass.

Tests run hermetic — Postgres SELECT, Qdrant search, and the proposal
INSERT are monkeypatched. The summarizer is wired in as a real
:class:`TemplateSummarizer` so the cluster→proposal payload mapping is
exercised end-to-end without an LLM.

Coverage matrix:

* Single seed, one above-threshold neighbor ⇒ 1 proposal with both
  members listed.
* Single seed, no neighbors above threshold ⇒ no proposal (skipped:
  below_min_size).
* Two seeds in the same cluster ⇒ exactly 1 proposal (canonical
  dedupe_key collapses them).
* Existing-proposal idempotency: ``ON CONFLICT DO NOTHING`` returns no
  row from ``RETURNING``; the helper detects this and increments the
  skipped counter.
* Per-run cap: stops emitting after ``DREAM_DEDUPE_BATCH_CAP`` proposals
  and flags ``items_capped``.
* Empty env: clean zeros, no exceptions.
* Below-threshold-only neighbors stay out of the cluster (the seed must
  cluster with at least one above-threshold neighbor).
* Cosine scores in payload mirror the Qdrant order.
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from memory_mcp.config import Settings
from memory_mcp.dream.passes.dedupe import (
    DedupePassResult,
    _build_dedupe_key,
    _select_primary_id,
    run_dedupe,
)
from memory_mcp.dream.summarizer import (
    MergeClusterMember,
    TemplateSummarizer,
)

UTC = dt.UTC
NOW = dt.datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    base = {
        "dream_dedupe_window_days": 7,
        "dream_dedupe_threshold": 0.92,
        "dream_dedupe_top_k": 10,
        "dream_dedupe_batch_cap": 200,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class _FakeSeed:
    """Stand-in for the ``_SeedRow`` dataclass used by the pass."""

    def __init__(
        self,
        *,
        memory_id: UUID | None = None,
        title: str | None = "Seed Title",
        body: str = "Seed body content for embedding",
        salience: float = 0.5,
        kind: str = "fact",
        created_at: dt.datetime | None = None,
    ) -> None:
        from memory_mcp.dream.passes.dedupe import _SeedRow

        self._row = _SeedRow(
            id=memory_id or uuid4(),
            title=title,
            body=body,
            salience=salience,
            kind=kind,
            created_at=created_at or (NOW - dt.timedelta(days=1)),
        )

    def row(self) -> Any:
        return self._row


def _make_qdrant_hit(
    *,
    memory_id: UUID,
    score: float,
    title: str = "Hit Title",
    body: str = "Hit body",
    salience: float = 0.4,
    created_at: dt.datetime | None = None,
) -> dict[str, Any]:
    return {
        "id": str(memory_id),
        "score": score,
        "payload": {
            "title": title,
            "body": body,
            "salience": salience,
            "created_at": (created_at or (NOW - dt.timedelta(days=2))).isoformat(),
        },
    }


def _make_embedder() -> MagicMock:
    e = MagicMock()
    e.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    return e


def _patch_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seeds: list[Any],
) -> AsyncMock:
    loader = AsyncMock(return_value=[s.row() if hasattr(s, "row") else s for s in seeds])
    monkeypatch.setattr(
        "memory_mcp.dream.passes.dedupe._load_seed_rows", loader,
    )
    return loader


def _patch_insert(
    monkeypatch: pytest.MonkeyPatch,
    *,
    side_effect: list[Any] | None = None,
    exists: bool = False,
) -> AsyncMock:
    """Mock ``_insert_proposal``. Also stubs ``_open_proposal_exists`` to
    ``exists`` (default: ``False`` — no existing proposal). Tests that
    want to exercise the pre-summarize skip path should pass
    ``exists=True``.
    """
    insert_mock = (
        AsyncMock(return_value=True) if side_effect is None
        else AsyncMock(side_effect=side_effect)
    )
    monkeypatch.setattr(
        "memory_mcp.dream.passes.dedupe._insert_proposal", insert_mock,
    )
    monkeypatch.setattr(
        "memory_mcp.dream.passes.dedupe._open_proposal_exists",
        AsyncMock(return_value=exists),
    )
    return insert_mock


# ---------------------------------------------------------------------------
# Helper: pure functions
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_dedupe_key_is_deterministic_across_input_order(self) -> None:
        a = uuid4()
        b = uuid4()
        c = uuid4()
        assert _build_dedupe_key([a, b, c]) == _build_dedupe_key([c, b, a])
        assert _build_dedupe_key([a, b, c]) == _build_dedupe_key([b, a, c])
        # Includes the merge: prefix
        assert _build_dedupe_key([a]).startswith("merge:")

    def test_select_primary_picks_highest_salience(self) -> None:
        a = uuid4()
        b = uuid4()
        members = {
            a: MergeClusterMember(
                memory_id=a, title=None, body="x", salience=0.3, created_at=NOW,
            ),
            b: MergeClusterMember(
                memory_id=b, title=None, body="y", salience=0.7, created_at=NOW,
            ),
        }
        assert _select_primary_id(members) == b

    def test_select_primary_uses_lex_tiebreak_on_salience_tie(self) -> None:
        # Construct two UUIDs with deterministic ordering.
        low = UUID("00000000-0000-0000-0000-000000000001")
        high = UUID("ff000000-0000-0000-0000-000000000000")
        members = {
            low: MergeClusterMember(
                memory_id=low, title=None, body="x", salience=0.5, created_at=NOW,
            ),
            high: MergeClusterMember(
                memory_id=high, title=None, body="y", salience=0.5, created_at=NOW,
            ),
        }
        assert _select_primary_id(members) == low


# ---------------------------------------------------------------------------
# Cluster formation
# ---------------------------------------------------------------------------


class TestClusterFormation:
    @pytest.mark.asyncio
    async def test_single_seed_one_neighbor_emits_proposal(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        seed = _FakeSeed()
        neighbor_id = uuid4()
        _patch_loaders(monkeypatch, seeds=[seed])
        insert_mock = _patch_insert(monkeypatch)

        qdrant = MagicMock()
        qdrant.search = AsyncMock(return_value=[
            _make_qdrant_hit(memory_id=neighbor_id, score=0.95),
        ])

        result = await run_dedupe(
            env_id,
            qdrant=qdrant,
            embedder=_make_embedder(),
            summarizer=TemplateSummarizer(),
            settings=_settings(),
            now=NOW,
        )

        assert isinstance(result, DedupePassResult)
        assert result.seeds_examined == 1
        assert result.clusters_found == 1
        assert result.proposals_emitted == 1
        assert result.proposals_skipped_below_min_size == 0
        assert result.summarizer_kind == "template"
        # Verify the cluster payload — both members present.
        call_kwargs = insert_mock.await_args.kwargs
        assert call_kwargs["env_id"] == env_id
        cluster = call_kwargs["cluster"]
        member_ids = {m.memory_id for m in cluster.members}
        assert seed.row().id in member_ids
        assert neighbor_id in member_ids
        # Cosine for the seed itself is 1.0; for the neighbor matches Qdrant.
        scores = dict(zip(
            [m.memory_id for m in cluster.members],
            cluster.cosine_scores,
            strict=True,
        ))
        assert scores[seed.row().id] == pytest.approx(1.0)
        assert scores[neighbor_id] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_no_above_threshold_neighbors_skipped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        _patch_loaders(monkeypatch, seeds=[_FakeSeed()])
        insert_mock = _patch_insert(monkeypatch)

        qdrant = MagicMock()
        qdrant.search = AsyncMock(return_value=[
            _make_qdrant_hit(memory_id=uuid4(), score=0.85),
            _make_qdrant_hit(memory_id=uuid4(), score=0.50),
        ])

        result = await run_dedupe(
            env_id,
            qdrant=qdrant,
            embedder=_make_embedder(),
            summarizer=TemplateSummarizer(),
            settings=_settings(),
            now=NOW,
        )

        assert result.seeds_examined == 1
        assert result.clusters_found == 0
        assert result.proposals_emitted == 0
        assert result.proposals_skipped_below_min_size == 1
        insert_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_seeds_same_cluster_emits_once(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If two seeds happen to be in the same cluster, only one
        proposal is emitted (the second seed's cluster has the same
        dedupe_key)."""
        env_id = uuid4()
        seed_a_id = UUID("11111111-1111-1111-1111-111111111111")
        seed_b_id = UUID("22222222-2222-2222-2222-222222222222")
        seed_a = _FakeSeed(memory_id=seed_a_id)
        seed_b = _FakeSeed(memory_id=seed_b_id)
        _patch_loaders(monkeypatch, seeds=[seed_a, seed_b])
        insert_mock = _patch_insert(monkeypatch)

        qdrant = MagicMock()

        async def search_side_effect(**kwargs: Any) -> list[dict[str, Any]]:
            # Each seed sees the *other* as an above-threshold neighbor.
            # Self is filtered inside the pass.
            return [
                _make_qdrant_hit(memory_id=seed_a_id, score=0.96),
                _make_qdrant_hit(memory_id=seed_b_id, score=0.96),
            ]
        qdrant.search = AsyncMock(side_effect=search_side_effect)

        result = await run_dedupe(
            env_id,
            qdrant=qdrant,
            embedder=_make_embedder(),
            summarizer=TemplateSummarizer(),
            settings=_settings(),
            now=NOW,
        )

        assert result.seeds_examined == 2
        # Both seeds form the same cluster ⇒ 1 proposal, both clusters_found.
        # (clusters_found counts cluster materializations; the second seed's
        # cluster is materialized but skipped via the in-run dedupe_key set.)
        assert result.proposals_emitted == 1
        assert insert_mock.await_count == 1


# ---------------------------------------------------------------------------
# Idempotency (DB unique-index collision)
# ---------------------------------------------------------------------------


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_existing_proposal_skipped_and_counted(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        _patch_loaders(monkeypatch, seeds=[_FakeSeed()])
        insert_mock = _patch_insert(monkeypatch, side_effect=[False])

        qdrant = MagicMock()
        qdrant.search = AsyncMock(return_value=[
            _make_qdrant_hit(memory_id=uuid4(), score=0.95),
        ])

        result = await run_dedupe(
            env_id,
            qdrant=qdrant,
            embedder=_make_embedder(),
            summarizer=TemplateSummarizer(),
            settings=_settings(),
            now=NOW,
        )

        assert result.proposals_emitted == 0
        assert result.proposals_skipped_existing == 1
        assert insert_mock.await_count == 1

    @pytest.mark.asyncio
    async def test_pre_summarize_skip_avoids_summarizer_call(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If an open proposal already covers this cluster, the
        summarizer is NOT called and the insert is NOT attempted —
        rubber-duck finding #2 from the p2.2-dedupe review.
        """
        env_id = uuid4()
        _patch_loaders(monkeypatch, seeds=[_FakeSeed()])
        insert_mock = _patch_insert(monkeypatch, exists=True)

        qdrant = MagicMock()
        qdrant.search = AsyncMock(return_value=[
            _make_qdrant_hit(memory_id=uuid4(), score=0.95),
        ])

        # Wrap a real TemplateSummarizer so we can spy on calls.
        summarizer = TemplateSummarizer()
        summarize_spy = AsyncMock(side_effect=summarizer.summarize_merge)
        summarizer.summarize_merge = summarize_spy  # type: ignore[method-assign]

        result = await run_dedupe(
            env_id,
            qdrant=qdrant,
            embedder=_make_embedder(),
            summarizer=summarizer,
            settings=_settings(),
            now=NOW,
        )

        assert result.proposals_emitted == 0
        assert result.proposals_skipped_existing == 1
        # CRITICAL: neither summarizer nor insert was called.
        summarize_spy.assert_not_called()
        insert_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_conflict_returning_none_returns_false(
        self,
    ) -> None:
        """Direct unit on _insert_proposal with the new ``ON CONFLICT
        DO NOTHING`` flow: when the unique index rejects, ``RETURNING``
        produces ``None``; the function returns ``False`` cleanly with
        no exception.
        """
        from unittest.mock import patch

        from memory_mcp.dream.passes.dedupe import _insert_proposal
        from memory_mcp.dream.summarizer import (
            MergeCluster,
            MergeSummary,
            SummarizerKind,
        )

        a = UUID("00000000-0000-0000-0000-00000000000a")
        b = UUID("00000000-0000-0000-0000-00000000000b")
        cluster = MergeCluster(
            primary_id=a,
            members=[
                MergeClusterMember(
                    memory_id=a, title=None, body="x", salience=0.5, created_at=NOW,
                ),
                MergeClusterMember(
                    memory_id=b, title=None, body="y", salience=0.5, created_at=NOW,
                ),
            ],
            cosine_scores=[1.0, 0.95],
        )
        summary = MergeSummary(
            suggested_merged_title="t",
            suggested_merged_body="b",
            summarizer_kind=SummarizerKind.template,
        )

        class _FakeResult:
            def scalar(self) -> Any:
                return None  # ON CONFLICT DO NOTHING + RETURNING ⇒ None

        class _FakeSession:
            async def execute(self, _stmt: Any) -> _FakeResult:
                return _FakeResult()

        class _FakeCM:
            async def __aenter__(self) -> _FakeSession:
                return _FakeSession()

            async def __aexit__(self, *_args: Any) -> None:
                return None

        with patch(
            "memory_mcp.dream.passes.dedupe.session_scope",
            return_value=_FakeCM(),
        ):
            result = await _insert_proposal(
                env_id=uuid4(),
                dream_run_id=None,
                cluster=cluster,
                sorted_member_ids=[a, b],
                summary=summary,
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_on_conflict_returning_id_returns_true(
        self,
    ) -> None:
        """Inverse case: a successful insert returns the new id and
        the helper returns ``True``."""
        from unittest.mock import patch

        from memory_mcp.dream.passes.dedupe import _insert_proposal
        from memory_mcp.dream.summarizer import (
            MergeCluster,
            MergeSummary,
            SummarizerKind,
        )

        a = UUID("00000000-0000-0000-0000-00000000000a")
        b = UUID("00000000-0000-0000-0000-00000000000b")
        cluster = MergeCluster(
            primary_id=a,
            members=[
                MergeClusterMember(
                    memory_id=a, title=None, body="x", salience=0.5, created_at=NOW,
                ),
                MergeClusterMember(
                    memory_id=b, title=None, body="y", salience=0.5, created_at=NOW,
                ),
            ],
            cosine_scores=[1.0, 0.95],
        )
        summary = MergeSummary(
            suggested_merged_title="t",
            suggested_merged_body="b",
            summarizer_kind=SummarizerKind.template,
        )

        new_id = uuid4()

        class _FakeResult:
            def scalar(self) -> Any:
                return new_id

        class _FakeSession:
            async def execute(self, _stmt: Any) -> _FakeResult:
                return _FakeResult()

        class _FakeCM:
            async def __aenter__(self) -> _FakeSession:
                return _FakeSession()

            async def __aexit__(self, *_args: Any) -> None:
                return None

        with patch(
            "memory_mcp.dream.passes.dedupe.session_scope",
            return_value=_FakeCM(),
        ):
            result = await _insert_proposal(
                env_id=uuid4(),
                dream_run_id=None,
                cluster=cluster,
                sorted_member_ids=[a, b],
                summary=summary,
            )

        assert result is True


# ---------------------------------------------------------------------------
# Per-run cap
# ---------------------------------------------------------------------------


class TestBatchCap:
    @pytest.mark.asyncio
    async def test_cap_stops_emission_and_flags_items_capped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        # 5 distinct seeds, each with its own near-duplicate neighbor.
        seeds = [_FakeSeed() for _ in range(5)]
        _patch_loaders(monkeypatch, seeds=seeds)
        insert_mock = _patch_insert(monkeypatch)

        qdrant = MagicMock()

        async def search_side_effect(**kwargs: Any) -> list[dict[str, Any]]:
            return [_make_qdrant_hit(memory_id=uuid4(), score=0.95)]
        qdrant.search = AsyncMock(side_effect=search_side_effect)

        # Cap=2 ⇒ stop after 2 proposals.
        result = await run_dedupe(
            env_id,
            qdrant=qdrant,
            embedder=_make_embedder(),
            summarizer=TemplateSummarizer(),
            settings=_settings(dream_dedupe_batch_cap=2),
            now=NOW,
        )

        assert result.proposals_emitted == 2
        assert result.items_capped is True
        assert insert_mock.await_count == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBridgeAndOverlap:
    """Lock in the non-transitive clustering decision.

    A↔B above threshold and B↔C above threshold but A↔C below
    threshold MUST NOT merge into ``{A,B,C}`` — that would be a
    false-positive mega-merge. The pass instead emits two separate
    proposals (one per seed). This test makes the decision explicit
    so a future "let's add union-find" change has to confront it
    intentionally.
    """

    @pytest.mark.asyncio
    async def test_bridge_does_not_transitively_merge(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        a = UUID("aaaaaaaa-0000-0000-0000-000000000001")
        b = UUID("bbbbbbbb-0000-0000-0000-000000000002")
        c = UUID("cccccccc-0000-0000-0000-000000000003")
        seed_a = _FakeSeed(memory_id=a)
        seed_b = _FakeSeed(memory_id=b)
        # Note: we don't include c as a seed — bridge case is about
        # how a NON-seed neighbor is excluded from another seed's cluster.
        _patch_loaders(monkeypatch, seeds=[seed_a, seed_b])
        insert_mock = _patch_insert(monkeypatch)

        qdrant = MagicMock()
        # Per-seed neighbor lists model the bridge:
        # A sees only B above threshold (C is below).
        # B sees both A and C above threshold.
        async def search_side_effect(**_kwargs: Any) -> list[dict[str, Any]]:
            # The fake embedder always returns [0.1, 0.2, 0.3] so we
            # can't dispatch on vector — use call order instead. First
            # call is from seed A; second from seed B.
            if not search_side_effect.calls:
                search_side_effect.calls.append("a")
                return [_make_qdrant_hit(memory_id=b, score=0.95)]
            return [
                _make_qdrant_hit(memory_id=a, score=0.95),
                _make_qdrant_hit(memory_id=c, score=0.94),
            ]
        search_side_effect.calls = []  # type: ignore[attr-defined]
        qdrant.search = AsyncMock(side_effect=search_side_effect)

        result = await run_dedupe(
            env_id,
            qdrant=qdrant,
            embedder=_make_embedder(),
            summarizer=TemplateSummarizer(),
            settings=_settings(),
            now=NOW,
        )

        # A → cluster {A, B}; B → cluster {A, B, C}. Different
        # member sets ⇒ different dedupe_keys ⇒ TWO proposals.
        assert result.proposals_emitted == 2
        assert insert_mock.await_count == 2
        # Verify the two distinct member sets.
        cluster_sizes = sorted(
            len(call.kwargs["sorted_member_ids"])
            for call in insert_mock.await_args_list
        )
        assert cluster_sizes == [2, 3]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_env_clean_result(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        _patch_loaders(monkeypatch, seeds=[])
        insert_mock = _patch_insert(monkeypatch)

        qdrant = MagicMock()
        qdrant.search = AsyncMock(return_value=[])

        result = await run_dedupe(
            env_id,
            qdrant=qdrant,
            embedder=_make_embedder(),
            summarizer=TemplateSummarizer(),
            settings=_settings(),
            now=NOW,
        )

        assert result.seeds_examined == 0
        assert result.proposals_emitted == 0
        assert result.proposals_skipped_existing == 0
        assert result.proposals_skipped_below_min_size == 0
        assert result.items_capped is False
        insert_mock.assert_not_called()
        qdrant.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_seed_text_skipped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A memory whose embed input would be empty should be skipped
        (no search, no cluster, no proposal)."""
        env_id = uuid4()
        empty_seed = _FakeSeed(title=None, body="")
        _patch_loaders(monkeypatch, seeds=[empty_seed])
        _patch_insert(monkeypatch)

        qdrant = MagicMock()
        qdrant.search = AsyncMock(return_value=[])

        result = await run_dedupe(
            env_id,
            qdrant=qdrant,
            embedder=_make_embedder(),
            summarizer=TemplateSummarizer(),
            settings=_settings(),
            now=NOW,
        )

        assert result.proposals_emitted == 0
        assert result.proposals_skipped_below_min_size == 1
        qdrant.search.assert_not_called()


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


class TestDedupeSettingsValidation:
    def test_window_days_must_be_positive(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            Settings(dream_dedupe_window_days=0)  # type: ignore[arg-type]

    def test_threshold_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            Settings(dream_dedupe_threshold=1.5)  # type: ignore[arg-type]

    def test_top_k_must_be_at_least_2(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            Settings(dream_dedupe_top_k=1)  # type: ignore[arg-type]

    def test_batch_cap_must_be_positive(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            Settings(dream_dedupe_batch_cap=0)  # type: ignore[arg-type]
