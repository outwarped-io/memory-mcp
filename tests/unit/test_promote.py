"""Unit tests for the promote pass.

Test patterns mirror :mod:`tests.unit.test_dedupe` exactly:

* Loaders, ``_open_proposal_exists`` and ``_insert_proposal`` are
  monkey-patched at module scope as ``AsyncMock``s — none of the tests
  hit a real database.
* ``TemplateSummarizer`` is the default for hermetic tests; an
  ``AsyncMock`` summarizer is used when call-count assertions matter.
* Counters in :class:`PromotePassResult` are asserted directly so the
  observability invariant ``entity_clusters_found == proposals_emitted
  + proposals_skipped_existing + proposals_skipped_capped`` is locked in.

The tests cover both rubber-duck blockers from the p2.2-promote design
critique:

* **Blocker #1** — ``_open_proposal_exists`` checks across **all**
  statuses (not just open) so accepted/rejected proposals over the same
  evidence don't re-emit. ``test_pre_summarize_skip_uses_cross_status_check``
  + ``test_existing_proposal_skipped_no_summarizer_call``.
* **Blocker #2** — relation multiplicity at the loader level cannot
  inflate cluster size. ``test_relation_multiplicity_does_not_inflate_cluster``
  exercises Python-level set semantics (the SQL ``DISTINCT`` is exercised
  in invariants/integration tests).
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from memory_mcp.config import Settings
from memory_mcp.dream.passes.promote import (
    PromotePassResult,
    _build_dedupe_key,
    _EntityRefRow,
    _ObservationRow,
    run_promote,
)
from memory_mcp.dream.summarizer import (
    PromotionCluster,
    PromotionSummary,
    SummarizerKind,
    TemplateSummarizer,
)

NOW = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _settings(
    *,
    window_days: int = 14,
    min_cluster: int = 3,
    cap: int = 100,
    obs_per_cluster: int = 20,
) -> Settings:
    """Build Settings with promote knobs overridden.

    Other knobs use Settings defaults — this is a partial override that
    exercises the validators on the dream_promote_* fields too.
    """
    return Settings(
        dream_promote_window_days=window_days,
        dream_promote_min_cluster_size=min_cluster,
        dream_promote_batch_cap=cap,
        dream_promote_observations_per_cluster=obs_per_cluster,
    )


def _make_obs(
    *,
    memory_id: UUID | None = None,
    body: str = "an observation",
    created_at: dt.datetime | None = None,
) -> _ObservationRow:
    return _ObservationRow(
        id=memory_id or uuid4(),
        body=body,
        created_at=created_at or NOW,
    )


def _patch_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    observations: list[_ObservationRow],
    refs: list[_EntityRefRow],
) -> None:
    monkeypatch.setattr(
        "memory_mcp.dream.passes.promote._load_observation_rows",
        AsyncMock(return_value=observations),
    )
    monkeypatch.setattr(
        "memory_mcp.dream.passes.promote._load_observation_entity_refs",
        AsyncMock(return_value=refs),
    )


def _patch_insert(
    monkeypatch: pytest.MonkeyPatch,
    *,
    side_effect: list[Any] | None = None,
    exists: bool = False,
) -> AsyncMock:
    """Patch ``_insert_proposal`` (default returns ``True``) AND
    ``_open_proposal_exists`` (default ``False`` — no prior proposal).

    Pass ``exists=True`` to test the pre-summarize skip path.
    """
    insert_mock = (
        AsyncMock(return_value=True)
        if side_effect is None
        else AsyncMock(side_effect=side_effect)
    )
    monkeypatch.setattr(
        "memory_mcp.dream.passes.promote._insert_proposal", insert_mock,
    )
    monkeypatch.setattr(
        "memory_mcp.dream.passes.promote._open_proposal_exists",
        AsyncMock(return_value=exists),
    )
    return insert_mock


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_dedupe_key_is_stable_across_orderings(self) -> None:
        eid = UUID("12345678-1234-1234-1234-123456789abc")
        a = UUID("aaaaaaaa-0000-0000-0000-000000000001")
        b = UUID("bbbbbbbb-0000-0000-0000-000000000002")
        c = UUID("cccccccc-0000-0000-0000-000000000003")

        k1 = _build_dedupe_key(entity_id=eid, evidence_observation_ids=[a, b, c])
        k2 = _build_dedupe_key(entity_id=eid, evidence_observation_ids=[c, a, b])
        k3 = _build_dedupe_key(entity_id=eid, evidence_observation_ids=[b, c, a])

        assert k1 == k2 == k3
        assert k1.startswith(f"promote:entity={eid}:evidence=")

    def test_different_entity_yields_different_key(self) -> None:
        e1 = UUID("11111111-0000-0000-0000-000000000001")
        e2 = UUID("22222222-0000-0000-0000-000000000002")
        a = UUID("aaaaaaaa-0000-0000-0000-000000000001")

        k1 = _build_dedupe_key(entity_id=e1, evidence_observation_ids=[a])
        k2 = _build_dedupe_key(entity_id=e2, evidence_observation_ids=[a])

        assert k1 != k2

    def test_different_evidence_yields_different_key(self) -> None:
        eid = UUID("12345678-1234-1234-1234-123456789abc")
        a = UUID("aaaaaaaa-0000-0000-0000-000000000001")
        b = UUID("bbbbbbbb-0000-0000-0000-000000000002")

        k1 = _build_dedupe_key(entity_id=eid, evidence_observation_ids=[a])
        k2 = _build_dedupe_key(entity_id=eid, evidence_observation_ids=[a, b])

        assert k1 != k2


# ---------------------------------------------------------------------------
# Cluster formation + emission
# ---------------------------------------------------------------------------


class TestClusterFormation:
    @pytest.mark.asyncio
    async def test_three_observations_one_entity_emits_one_proposal(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        eid = uuid4()
        o1, o2, o3 = uuid4(), uuid4(), uuid4()
        obs = [
            _make_obs(memory_id=o1, body="obs 1", created_at=NOW),
            _make_obs(memory_id=o2, body="obs 2", created_at=NOW),
            _make_obs(memory_id=o3, body="obs 3", created_at=NOW),
        ]
        refs = [
            _EntityRefRow(memory_id=o1, entity_id=eid, entity_name="Foo"),
            _EntityRefRow(memory_id=o2, entity_id=eid, entity_name="Foo"),
            _EntityRefRow(memory_id=o3, entity_id=eid, entity_name="Foo"),
        ]
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch)

        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        assert result.entity_clusters_found == 1
        assert result.proposals_emitted == 1
        assert result.proposals_skipped_existing == 0
        assert result.proposals_skipped_capped == 0
        assert result.observations_examined == 3
        assert insert_mock.await_count == 1

        # Verify payload shape: split into all/evidence + chronological.
        call_kwargs = insert_mock.await_args_list[0].kwargs
        payload = call_kwargs["payload"]
        assert payload["observation_count"] == 3
        assert payload["evidence_observation_count"] == 3
        assert set(payload["all_observation_ids"]) == {str(o1), str(o2), str(o3)}
        assert payload["target_kind"] == "fact"
        assert payload["source_entity_id"] == str(eid)

    @pytest.mark.asyncio
    async def test_below_min_cluster_size_skipped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        eid = uuid4()
        o1, o2 = uuid4(), uuid4()
        obs = [_make_obs(memory_id=o1), _make_obs(memory_id=o2)]
        refs = [
            _EntityRefRow(memory_id=o1, entity_id=eid, entity_name="Foo"),
            _EntityRefRow(memory_id=o2, entity_id=eid, entity_name="Foo"),
        ]
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch)

        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        assert result.entity_clusters_found == 0
        assert result.proposals_emitted == 0
        insert_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_observation_referencing_two_entities_contributes_to_both(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-transitive clustering: an observation referencing entities
        A and B contributes to the A-cluster and the B-cluster
        independently. If both pass min_cluster_size, two proposals
        emit.
        """
        env_id = uuid4()
        ea, eb = uuid4(), uuid4()
        o1, o2, o3 = uuid4(), uuid4(), uuid4()
        # All 3 observations reference both entities ⇒ both clusters
        # qualify under min_cluster_size=3.
        obs = [_make_obs(memory_id=mid) for mid in (o1, o2, o3)]
        refs = []
        for mid in (o1, o2, o3):
            refs.append(_EntityRefRow(memory_id=mid, entity_id=ea, entity_name="A"))
            refs.append(_EntityRefRow(memory_id=mid, entity_id=eb, entity_name="B"))
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch)

        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        assert result.entity_clusters_found == 2
        assert result.proposals_emitted == 2
        assert insert_mock.await_count == 2

        # Both proposals should have distinct dedupe keys.
        keys = {call.kwargs["dedupe_key"] for call in insert_mock.await_args_list}
        assert len(keys) == 2

    @pytest.mark.asyncio
    async def test_relation_multiplicity_does_not_inflate_cluster(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rubber-duck blocker #2 from p2.2-promote critique.

        If one observation has multiple ``_EntityRefRow`` entries against
        the same entity (which can happen if the SQL DISTINCT layer is
        ever weakened), Python-level set semantics MUST collapse them so
        a single observation cannot satisfy ``min_cluster_size=3``.
        """
        env_id = uuid4()
        eid = uuid4()
        o1 = uuid4()
        obs = [_make_obs(memory_id=o1)]
        # 3 ref rows for ONE observation against ONE entity.
        refs = [
            _EntityRefRow(memory_id=o1, entity_id=eid, entity_name="Foo"),
            _EntityRefRow(memory_id=o1, entity_id=eid, entity_name="Foo"),
            _EntityRefRow(memory_id=o1, entity_id=eid, entity_name="Foo"),
        ]
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch)

        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        # ONE observation cannot form a 3-observation cluster regardless
        # of how many ref rows duplicate it.
        assert result.entity_clusters_found == 0
        assert result.proposals_emitted == 0
        insert_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_refs_for_unloaded_observations_are_dropped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Defensive: a ref pointing at a memory_id we never loaded
        (e.g., an observation outside the recent window that the JOIN
        surfaced anyway) must NOT inflate the cluster.
        """
        env_id = uuid4()
        eid = uuid4()
        o1, o2, o3 = uuid4(), uuid4(), uuid4()
        ghost = uuid4()  # never appears in observations
        obs = [
            _make_obs(memory_id=o1),
            _make_obs(memory_id=o2),
        ]
        refs = [
            _EntityRefRow(memory_id=o1, entity_id=eid, entity_name="Foo"),
            _EntityRefRow(memory_id=o2, entity_id=eid, entity_name="Foo"),
            _EntityRefRow(memory_id=o3, entity_id=eid, entity_name="Foo"),
            _EntityRefRow(memory_id=ghost, entity_id=eid, entity_name="Foo"),
        ]
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch)

        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        # Only o1 and o2 are loaded; ghost and o3 are dropped. Cluster
        # has 2 members, below threshold.
        assert result.entity_clusters_found == 0
        insert_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_pre_summarize_skip_uses_cross_status_check(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rubber-duck blocker #1 from p2.2-promote critique.

        The pre-summarize EXISTS check checks across **any** proposal
        status — not just ``open`` — so an already-accepted/rejected
        proposal over the same evidence DOES NOT re-emit.

        We don't directly test the SQL here (that's covered in
        invariants); we test that the pass calls
        ``_open_proposal_exists`` BEFORE the summarizer.
        """
        env_id = uuid4()
        eid = uuid4()
        obs = [_make_obs() for _ in range(3)]
        refs = [
            _EntityRefRow(memory_id=o.id, entity_id=eid, entity_name="Foo")
            for o in obs
        ]
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch, exists=True)

        # Spy summarizer: ensure summarize_promotion is NOT called.
        summarizer = TemplateSummarizer()
        summarize_spy = AsyncMock(side_effect=summarizer.summarize_promotion)
        summarizer.summarize_promotion = summarize_spy  # type: ignore[method-assign]

        result = await run_promote(
            env_id,
            summarizer=summarizer,
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        assert result.proposals_emitted == 0
        assert result.proposals_skipped_existing == 1
        # CRITICAL: summarizer NOT called when pre-existing proposal found.
        summarize_spy.assert_not_called()
        insert_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_on_conflict_returns_false_counts_as_skipped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``_insert_proposal`` returns ``False`` (interleaved-worker
        race: another worker beat us to the same open proposal), the
        cluster counts toward ``proposals_skipped_existing``.
        """
        env_id = uuid4()
        eid = uuid4()
        obs = [_make_obs() for _ in range(3)]
        refs = [
            _EntityRefRow(memory_id=o.id, entity_id=eid, entity_name="Foo")
            for o in obs
        ]
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch, side_effect=[False])

        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        assert result.proposals_emitted == 0
        assert result.proposals_skipped_existing == 1
        assert insert_mock.await_count == 1


# ---------------------------------------------------------------------------
# Per-run cap
# ---------------------------------------------------------------------------


class TestBatchCap:
    @pytest.mark.asyncio
    async def test_cap_stops_emission_and_records_skipped_capped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the cap is hit, remaining eligible clusters surface as
        ``proposals_skipped_capped``. The accounting invariant
        ``entity_clusters_found == proposals_emitted +
        proposals_skipped_existing + proposals_skipped_capped`` MUST
        hold (rubber-duck finding #5).
        """
        env_id = uuid4()
        # 5 entities, 3 observations each, all in one big load.
        entities = [uuid4() for _ in range(5)]
        # Each entity gets its own 3 observations (15 total).
        obs_per_entity: dict[UUID, list[UUID]] = {}
        all_obs: list[_ObservationRow] = []
        refs: list[_EntityRefRow] = []
        for i, eid in enumerate(entities):
            ids_for_entity = [uuid4() for _ in range(3)]
            obs_per_entity[eid] = ids_for_entity
            for oid in ids_for_entity:
                all_obs.append(_make_obs(memory_id=oid, body=f"e{i}"))
                refs.append(_EntityRefRow(
                    memory_id=oid, entity_id=eid, entity_name=f"E{i}",
                ))
        _patch_loaders(monkeypatch, observations=all_obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch)

        # Cap = 2. 5 eligible clusters. Expect 2 emitted, 3 capped.
        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3, cap=2),
            now=NOW,
        )

        assert result.entity_clusters_found == 5
        assert result.proposals_emitted == 2
        assert result.proposals_skipped_existing == 0
        assert result.proposals_skipped_capped == 3
        assert result.items_capped is True
        # Accounting invariant.
        assert (
            result.entity_clusters_found
            == result.proposals_emitted
            + result.proposals_skipped_existing
            + result.proposals_skipped_capped
        )
        assert insert_mock.await_count == 2


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    @pytest.mark.asyncio
    async def test_evidence_truncated_full_set_preserved_in_payload(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``observations_per_cluster`` is smaller than the cluster:

        * the summarizer sees only the truncated (most-recent-N) set;
        * the payload preserves the full set as ``all_observation_ids``;
        * the dedupe key derives from the truncated evidence set.
        """
        env_id = uuid4()
        eid = uuid4()
        # 5 observations spanning 5 days, oldest first.
        obs_ids = [uuid4() for _ in range(5)]
        obs = []
        for i, oid in enumerate(obs_ids):
            obs.append(_make_obs(
                memory_id=oid,
                body=f"obs {i}",
                created_at=NOW - dt.timedelta(days=4 - i),  # newer last
            ))
        refs = [
            _EntityRefRow(memory_id=oid, entity_id=eid, entity_name="Foo")
            for oid in obs_ids
        ]
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch)

        # Truncate to 3.
        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3, obs_per_cluster=3),
            now=NOW,
        )

        assert result.proposals_emitted == 1
        call = insert_mock.await_args_list[0].kwargs
        payload = call["payload"]

        assert payload["observation_count"] == 5
        assert payload["evidence_observation_count"] == 3
        assert set(payload["all_observation_ids"]) == {str(o) for o in obs_ids}
        # Evidence is the most-recent-3.
        evidence_set = set(payload["evidence_observation_ids"])
        assert evidence_set == {str(oid) for oid in obs_ids[2:]}
        # Dedupe key uses the truncated evidence (sorted).
        assert all(
            oid in call["dedupe_key"]
            for oid in (str(o) for o in obs_ids[2:])
        )
        # Older two observations are NOT in the dedupe key.
        for old_oid in obs_ids[:2]:
            assert str(old_oid) not in call["dedupe_key"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_env_clean_result(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        _patch_loaders(monkeypatch, observations=[], refs=[])
        insert_mock = _patch_insert(monkeypatch)

        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(),
            now=NOW,
        )

        assert result.observations_examined == 0
        assert result.entity_clusters_found == 0
        assert result.proposals_emitted == 0
        assert result.summarizer_kind == SummarizerKind.template.value
        insert_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_observations_with_no_entity_refs_are_ignored(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        # 5 observations, zero refs.
        obs = [_make_obs() for _ in range(5)]
        _patch_loaders(monkeypatch, observations=obs, refs=[])
        insert_mock = _patch_insert(monkeypatch)

        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        assert result.observations_examined == 5
        assert result.entity_clusters_found == 0
        insert_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarizer_kind_recorded_on_result_and_payload(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_id = uuid4()
        eid = uuid4()
        obs = [_make_obs() for _ in range(3)]
        refs = [
            _EntityRefRow(memory_id=o.id, entity_id=eid, entity_name="Foo")
            for o in obs
        ]
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch)

        # Use a mock summarizer with a controlled return.
        class FakeSummarizer:
            kind = SummarizerKind.template

            async def summarize_promotion(
                self, cluster: PromotionCluster,
            ) -> PromotionSummary:
                return PromotionSummary(
                    suggested_title=f"About {cluster.source_entity_name}",
                    suggested_body="some body",
                    suggested_confidence=0.55,
                    summarizer_kind=SummarizerKind.template,
                    llm_failed=False,
                    llm_model_id=None,
                )

            async def summarize_merge(
                self, cluster: Any,
            ) -> Any:
                raise NotImplementedError

        result = await run_promote(
            env_id,
            summarizer=FakeSummarizer(),
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        assert result.summarizer_kind == SummarizerKind.template.value
        assert result.proposals_emitted == 1

        payload = insert_mock.await_args_list[0].kwargs["payload"]
        assert payload["summarizer_kind"] == "template"
        assert payload["suggested_title"] == "About Foo"
        assert payload["suggested_body"] == "some body"
        assert payload["suggested_confidence"] == 0.55
        assert payload["llm_failed"] is False

    @pytest.mark.asyncio
    async def test_existing_proposal_skipped_no_summarizer_call(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``test_pre_summarize_skip_uses_cross_status_check`` covers the
        same invariant with a real ``TemplateSummarizer``; this version
        uses a strict ``MagicMock`` summarizer so the assertion is
        airtight even if the template impl ever becomes a no-op.
        """
        env_id = uuid4()
        eid = uuid4()
        obs = [_make_obs() for _ in range(3)]
        refs = [
            _EntityRefRow(memory_id=o.id, entity_id=eid, entity_name="Foo")
            for o in obs
        ]
        _patch_loaders(monkeypatch, observations=obs, refs=refs)
        insert_mock = _patch_insert(monkeypatch, exists=True)

        summarizer = MagicMock()
        summarizer.kind = SummarizerKind.template
        summarizer.summarize_promotion = AsyncMock()

        result = await run_promote(
            env_id,
            summarizer=summarizer,
            settings=_settings(min_cluster=3),
            now=NOW,
        )

        summarizer.summarize_promotion.assert_not_called()
        insert_mock.assert_not_called()
        assert result.proposals_skipped_existing == 1


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


class TestPromoteSettingsValidation:
    def test_min_cluster_size_rejects_below_2(self) -> None:
        with pytest.raises(ValueError):
            Settings(dream_promote_min_cluster_size=1)

    def test_window_days_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            Settings(dream_promote_window_days=0)

    def test_batch_cap_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            Settings(dream_promote_batch_cap=0)

    def test_observations_per_cluster_rejects_below_2(self) -> None:
        with pytest.raises(ValueError):
            Settings(dream_promote_observations_per_cluster=1)


# ---------------------------------------------------------------------------
# Result invariant
# ---------------------------------------------------------------------------


class TestResultInvariant:
    """The accounting invariant must hold across all paths."""

    @pytest.mark.asyncio
    async def test_invariant_holds_with_mix_of_outcomes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """3 eligible clusters; 1 emits, 1 has prior proposal, 1 races
        — but cap=10 so no capped. Still: emitted + skipped_existing ==
        clusters_found.
        """
        env_id = uuid4()
        # 3 distinct entities, each with 3 observations.
        entities = [uuid4() for _ in range(3)]
        obs: list[_ObservationRow] = []
        refs: list[_EntityRefRow] = []
        for i, eid in enumerate(entities):
            for _ in range(3):
                oid = uuid4()
                obs.append(_make_obs(memory_id=oid))
                refs.append(_EntityRefRow(
                    memory_id=oid, entity_id=eid, entity_name=f"E{i}",
                ))
        _patch_loaders(monkeypatch, observations=obs, refs=refs)

        # First insert succeeds; second is a race-loss; third succeeds.
        # Set up: insert returns [True, False, True] but ALSO need to
        # vary `_open_proposal_exists`. Use side_effect here.
        insert_mock = AsyncMock(side_effect=[True, False, True])
        monkeypatch.setattr(
            "memory_mcp.dream.passes.promote._insert_proposal", insert_mock,
        )
        monkeypatch.setattr(
            "memory_mcp.dream.passes.promote._open_proposal_exists",
            AsyncMock(return_value=False),
        )

        result = await run_promote(
            env_id,
            summarizer=TemplateSummarizer(),
            settings=_settings(min_cluster=3, cap=10),
            now=NOW,
        )

        assert result.entity_clusters_found == 3
        assert result.proposals_emitted == 2
        assert result.proposals_skipped_existing == 1
        assert result.proposals_skipped_capped == 0
        # Invariant.
        assert (
            result.entity_clusters_found
            == result.proposals_emitted
            + result.proposals_skipped_existing
            + result.proposals_skipped_capped
        )


# ---------------------------------------------------------------------------
# Dataclass smoke
# ---------------------------------------------------------------------------


def test_pass_result_default_values() -> None:
    env_id = uuid4()
    r = PromotePassResult(env_id=env_id)
    assert r.env_id == env_id
    assert r.observations_examined == 0
    assert r.entity_clusters_found == 0
    assert r.proposals_emitted == 0
    assert r.proposals_skipped_existing == 0
    assert r.proposals_skipped_capped == 0
    assert r.items_capped is False
    assert r.summarizer_kind is None
    assert r.duration_seconds == 0.0
