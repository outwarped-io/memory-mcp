from __future__ import annotations

import logging
from uuid import UUID, uuid4

import pytest

from dream_worker import decision_conflicts as dc
from memory_mcp.identity import AgentContext


ENV_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
ENV_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
DECISION_A = UUID("00000000-0000-0000-0000-00000000000a")
DECISION_B = UUID("00000000-0000-0000-0000-00000000000b")


class _FakeQdrant:
    def __init__(self, vectors: dict[UUID, list[float] | None]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[UUID, list[UUID], str]] = []

    async def get_vectors(
        self,
        *,
        env_id: UUID,
        ids: list[UUID],
        vector_name: str = "body",
    ) -> dict[UUID, list[float] | None]:
        self.calls.append((env_id, ids, vector_name))
        return {memory_id: self.vectors.get(memory_id) for memory_id in ids}


def _ctx(env_id: UUID) -> AgentContext:
    return AgentContext(
        agent_id=uuid4(),
        agent_name="test",
        attached_env_ids=[env_id],
        is_default_agent=True,
    )


async def _patch_rows_and_collect(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dc.DecisionRow],
    *,
    existing_keys: set[str] | None = None,
) -> list[dict[str, object]]:
    proposals: list[dict[str, object]] = []
    keys = existing_keys if existing_keys is not None else set()

    async def _load(*, env_id: UUID) -> list[dc.DecisionRow]:
        return rows

    async def _insert(
        *,
        env_id: UUID,
        dream_run_id: UUID | None,
        a_id: UUID,
        b_id: UUID,
        cosine: float,
    ) -> bool:
        key = dc._decision_conflict_dedupe_key(a_id, b_id)  # noqa: SLF001
        if key in keys:
            return False
        keys.add(key)
        a_id, b_id = sorted([a_id, b_id], key=str)
        proposals.append({
            "env_id": env_id,
            "dream_run_id": dream_run_id,
            "dedupe_key": key,
            "payload": {
                "decision_a": str(a_id),
                "decision_b": str(b_id),
                "cosine": float(cosine),
            },
        })
        return True

    monkeypatch.setattr(dc, "_load_accepted_decisions", _load)
    monkeypatch.setattr(dc, "_insert_decision_conflict_proposal", _insert)
    return proposals


@pytest.mark.asyncio
async def test_no_accepted_decisions_emits_no_proposals(monkeypatch: pytest.MonkeyPatch) -> None:
    proposals = await _patch_rows_and_collect(monkeypatch, [])

    result = await dc.run_decision_conflict_pass(
        ENV_A,
        actor_ctx=_ctx(ENV_A),
        qdrant=_FakeQdrant({}),
        threshold=0.85,
    )

    assert result.decisions_examined == 0
    assert result.proposals_emitted == 0
    assert proposals == []


@pytest.mark.asyncio
async def test_single_accepted_decision_emits_no_proposals(monkeypatch: pytest.MonkeyPatch) -> None:
    proposals = await _patch_rows_and_collect(monkeypatch, [dc.DecisionRow(DECISION_A, "a")])

    result = await dc.run_decision_conflict_pass(
        ENV_A,
        actor_ctx=_ctx(ENV_A),
        qdrant=_FakeQdrant({DECISION_A: [1.0, 0.0]}),
        threshold=0.85,
    )

    assert result.decisions_examined == 1
    assert result.proposals_emitted == 0
    assert proposals == []


@pytest.mark.asyncio
async def test_two_decisions_below_threshold_emit_no_proposals(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [dc.DecisionRow(DECISION_A, "a"), dc.DecisionRow(DECISION_B, "b")]
    proposals = await _patch_rows_and_collect(monkeypatch, rows)

    result = await dc.run_decision_conflict_pass(
        ENV_A,
        actor_ctx=_ctx(ENV_A),
        qdrant=_FakeQdrant({DECISION_A: [1.0, 0.0], DECISION_B: [0.0, 1.0]}),
        threshold=0.85,
    )

    assert result.pairs_examined == 1
    assert result.proposals_emitted == 0
    assert proposals == []


@pytest.mark.asyncio
async def test_two_decisions_above_threshold_emit_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [dc.DecisionRow(DECISION_A, "a"), dc.DecisionRow(DECISION_B, "b")]
    proposals = await _patch_rows_and_collect(monkeypatch, rows)

    result = await dc.run_decision_conflict_pass(
        ENV_A,
        actor_ctx=_ctx(ENV_A),
        qdrant=_FakeQdrant({DECISION_A: [1.0, 0.0], DECISION_B: [0.9, 0.1]}),
        threshold=0.85,
    )

    assert result.proposals_emitted == 1
    assert proposals[0]["dedupe_key"] == f"decision-conflict:{DECISION_A}:{DECISION_B}"
    assert proposals[0]["payload"]["decision_a"] == str(DECISION_A)  # type: ignore[index]
    assert proposals[0]["payload"]["decision_b"] == str(DECISION_B)  # type: ignore[index]
    assert proposals[0]["payload"]["cosine"] >= 0.85  # type: ignore[index]


@pytest.mark.asyncio
async def test_cross_env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    env_rows = {
        ENV_A: [dc.DecisionRow(DECISION_A, "a"), dc.DecisionRow(DECISION_B, "b")],
        ENV_B: [],
    }
    proposals: list[dict[str, object]] = []

    async def _load(*, env_id: UUID) -> list[dc.DecisionRow]:
        return env_rows[env_id]

    async def _insert(**kwargs) -> bool:  # type: ignore[no-untyped-def]
        proposals.append(kwargs)
        return True

    monkeypatch.setattr(dc, "_load_accepted_decisions", _load)
    monkeypatch.setattr(dc, "_insert_decision_conflict_proposal", _insert)
    qdrant = _FakeQdrant({DECISION_A: [1.0, 0.0], DECISION_B: [0.9, 0.1]})

    result_a = await dc.run_decision_conflict_pass(ENV_A, actor_ctx=_ctx(ENV_A), qdrant=qdrant, threshold=0.85)
    result_b = await dc.run_decision_conflict_pass(ENV_B, actor_ctx=_ctx(ENV_B), qdrant=qdrant, threshold=0.85)

    assert result_a.proposals_emitted == 1
    assert result_b.proposals_emitted == 0
    assert [p["env_id"] for p in proposals] == [ENV_A]


@pytest.mark.asyncio
async def test_missing_vector_is_logged_and_skipped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [dc.DecisionRow(DECISION_A, "a"), dc.DecisionRow(DECISION_B, "b")]
    proposals = await _patch_rows_and_collect(monkeypatch, rows)
    caplog.set_level(logging.DEBUG, logger=dc.__name__)

    result = await dc.run_decision_conflict_pass(
        ENV_A,
        actor_ctx=_ctx(ENV_A),
        qdrant=_FakeQdrant({DECISION_A: [1.0, 0.0], DECISION_B: None}),
        threshold=0.85,
    )

    assert result.missing_vectors == 1
    assert result.proposals_emitted == 0
    assert proposals == []
    assert "skipped missing body vector" in caplog.text


@pytest.mark.asyncio
async def test_duplicate_suppression_keeps_one_open_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [dc.DecisionRow(DECISION_A, "a"), dc.DecisionRow(DECISION_B, "b")]
    existing_keys: set[str] = set()
    proposals = await _patch_rows_and_collect(monkeypatch, rows, existing_keys=existing_keys)
    qdrant = _FakeQdrant({DECISION_A: [1.0, 0.0], DECISION_B: [0.9, 0.1]})

    first = await dc.run_decision_conflict_pass(ENV_A, actor_ctx=_ctx(ENV_A), qdrant=qdrant, threshold=0.85)
    second = await dc.run_decision_conflict_pass(ENV_A, actor_ctx=_ctx(ENV_A), qdrant=qdrant, threshold=0.85)

    assert first.proposals_emitted == 1
    assert second.proposals_emitted == 0
    assert second.proposals_skipped_existing == 1
    assert len(proposals) == 1


@pytest.mark.asyncio
async def test_pair_canonicalization_suppresses_reverse_order_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    existing_keys: set[str] = set()
    rows_by_call = [
        [dc.DecisionRow(DECISION_B, "b"), dc.DecisionRow(DECISION_A, "a")],
        [dc.DecisionRow(DECISION_A, "a"), dc.DecisionRow(DECISION_B, "b")],
    ]
    proposals: list[dict[str, object]] = []

    async def _load(*, env_id: UUID) -> list[dc.DecisionRow]:
        return rows_by_call.pop(0)

    async def _insert(
        *,
        env_id: UUID,
        dream_run_id: UUID | None,
        a_id: UUID,
        b_id: UUID,
        cosine: float,
    ) -> bool:
        key = dc._decision_conflict_dedupe_key(a_id, b_id)  # noqa: SLF001
        if key in existing_keys:
            return False
        existing_keys.add(key)
        proposals.append({"dedupe_key": key, "env_id": env_id, "cosine": cosine})
        return True

    monkeypatch.setattr(dc, "_load_accepted_decisions", _load)
    monkeypatch.setattr(dc, "_insert_decision_conflict_proposal", _insert)
    qdrant = _FakeQdrant({DECISION_A: [1.0, 0.0], DECISION_B: [0.9, 0.1]})

    first = await dc.run_decision_conflict_pass(ENV_A, actor_ctx=_ctx(ENV_A), qdrant=qdrant, threshold=0.85)
    second = await dc.run_decision_conflict_pass(ENV_A, actor_ctx=_ctx(ENV_A), qdrant=qdrant, threshold=0.85)

    assert first.proposals_emitted == 1
    assert second.proposals_skipped_existing == 1
    assert len(proposals) == 1
    assert proposals[0]["dedupe_key"] == f"decision-conflict:{DECISION_A}:{DECISION_B}"
