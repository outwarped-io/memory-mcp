"""Decision-conflict dream pass.

Scans accepted decision memories in one environment, compares their stored
body vectors, and emits human-reviewable proposals for highly similar pairs.
"""

from __future__ import annotations

import logging
import math
import time
import uuid as uuidlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import and_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from memory_mcp.db.models import DreamProposal, Memory
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import (
    DecisionStatus,
    DreamProposalKind,
    DreamProposalStatus,
    MemoryKind,
)
from memory_mcp.identity import AgentContext

log = logging.getLogger(__name__)

SCAN_LIMIT = 500


class _BatchVectorStore(Protocol):
    async def get_vectors(
        self,
        *,
        env_id: UUID,
        ids: list[UUID],
        vector_name: str = "body",
    ) -> dict[UUID, list[float] | None]: ...


@dataclass(frozen=True)
class DecisionConflictPassResult:
    env_id: UUID
    decisions_examined: int = 0
    pairs_examined: int = 0
    proposals_emitted: int = 0
    proposals_skipped_existing: int = 0
    missing_vectors: int = 0
    items_capped: bool = False
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class DecisionRow:
    id: UUID
    body: str


async def _load_accepted_decisions(*, env_id: UUID) -> list[DecisionRow]:
    async with session_scope() as s:
        stmt = (
            select(Memory.id, Memory.body)
            .where(
                and_(
                    Memory.env_id == env_id,
                    Memory.kind == MemoryKind.decision.value,
                    Memory.decision_meta.is_not(None),
                    Memory.decision_meta["status"].astext == DecisionStatus.accepted.value,
                )
            )
            .order_by(Memory.updated_at.desc(), Memory.id)
            .limit(SCAN_LIMIT)
        )
        rows = (await s.execute(stmt)).all()

    if len(rows) == SCAN_LIMIT:
        log.warning("decision conflict scan truncated to 500 in env=%s", env_id)

    return [DecisionRow(id=r[0], body=r[1] or "") for r in rows]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _decision_conflict_dedupe_key(a_id: UUID, b_id: UUID) -> str:
    a_id, b_id = sorted([a_id, b_id], key=str)
    return f"decision-conflict:{a_id}:{b_id}"


async def _insert_decision_conflict_proposal(
    *,
    env_id: UUID,
    dream_run_id: UUID | None,
    a_id: UUID,
    b_id: UUID,
    cosine: float,
) -> bool:
    a_id, b_id = sorted([a_id, b_id], key=str)
    dedupe_key = _decision_conflict_dedupe_key(a_id, b_id)
    payload = {
        "decision_a": str(a_id),
        "decision_b": str(b_id),
        "cosine": float(cosine),
    }

    async with session_scope() as s:
        stmt = (
            pg_insert(DreamProposal)
            .values(
                id=uuidlib.uuid4(),
                env_id=env_id,
                kind=DreamProposalKind.decision_conflict_candidate.value,
                status=DreamProposalStatus.open.value,
                payload=payload,
                dedupe_key=dedupe_key,
                dream_run_id=dream_run_id,
            )
            .on_conflict_do_nothing(
                index_elements=["env_id", "kind", "dedupe_key"],
                index_where=text("status = 'open' AND dedupe_key IS NOT NULL"),
            )
            .returning(DreamProposal.id)
        )
        inserted_id = (await s.execute(stmt)).scalar()

    return inserted_id is not None


async def run_decision_conflict_pass(
    env_id: UUID,
    *,
    actor_ctx: AgentContext,
    qdrant: _BatchVectorStore,
    threshold: float,
    dream_run_id: UUID | None = None,
) -> DecisionConflictPassResult:
    """Run one accepted-decision conflict scan for ``env_id``."""

    del actor_ctx  # kept for parity with other dream-worker pass call signatures
    started = time.perf_counter()
    rows = await _load_accepted_decisions(env_id=env_id)
    if len(rows) < 2:
        return DecisionConflictPassResult(
            env_id=env_id,
            decisions_examined=len(rows),
            duration_seconds=time.perf_counter() - started,
        )

    vectors = await qdrant.get_vectors(
        env_id=env_id,
        ids=[r.id for r in rows],
        vector_name="body",
    )
    usable: list[tuple[DecisionRow, list[float]]] = []
    missing_vectors = 0
    for row in rows:
        vector = vectors.get(row.id)
        if vector is None:
            missing_vectors += 1
            log.debug(
                "decision conflict scan skipped missing body vector for memory %s in env=%s",
                row.id,
                env_id,
            )
            continue
        usable.append((row, vector))

    emitted_keys: set[str] = set()
    pairs_examined = 0
    proposals_emitted = 0
    proposals_skipped_existing = 0

    for i, (left, left_vector) in enumerate(usable):
        for right, right_vector in usable[i + 1 :]:
            pairs_examined += 1
            cosine = _cosine(left_vector, right_vector)
            if cosine < threshold:
                continue
            dedupe_key = _decision_conflict_dedupe_key(left.id, right.id)
            if dedupe_key in emitted_keys:
                continue
            inserted = await _insert_decision_conflict_proposal(
                env_id=env_id,
                dream_run_id=dream_run_id,
                a_id=left.id,
                b_id=right.id,
                cosine=cosine,
            )
            emitted_keys.add(dedupe_key)
            if inserted:
                proposals_emitted += 1
            else:
                proposals_skipped_existing += 1

    return DecisionConflictPassResult(
        env_id=env_id,
        decisions_examined=len(rows),
        pairs_examined=pairs_examined,
        proposals_emitted=proposals_emitted,
        proposals_skipped_existing=proposals_skipped_existing,
        missing_vectors=missing_vectors,
        items_capped=len(rows) == SCAN_LIMIT,
        duration_seconds=time.perf_counter() - started,
    )
