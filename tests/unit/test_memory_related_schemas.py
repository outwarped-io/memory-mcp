"""Pure-Python schema tests for the Sprint B memory-related tool."""

from __future__ import annotations

import datetime as dt
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.errors import InvalidCursorError
from memory_mcp.graph import MemRelatedHit, MemRelatedRequest, MemRelatedResponse, _memory_related_semantic
from memory_mcp.memories import MemoryResponse


def _make_memory_response(env_id: UUID | None = None) -> MemoryResponse:
    now = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
    return MemoryResponse(
        id=uuid4(),
        env_id=env_id or uuid4(),
        kind=MemoryKind.fact,
        status=MemoryStatus.active,
        title="related",
        body="related body",
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


# ---------------------------------------------------------------------------
# MemRelatedRequest schema
# ---------------------------------------------------------------------------


class TestMemRelatedRequest:
    def test_defaults(self) -> None:
        req = MemRelatedRequest(memory_id=uuid4())
        assert req.relation == "shared_entity"
        assert req.limit == 20
        assert req.cursor is None

    def test_relation_enum(self) -> None:
        MemRelatedRequest(memory_id=uuid4(), relation="shared_entity")
        MemRelatedRequest(memory_id=uuid4(), relation="semantic")
        with pytest.raises(ValidationError):
            MemRelatedRequest(memory_id=uuid4(), relation="lineage")  # type: ignore[arg-type]

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            MemRelatedRequest(memory_id=uuid4(), bogus=1)  # type: ignore[call-arg]

    def test_limit_bounds(self) -> None:
        MemRelatedRequest(memory_id=uuid4(), limit=1)
        MemRelatedRequest(memory_id=uuid4(), limit=500)
        with pytest.raises(ValidationError):
            MemRelatedRequest(memory_id=uuid4(), limit=0)
        with pytest.raises(ValidationError):
            MemRelatedRequest(memory_id=uuid4(), limit=501)


# ---------------------------------------------------------------------------
# MemRelatedHit / MemRelatedResponse schema
# ---------------------------------------------------------------------------


class TestMemRelatedHit:
    def test_hit_shape(self) -> None:
        memory_id = uuid4()
        shared_entity_id = uuid4()
        h = MemRelatedHit(
            memory_id=memory_id,
            score=2.0,
            shared_entity_ids=[shared_entity_id],
            memory=_make_memory_response(),
        )
        assert h.memory_id == memory_id
        assert h.score == 2.0
        assert h.shared_entity_ids == [shared_entity_id]

        semantic = MemRelatedHit(
            memory_id=uuid4(),
            score=0.75,
            shared_entity_ids=None,
            memory=_make_memory_response(),
        )
        assert semantic.shared_entity_ids is None


class TestMemRelatedResponse:
    def test_response_note_enum(self) -> None:
        MemRelatedResponse(hits=[], note="ok")
        MemRelatedResponse(hits=[], note="no_embedding")
        MemRelatedResponse(hits=[], note="vector_store_unavailable")
        with pytest.raises(ValidationError):
            MemRelatedResponse(hits=[], note="unknown")  # type: ignore[arg-type]

    def test_response_defaults(self) -> None:
        r = MemRelatedResponse(hits=[])
        assert r.next_cursor is None
        assert r.note == "ok"


async def test_memory_related_semantic_cursor_rejected() -> None:
    req = MemRelatedRequest(memory_id=uuid4(), relation="semantic", cursor="not-empty")

    with pytest.raises(InvalidCursorError, match="semantic mode does not support cursor pagination"):
        await _memory_related_semantic(
            req,
            env_id=uuid4(),
            settings=object(),  # type: ignore[arg-type]
            vector_store=None,
        )
