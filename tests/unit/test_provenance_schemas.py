"""Pure-Python schema tests for the Sprint B provenance tools."""

from __future__ import annotations

import datetime as dt
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from memory_mcp.db.models import Memory, MemorySource
from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.memories import MemoryResponse
from memory_mcp.provenance import (
    MemLineageEdge,
    MemLineageRequest,
    MemLineageResponse,
    MemSourceHit,
    MemSourcesBrowseRequest,
    MemSourcesBrowseResponse,
    _apply_lineage_edge_cap,
    _apply_sources_filters,
    _lineage_edges_from_rows,
    _lineage_params,
    _lineage_sql,
)


def _make_memory_response(env_id: UUID | None = None) -> MemoryResponse:
    now = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
    return MemoryResponse(
        id=uuid4(),
        env_id=env_id or uuid4(),
        kind=MemoryKind.fact,
        status=MemoryStatus.active,
        title="provenance",
        body="provenance body",
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
# MemLineageRequest schema
# ---------------------------------------------------------------------------


class TestMemLineageRequest:
    def test_lineage_defaults(self) -> None:
        req = MemLineageRequest(memory_id=uuid4())
        assert req.direction == "both"
        assert req.relations is None
        assert req.max_depth == 10
        assert req.env_id is None

    def test_lineage_direction_enum(self) -> None:
        MemLineageRequest(memory_id=uuid4(), direction="ancestors")
        MemLineageRequest(memory_id=uuid4(), direction="descendants")
        MemLineageRequest(memory_id=uuid4(), direction="both")
        with pytest.raises(ValidationError):
            MemLineageRequest(memory_id=uuid4(), direction="sideways")  # type: ignore[arg-type]

    def test_lineage_max_depth_bounds(self) -> None:
        MemLineageRequest(memory_id=uuid4(), max_depth=1)
        MemLineageRequest(memory_id=uuid4(), max_depth=50)
        with pytest.raises(ValidationError):
            MemLineageRequest(memory_id=uuid4(), max_depth=0)
        with pytest.raises(ValidationError):
            MemLineageRequest(memory_id=uuid4(), max_depth=51)

    def test_lineage_max_edges_bounds(self) -> None:
        assert MemLineageRequest(memory_id=uuid4()).max_edges == 500
        MemLineageRequest(memory_id=uuid4(), max_edges=1)
        MemLineageRequest(memory_id=uuid4(), max_edges=5000)
        with pytest.raises(ValidationError):
            MemLineageRequest(memory_id=uuid4(), max_edges=0)
        with pytest.raises(ValidationError):
            MemLineageRequest(memory_id=uuid4(), max_edges=5001)

    def test_lineage_relations_max_length(self) -> None:
        MemLineageRequest(memory_id=uuid4(), relations=[f"rel_{i}" for i in range(20)])
        with pytest.raises(ValidationError):
            MemLineageRequest(memory_id=uuid4(), relations=[f"rel_{i}" for i in range(21)])

    def test_lineage_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            MemLineageRequest(memory_id=uuid4(), bogus=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# MemLineageEdge / MemLineageResponse schema
# ---------------------------------------------------------------------------


class TestMemLineageResponse:
    def test_edge_shape(self) -> None:
        e = MemLineageEdge(
            parent_memory_id=uuid4(),
            child_memory_id=uuid4(),
            relation="supersedes",
            created_at=dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC),
            depth=1,
        )
        assert e.depth == 1
        with pytest.raises(ValidationError):
            MemLineageEdge(
                parent_memory_id=uuid4(),
                child_memory_id=uuid4(),
                relation="supersedes",
                created_at=dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC),
                depth=0,
            )

    def test_lineage_response_shape(self) -> None:
        seed = _make_memory_response()
        r = MemLineageResponse(
            seed=seed,
            ancestors=[],
            descendants=[],
            nodes={seed.id: seed},
            truncated=False,
        )
        assert r.seed == seed
        assert r.ancestors == []
        assert r.descendants == []
        assert r.nodes == {seed.id: seed}
        assert r.truncated is False


# ---------------------------------------------------------------------------
# MemLineage SQL helpers
# ---------------------------------------------------------------------------


class TestMemLineageSql:
    def test_lineage_sql_filters_relations_in_seed_and_recursive_branches(self) -> None:
        req = MemLineageRequest(memory_id=uuid4(), relations=["supersedes"], max_depth=3)
        sql = _lineage_sql(req, ancestors=True)

        assert sql.count("ANY(:relations)") == 2
        assert "AND relation = ANY(:relations)" in sql
        assert "AND ml.relation = ANY(:relations)" in sql
        assert "ORDER BY depth ASC, created_at ASC, parent_memory_id ASC, child_memory_id ASC" in sql
        assert _lineage_params(req)["max_depth"] == 4

    def test_lineage_rows_trim_depth_probe_and_report_truncated(self) -> None:
        now = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
        rows = [
            {
                "parent_memory_id": uuid4(),
                "child_memory_id": uuid4(),
                "relation": "supersedes",
                "created_at": now,
                "depth": 1,
            },
            {
                "parent_memory_id": uuid4(),
                "child_memory_id": uuid4(),
                "relation": "supersedes",
                "created_at": now,
                "depth": 3,
            },
        ]

        edges, truncated = _lineage_edges_from_rows(rows, visible_max_depth=2)

        assert [edge.depth for edge in edges] == [1]
        assert truncated is True

    def test_lineage_edge_cap_truncates_combined_edges_by_depth_then_created_at(self) -> None:
        now = dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC)
        older = now - dt.timedelta(minutes=1)
        ancestor = MemLineageEdge(
            parent_memory_id=uuid4(),
            child_memory_id=uuid4(),
            relation="promoted_from",
            created_at=now,
            depth=2,
        )
        descendant_early = MemLineageEdge(
            parent_memory_id=uuid4(),
            child_memory_id=uuid4(),
            relation="supersedes",
            created_at=older,
            depth=1,
        )
        descendant_late = MemLineageEdge(
            parent_memory_id=uuid4(),
            child_memory_id=uuid4(),
            relation="supersedes",
            created_at=now,
            depth=1,
        )

        ancestors, descendants, truncated = _apply_lineage_edge_cap(
            [ancestor],
            [descendant_late, descendant_early],
            max_edges=2,
        )

        assert ancestors == []
        assert descendants == [descendant_early, descendant_late]
        assert truncated is True


# ---------------------------------------------------------------------------
# MemSourcesBrowseRequest schema
# ---------------------------------------------------------------------------


class TestMemSourcesBrowseRequest:
    def test_sources_defaults(self) -> None:
        req = MemSourcesBrowseRequest()
        assert req.hydrate_memories is False
        assert req.limit == 50
        assert req.descending is True

    def test_sources_max_lengths(self) -> None:
        MemSourcesBrowseRequest(memory_ids=[uuid4() for _ in range(100)])
        MemSourcesBrowseRequest(source_types=[f"type_{i}" for i in range(20)])
        MemSourcesBrowseRequest(source_refs=[f"ref_{i}" for i in range(100)])
        MemSourcesBrowseRequest(agent_ids=[uuid4() for _ in range(50)])

        with pytest.raises(ValidationError):
            MemSourcesBrowseRequest(memory_ids=[uuid4() for _ in range(101)])
        with pytest.raises(ValidationError):
            MemSourcesBrowseRequest(source_types=[f"type_{i}" for i in range(21)])
        with pytest.raises(ValidationError):
            MemSourcesBrowseRequest(source_refs=[f"ref_{i}" for i in range(101)])
        with pytest.raises(ValidationError):
            MemSourcesBrowseRequest(agent_ids=[uuid4() for _ in range(51)])

    def test_sources_limit_bounds(self) -> None:
        MemSourcesBrowseRequest(limit=1)
        MemSourcesBrowseRequest(limit=500)
        with pytest.raises(ValidationError):
            MemSourcesBrowseRequest(limit=0)
        with pytest.raises(ValidationError):
            MemSourcesBrowseRequest(limit=501)

    def test_sources_browse_filters_hidden_memories_at_row_level(self) -> None:
        stmt = select(MemorySource, Memory.env_id).join(Memory, Memory.id == MemorySource.memory_id)
        stmt = _apply_sources_filters(stmt, MemSourcesBrowseRequest(), env_ids=[uuid4()])

        sql = str(stmt)

        assert "memories.status IN" in sql


# ---------------------------------------------------------------------------
# MemSourceHit / MemSourcesBrowseResponse schema
# ---------------------------------------------------------------------------


class TestMemSourcesBrowseResponse:
    def test_source_hit_shape(self) -> None:
        h = MemSourceHit(
            id=12345,
            memory_id=uuid4(),
            env_id=uuid4(),
            source_type="file",
            source_ref=None,
            agent_id=None,
            created_at=dt.datetime(2026, 5, 10, 12, 0, 0, tzinfo=dt.UTC),
            evidence_span=None,
        )
        assert h.id == 12345
        assert isinstance(h.id, int)
        assert h.source_ref is None
        assert h.agent_id is None
        assert h.evidence_span is None

    def test_sources_response_shape(self) -> None:
        r = MemSourcesBrowseResponse(hits=[], next_cursor=None, nodes=None)
        assert r.hits == []
        assert r.next_cursor is None
        assert r.nodes is None

        defaulted = MemSourcesBrowseResponse(hits=[])
        assert defaulted.next_cursor is None
        assert defaulted.nodes is None
