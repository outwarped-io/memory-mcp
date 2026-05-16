"""Pure-Python schema + helper tests for the Sprint A browse tools.

DB-bound paths (query construction → execution against Postgres) are
exercised by the integration suite (``tests/integration/test_compose_e2e.py``
and friends). This module focuses on the request/response Pydantic
contracts, validation, default-statuses, filter-fingerprint dict
shapes, and cursor mismatch detection — all of which run without a
running database.
"""

from __future__ import annotations

import datetime as dt
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from memory_mcp.browse import (
    FacetBucket,
    MemBrowseRequest,
    MemBrowseResponse,
    MemFacetsRequest,
    MemFacetsResponse,
    _browse_filter_dict,
    _resolve_statuses,
)
from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.entities import EntityBrowseRequest, EntityBrowseResponse
from memory_mcp.errors import InvalidInputError
from memory_mcp.pagination import compute_filter_fingerprint, decode_cursor, encode_cursor
from memory_mcp.relations import (
    RelationBrowseHit,
    RelationBrowseRequest,
    RelationBrowseResponse,
)


# ---------------------------------------------------------------------------
# MemBrowseRequest schema
# ---------------------------------------------------------------------------


class TestMemBrowseRequest:
    def test_defaults(self) -> None:
        req = MemBrowseRequest()
        assert req.env_ids is None
        assert req.kinds is None
        assert req.tags is None
        assert req.statuses is None
        assert req.order_by == "updated_at"
        assert req.descending is True
        assert req.limit == 50
        assert req.cursor is None

    def test_limit_bounds(self) -> None:
        MemBrowseRequest(limit=1)
        MemBrowseRequest(limit=500)
        with pytest.raises(ValidationError):
            MemBrowseRequest(limit=0)
        with pytest.raises(ValidationError):
            MemBrowseRequest(limit=501)

    def test_order_by_locked_to_two_values(self) -> None:
        MemBrowseRequest(order_by="updated_at")
        MemBrowseRequest(order_by="created_at")
        with pytest.raises(ValidationError):
            MemBrowseRequest(order_by="salience")  # explicitly NOT allowed in v1

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MemBrowseRequest(query="hello")  # type: ignore[call-arg]

    def test_cursor_max_length(self) -> None:
        MemBrowseRequest(cursor="x" * 4096)
        with pytest.raises(ValidationError):
            MemBrowseRequest(cursor="x" * 4097)


# ---------------------------------------------------------------------------
# _resolve_statuses
# ---------------------------------------------------------------------------


class TestResolveStatuses:
    def test_default_is_proposed_active(self) -> None:
        assert _resolve_statuses(None) == ["proposed", "active"]
        assert _resolve_statuses([]) == ["proposed", "active"]

    def test_passes_through_valid_statuses(self) -> None:
        out = _resolve_statuses([MemoryStatus.archived, MemoryStatus.active])
        assert out == ["archived", "active"]

    def test_dedupes_preserve_order(self) -> None:
        out = _resolve_statuses(
            [MemoryStatus.active, MemoryStatus.active, MemoryStatus.proposed]
        )
        assert out == ["active", "proposed"]


# ---------------------------------------------------------------------------
# _browse_filter_dict shape (drives the fingerprint)
# ---------------------------------------------------------------------------


class TestBrowseFilterDict:
    def test_shape_with_all_filters(self) -> None:
        env_a = uuid4()
        env_b = uuid4()
        req = MemBrowseRequest(
            env_ids=[env_a, env_b],
            kinds=[MemoryKind.fact, MemoryKind.observation],
            tags=["b", "a"],
            statuses=[MemoryStatus.active],
            created_after=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            order_by="created_at",
            descending=False,
        )
        d = _browse_filter_dict(req, [env_a, env_b])
        assert d["env_ids"] == [env_a, env_b]
        assert d["kinds"] == ["fact", "observation"]
        assert d["tags"] == ["a", "b"]  # sorted
        assert d["statuses"] == ["active"]
        assert d["order_by"] == "created_at"
        assert d["descending"] is False

    def test_fingerprint_changes_with_filter(self) -> None:
        env_a = uuid4()
        a = _browse_filter_dict(MemBrowseRequest(), [env_a])
        b = _browse_filter_dict(MemBrowseRequest(tags=["x"]), [env_a])
        assert compute_filter_fingerprint(a) != compute_filter_fingerprint(b)


# ---------------------------------------------------------------------------
# Cursor round-trip via browse-style payload
# ---------------------------------------------------------------------------


class TestBrowseCursorRoundtrip:
    def test_mem_browse_cursor_round_trip(self) -> None:
        env_a = uuid4()
        req = MemBrowseRequest(env_ids=[env_a], order_by="updated_at")
        filter_dict = _browse_filter_dict(req, [env_a])
        fp = compute_filter_fingerprint(filter_dict)
        last_id = uuid4()
        last_ts = dt.datetime(2026, 5, 1, 12, 0, tzinfo=dt.UTC)
        cur = encode_cursor(
            filter_fingerprint=fp,
            order_field="updated_at",
            order_value=last_ts,
            tiebreak_id=last_id,
            direction="desc",
        )
        decoded = decode_cursor(
            cur, expected_fingerprint=fp,
            expected_order_field="updated_at",
            expected_direction="desc",
        )
        assert decoded.tiebreak_id == last_id


# ---------------------------------------------------------------------------
# MemFacetsRequest
# ---------------------------------------------------------------------------


class TestMemFacetsRequest:
    def test_defaults(self) -> None:
        req = MemFacetsRequest()
        assert req.facets == ["kind", "status", "tag"]
        assert req.tag_limit == 50
        assert req.accuracy == "exact"
        assert req.max_rows == 100_000

    def test_facets_validated(self) -> None:
        MemFacetsRequest(facets=["kind"])
        MemFacetsRequest(facets=["kind", "status", "tag", "month"])
        with pytest.raises(ValidationError):
            MemFacetsRequest(facets=["lifecycle"])  # type: ignore[list-item]

    def test_max_rows_min(self) -> None:
        MemFacetsRequest(max_rows=1_000)
        with pytest.raises(ValidationError):
            MemFacetsRequest(max_rows=999)


class TestFacetBucket:
    def test_shape(self) -> None:
        b = FacetBucket(value="fact", count=12)
        assert b.value == "fact"
        assert b.count == 12

    def test_response_carries_schema_version(self) -> None:
        r = MemFacetsResponse(total=0, by_env={}, facets={})
        assert r.schema_version == 1
        assert r.approximate is False


# ---------------------------------------------------------------------------
# EntityBrowseRequest
# ---------------------------------------------------------------------------


class TestEntityBrowseRequest:
    def test_defaults(self) -> None:
        req = EntityBrowseRequest()
        assert req.order_by == "canonical_name"
        assert req.descending is False
        assert req.limit == 50
        assert req.name_prefix is None

    def test_name_prefix_min_length(self) -> None:
        EntityBrowseRequest(name_prefix="a")
        with pytest.raises(ValidationError):
            EntityBrowseRequest(name_prefix="")

    def test_name_prefix_max_length(self) -> None:
        EntityBrowseRequest(name_prefix="x" * 400)
        with pytest.raises(ValidationError):
            EntityBrowseRequest(name_prefix="x" * 401)

    def test_order_by_locked(self) -> None:
        EntityBrowseRequest(order_by="created_at")
        with pytest.raises(ValidationError):
            EntityBrowseRequest(order_by="updated_at")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RelationBrowseRequest
# ---------------------------------------------------------------------------


class TestRelationBrowseRequest:
    def test_defaults(self) -> None:
        req = RelationBrowseRequest()
        assert req.descending is True
        assert req.limit == 100
        assert req.types is None

    def test_types_max_20(self) -> None:
        RelationBrowseRequest(types=[f"type_{i}" for i in range(20)])
        with pytest.raises(ValidationError):
            RelationBrowseRequest(types=[f"type_{i}" for i in range(21)])

    def test_endpoint_kinds_locked(self) -> None:
        RelationBrowseRequest(src_kind="entity", dst_kind="memory")
        with pytest.raises(ValidationError):
            RelationBrowseRequest(src_kind="environment")  # type: ignore[arg-type]

    def test_id_requires_matching_kind(self) -> None:
        # Pinning src_id without src_kind is ambiguous (entity_id and
        # memory_id share the UUID namespace).
        with pytest.raises(ValidationError):
            RelationBrowseRequest(src_id=uuid4())
        with pytest.raises(ValidationError):
            RelationBrowseRequest(dst_id=uuid4())
        # With a kind, both endpoints are accepted.
        RelationBrowseRequest(src_kind="entity", src_id=uuid4())
        RelationBrowseRequest(dst_kind="memory", dst_id=uuid4())

    def test_hit_shape(self) -> None:
        h = RelationBrowseHit(
            id=uuid4(),
            env_id=uuid4(),
            type="mentions",
            src_kind="entity",
            src_id=uuid4(),
            dst_kind="memory",
            dst_id=uuid4(),
            properties={"weight": 1.0},
            created_at=dt.datetime(2026, 5, 1, tzinfo=dt.UTC),
            updated_at=dt.datetime(2026, 5, 1, tzinfo=dt.UTC),
        )
        assert h.src_kind == "entity"
        assert h.dst_kind == "memory"
