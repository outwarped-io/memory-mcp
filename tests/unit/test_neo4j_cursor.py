"""Unit tests for ``Neo4jGraphStore`` cursor encoding + shape-mismatch.

The cursor is **opaque** to callers but must round-trip across calls
with the same query shape and reject shape-mismatched cursors with
``ValueError``.  This catches regressions where a refactor changes
what fields contribute to the shape key without rotating the cursor
format.
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from memory_mcp.config import Settings
from memory_mcp.db.graph.base import GraphNodeRef
from memory_mcp.db.graph.neo4j import (
    Neo4jDriver,
    Neo4jGraphStore,
    _decode_cursor,
    _encode_cursor,
    _query_shape_key,
)

# ---------------------------------------------------------------------------
# Pure encode/decode round-trips
# ---------------------------------------------------------------------------


def test_encode_decode_round_trip() -> None:
    payload = {"shape": {"k": "v"}, "offset": 42}
    encoded = _encode_cursor(payload)
    assert isinstance(encoded, str)
    # Cursor must be url-safe base64 (no padding-free guarantee but
    # round-trip works regardless).
    decoded = _decode_cursor(encoded)
    assert decoded == payload


def test_decode_invalid_base64_raises_value_error() -> None:
    with pytest.raises(ValueError, match="invalid cursor"):
        _decode_cursor("!!!not-base64!!!")


def test_decode_invalid_json_raises_value_error() -> None:
    raw = base64.urlsafe_b64encode(b"not-json").decode("ascii")
    with pytest.raises(ValueError, match="invalid cursor"):
        _decode_cursor(raw)


def test_decode_empty_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match="invalid cursor"):
        _decode_cursor("")


# ---------------------------------------------------------------------------
# Query-shape key — every dimension that affects results must be in the key
# ---------------------------------------------------------------------------


def test_shape_key_includes_all_query_dimensions() -> None:
    env = uuid4()
    rid = uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=rid)
    shape = _query_shape_key(
        node=node,
        hops=2,
        direction="out",
        edge_types=["a", "b"],
        kinds=["entity"],
        limit=20,
    )
    assert shape["n"] == [str(env), "entity", str(rid)]
    assert shape["h"] == 2
    assert shape["d"] == "out"
    assert shape["e"] == ["a", "b"]
    assert shape["k"] == ["entity"]
    assert shape["l"] == 20


def test_shape_key_normalizes_edge_types_order() -> None:
    """Reordered edge_types must produce the same shape key — the
    semantic result is identical regardless of input order."""
    env, rid = uuid4(), uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=rid)
    s1 = _query_shape_key(
        node=node,
        hops=1,
        direction="both",
        edge_types=["a", "b"],
        kinds=None,
        limit=10,
    )
    s2 = _query_shape_key(
        node=node,
        hops=1,
        direction="both",
        edge_types=["b", "a"],
        kinds=None,
        limit=10,
    )
    assert s1 == s2


def test_shape_key_distinguishes_hops() -> None:
    env, rid = uuid4(), uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=rid)
    s1 = _query_shape_key(
        node=node,
        hops=1,
        direction="both",
        edge_types=None,
        kinds=None,
        limit=10,
    )
    s2 = _query_shape_key(
        node=node,
        hops=2,
        direction="both",
        edge_types=None,
        kinds=None,
        limit=10,
    )
    assert s1 != s2


def test_shape_key_distinguishes_direction() -> None:
    env, rid = uuid4(), uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=rid)
    s1 = _query_shape_key(
        node=node,
        hops=1,
        direction="out",
        edge_types=None,
        kinds=None,
        limit=10,
    )
    s2 = _query_shape_key(
        node=node,
        hops=1,
        direction="in",
        edge_types=None,
        kinds=None,
        limit=10,
    )
    assert s1 != s2


def test_shape_key_distinguishes_node() -> None:
    env = uuid4()
    rid_a, rid_b = uuid4(), uuid4()
    node_a = GraphNodeRef(env_id=env, kind="entity", record_id=rid_a)
    node_b = GraphNodeRef(env_id=env, kind="entity", record_id=rid_b)
    s1 = _query_shape_key(
        node=node_a,
        hops=1,
        direction="both",
        edge_types=None,
        kinds=None,
        limit=10,
    )
    s2 = _query_shape_key(
        node=node_b,
        hops=1,
        direction="both",
        edge_types=None,
        kinds=None,
        limit=10,
    )
    assert s1 != s2


# ---------------------------------------------------------------------------
# Cursor shape-mismatch detection in neighbors()
# ---------------------------------------------------------------------------


def _build_store(records: list[dict[str, Any]]) -> Neo4jGraphStore:
    """Build a Neo4jGraphStore whose driver returns ``records``."""

    async def _aiter(self: Any) -> Any:
        for rec in records:
            yield rec

    result = MagicMock()
    result.__aiter__ = _aiter
    session = MagicMock()
    session.run = AsyncMock(return_value=result)

    @asynccontextmanager
    async def _sessionmaker() -> Any:
        yield session

    drv_mock = MagicMock()
    drv_mock.session = _sessionmaker

    real_driver = Neo4jDriver(Settings(graph_backend="neo4j"))  # type: ignore[arg-type]
    real_driver._driver = drv_mock  # type: ignore[attr-defined]

    store = Neo4jGraphStore.__new__(Neo4jGraphStore)
    store._driver = real_driver  # type: ignore[attr-defined]
    return store


def test_neighbors_rejects_cursor_from_different_node() -> None:
    env = uuid4()
    node_a = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())
    node_b = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())
    other_shape = _query_shape_key(
        node=node_b,
        hops=1,
        direction="both",
        edge_types=None,
        kinds=None,
        limit=10,
    )
    bad_cursor = _encode_cursor({"shape": other_shape, "offset": 10})
    store = _build_store(records=[])

    with pytest.raises(ValueError, match="shape mismatch"):
        asyncio.run(
            store.neighbors(
                node_a,
                hops=1,
                direction="both",
                edge_types=None,
                kinds=None,
                limit=10,
                cursor=bad_cursor,
            )
        )


def test_neighbors_rejects_cursor_from_different_hops() -> None:
    env = uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())
    other_shape = _query_shape_key(
        node=node,
        hops=2,
        direction="both",
        edge_types=None,
        kinds=None,
        limit=10,
    )
    bad_cursor = _encode_cursor({"shape": other_shape, "offset": 5})
    store = _build_store(records=[])

    with pytest.raises(ValueError, match="shape mismatch"):
        asyncio.run(
            store.neighbors(
                node,
                hops=1,
                direction="both",
                edge_types=None,
                kinds=None,
                limit=10,
                cursor=bad_cursor,
            )
        )


def test_neighbors_accepts_matching_cursor() -> None:
    env = uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())
    matching_shape = _query_shape_key(
        node=node,
        hops=1,
        direction="both",
        edge_types=None,
        kinds=None,
        limit=10,
    )
    cursor = _encode_cursor({"shape": matching_shape, "offset": 0})
    store = _build_store(records=[])

    # Should not raise.
    hits, next_cursor = asyncio.run(
        store.neighbors(
            node,
            hops=1,
            direction="both",
            edge_types=None,
            kinds=None,
            limit=10,
            cursor=cursor,
        )
    )
    assert hits == []
    assert next_cursor is None


def test_neighbors_returns_next_cursor_when_more_records() -> None:
    """When the result set exceeds limit, neighbors() returns a non-None
    next_cursor whose shape matches the original query."""
    env = uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())
    # limit=2 → fetch_limit=3; supply 3 records → has_more = True
    fake_records = [
        {
            "term_label": "Entity",
            "term_id": str(uuid4()),
            "plen": 1,
            "rel_types": ["describes"],
            "src_ids": [str(node.record_id)],
            "dst_ids": [str(uuid4())],
            "src_labels": ["Entity"],
            "dst_labels": ["Entity"],
        }
        for _ in range(3)
    ]
    store = _build_store(records=fake_records)

    hits, next_cursor = asyncio.run(
        store.neighbors(
            node,
            hops=1,
            direction="both",
            edge_types=None,
            kinds=None,
            limit=2,
        )
    )
    assert len(hits) == 2
    assert next_cursor is not None
    decoded = _decode_cursor(next_cursor)
    expected_shape = _query_shape_key(
        node=node,
        hops=1,
        direction="both",
        edge_types=None,
        kinds=None,
        limit=2,
    )
    assert decoded["shape"] == expected_shape
    assert decoded["offset"] == 2  # initial offset 0 + limit 2


def test_neighbors_no_next_cursor_when_no_more_records() -> None:
    env = uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())
    # limit=5 → fetch_limit=6; supply 2 records → has_more = False
    fake_records = [
        {
            "term_label": "Memory",
            "term_id": str(uuid4()),
            "plen": 1,
            "rel_types": ["describes"],
            "src_ids": [str(node.record_id)],
            "dst_ids": [str(uuid4())],
            "src_labels": ["Entity"],
            "dst_labels": ["Memory"],
        }
        for _ in range(2)
    ]
    store = _build_store(records=fake_records)

    hits, next_cursor = asyncio.run(
        store.neighbors(
            node,
            hops=1,
            direction="both",
            edge_types=None,
            kinds=None,
            limit=5,
        )
    )
    assert len(hits) == 2
    assert next_cursor is None


def test_cursor_payload_is_url_safe_base64_json() -> None:
    """The cursor format is opaque but documented as base64-url-safe
    JSON.  Round-trip via stdlib should produce a small dict."""
    payload = {"shape": {"k": [1, 2]}, "offset": 7}
    cursor = _encode_cursor(payload)
    # url-safe alphabet: ASCII letters, digits, '-', '_', '='
    assert all(c.isalnum() or c in "-_=" for c in cursor)
    raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
    parsed = json.loads(raw)
    assert parsed == payload
