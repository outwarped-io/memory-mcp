"""Unit tests for :mod:`memory_mcp.db.graph.base` value types + factory.

Tests focus on the protocol contract:

* ``GraphEdge`` rejects cross-env construction.
* ``RESERVED_ATTRS`` is the documented set.
* ``get_graph_store`` dispatches by ``Settings.graph_backend``.
* ``NeighborHit.path`` length == ``path_length`` is the documented
  invariant — though we don't enforce it in the dataclass (the
  backends are responsible). We assert it against a hand-rolled hit.

Backend-specific behaviour (Neo4j MERGE, Postgres recursive CTE) lands
in their own test modules.
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest

from memory_mcp.config import Settings
from memory_mcp.db.graph import (
    RESERVED_ATTRS,
    GraphEdge,
    GraphNodeRef,
    GraphPathStep,
    NeighborHit,
    get_graph_store,
)


def _ref(env_id, kind="entity"):
    return GraphNodeRef(env_id=env_id, kind=kind, record_id=uuid4())


def test_reserved_attrs_set() -> None:
    assert frozenset({"env_id", "kind", "record_id", "id"}) == RESERVED_ATTRS


def test_graph_edge_same_env_ok() -> None:
    env = uuid4()
    src = _ref(env)
    dst = _ref(env)
    edge = GraphEdge(env_id=env, src=src, dst=dst, edge_type="describes")
    assert edge.env_id == env
    assert edge.properties == {}


def test_graph_edge_rejects_mismatched_src_env() -> None:
    src_env = uuid4()
    dst_env = uuid4()
    src = _ref(src_env)
    dst = _ref(dst_env)
    with pytest.raises(ValueError, match="env mismatch"):
        GraphEdge(env_id=src_env, src=src, dst=dst, edge_type="describes")


def test_graph_edge_rejects_mismatched_edge_env() -> None:
    env = uuid4()
    other = uuid4()
    src = _ref(env)
    dst = _ref(env)
    with pytest.raises(ValueError, match="env mismatch"):
        GraphEdge(env_id=other, src=src, dst=dst, edge_type="describes")


def test_neighbor_hit_path_length_matches_path() -> None:
    env = uuid4()
    a = _ref(env)
    b = _ref(env)
    step = GraphPathStep(src=a, dst=b, edge_type="describes")
    hit = NeighborHit(node=b, path_length=1, path=(step,))
    # Documented invariant: path_length == len(path) for backend-produced hits.
    assert len(hit.path) == hit.path_length


def test_factory_returns_neo4j_store_when_configured() -> None:
    s = Settings(graph_backend="neo4j")  # type: ignore[arg-type]
    with patch("memory_mcp.db.graph.neo4j.Neo4jGraphStore", create=True) as mock_cls:
        mock_cls.return_value = object()
        store = get_graph_store(s)
    mock_cls.assert_called_once_with(s)
    assert store is mock_cls.return_value


def test_factory_returns_postgres_store_when_configured() -> None:
    s = Settings(graph_backend="postgres")  # type: ignore[arg-type]
    with patch("memory_mcp.db.graph.postgres.PostgresGraphStore", create=True) as mock_cls:
        mock_cls.return_value = object()
        store = get_graph_store(s)
    mock_cls.assert_called_once_with(s)
    assert store is mock_cls.return_value


def test_factory_rejects_unknown_backend() -> None:
    class _StubSettings:
        graph_backend = "redis"

    with pytest.raises(ValueError, match="unsupported graph_backend"):
        get_graph_store(_StubSettings())  # type: ignore[arg-type]
