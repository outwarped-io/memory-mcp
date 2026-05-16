"""Unit tests for the Neo4j projection handler.

We mock :class:`memory_mcp.db.graph.base.GraphStore` to avoid spinning up
a real Neo4j (or Postgres) and assert the handler translates leased
events to the right Protocol calls.

Tested branches:

* ``aggregate_type=entity, op=upsert`` → ``upsert_node`` with attribute
  projection (no reserved keys leak through).
* ``aggregate_type=entity, op=update`` → same path.
* ``aggregate_type=entity, op=tombstone`` → ``delete_subgraph``.
* ``aggregate_type=relation, op=upsert`` → ``upsert_edge`` with both
  endpoints constructed from the payload's ``src``/``dst`` blocks.
* ``aggregate_type=relation, op=tombstone`` → ``NotImplementedError``.
* Unknown ``aggregate_type`` → ``ValueError``.
* Endpoint with ``id`` (canonical) is preferred over ``node_id``
  (registry id) — boundary contract.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from memory_mcp.db.graph.base import RESERVED_ATTRS, GraphEdge, GraphNodeRef
from memory_mcp.db.types import OutboxAggregateType, OutboxOp, OutboxSink
from projection_worker.handlers.neo4j import handle_neo4j_event
from projection_worker.lease import LeasedEvent


def _event(
    *,
    aggregate_type: str,
    op: str,
    aggregate_id=None,
    env_id=None,
    payload: dict[str, Any] | None = None,
    aggregate_version: int = 1,
) -> LeasedEvent:
    return LeasedEvent(
        event_id=42,
        sink=OutboxSink.neo4j,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id or uuid4(),
        aggregate_version=aggregate_version,
        env_id=env_id or uuid4(),
        op=op,
        payload=payload or {},
        attempt_count=0,
        created_at=dt.datetime.now(dt.UTC),
    )


def test_entity_upsert_calls_upsert_node_with_attrs() -> None:
    env_id = uuid4()
    entity_id = uuid4()
    payload = {
        "entity_id": str(entity_id),
        "env_id": str(env_id),
        "kind": "person",
        "canonical_name": "Alice",
        "normalized_name": "alice",
        "aliases": ["Alice Smith"],
        "version": 1,
        "updated_at": "2026-05-09T00:00:00Z",
    }
    event = _event(
        aggregate_type=OutboxAggregateType.entity.value,
        op=OutboxOp.upsert.value,
        aggregate_id=entity_id,
        env_id=env_id,
        payload=payload,
    )
    store = AsyncMock()

    asyncio.run(handle_neo4j_event(event, graph_store=store))

    store.upsert_node.assert_awaited_once()
    (node,), kwargs = store.upsert_node.call_args
    assert isinstance(node, GraphNodeRef)
    assert node.env_id == env_id
    assert node.kind == "entity"
    assert node.record_id == entity_id

    attrs = kwargs["attrs"]
    assert attrs["canonical_name"] == "Alice"
    assert attrs["normalized_name"] == "alice"
    assert attrs["aliases"] == ["Alice Smith"]
    assert attrs["kind_tag"] == "person"
    # Reserved keys must NOT appear in attrs.
    assert RESERVED_ATTRS.isdisjoint(attrs.keys())


def test_entity_update_uses_same_path() -> None:
    payload = {
        "kind": "service",
        "canonical_name": "API",
        "normalized_name": "api",
        "aliases": [],
        "version": 2,
    }
    event = _event(
        aggregate_type=OutboxAggregateType.entity.value,
        op=OutboxOp.update.value,
        payload=payload,
    )
    store = AsyncMock()
    asyncio.run(handle_neo4j_event(event, graph_store=store))
    store.upsert_node.assert_awaited_once()


def test_entity_tombstone_calls_delete_subgraph() -> None:
    env_id = uuid4()
    entity_id = uuid4()
    payload = {
        "entity_id": str(entity_id),
        "env_id": str(env_id),
        "merged_into": str(uuid4()),
    }
    event = _event(
        aggregate_type=OutboxAggregateType.entity.value,
        op=OutboxOp.tombstone.value,
        aggregate_id=entity_id,
        env_id=env_id,
        payload=payload,
    )
    store = AsyncMock()

    asyncio.run(handle_neo4j_event(event, graph_store=store))

    store.delete_subgraph.assert_awaited_once()
    _, kwargs = store.delete_subgraph.call_args
    assert kwargs["env_id"] == env_id
    nodes = kwargs["nodes"]
    assert len(nodes) == 1
    assert nodes[0] == GraphNodeRef(
        env_id=env_id, kind="entity", record_id=entity_id
    )


def test_entity_unexpected_op_raises() -> None:
    event = _event(
        aggregate_type=OutboxAggregateType.entity.value,
        op="weirdop",
    )
    store = AsyncMock()
    with pytest.raises(ValueError, match="unexpected op"):
        asyncio.run(handle_neo4j_event(event, graph_store=store))


def test_relation_upsert_calls_upsert_edge() -> None:
    env_id = uuid4()
    relation_id = uuid4()
    src_record_id = uuid4()
    dst_record_id = uuid4()
    payload = {
        "relation_id": str(relation_id),
        "env_id": str(env_id),
        "type": "describes",
        "properties": {"weight": 0.7},
        "src": {
            "kind": "entity",
            "id": str(src_record_id),
            "node_id": str(uuid4()),
        },
        "dst": {
            "kind": "memory",
            "id": str(dst_record_id),
            "node_id": str(uuid4()),
        },
        "version": 1,
    }
    event = _event(
        aggregate_type=OutboxAggregateType.relation.value,
        op=OutboxOp.upsert.value,
        aggregate_id=relation_id,
        env_id=env_id,
        payload=payload,
    )
    store = AsyncMock()

    asyncio.run(handle_neo4j_event(event, graph_store=store))

    store.upsert_edge.assert_awaited_once()
    (edge,), _ = store.upsert_edge.call_args
    assert isinstance(edge, GraphEdge)
    assert edge.env_id == env_id
    assert edge.edge_type == "describes"
    assert edge.properties == {"weight": 0.7}
    assert edge.src.kind == "entity"
    assert edge.src.record_id == src_record_id
    assert edge.dst.kind == "memory"
    assert edge.dst.record_id == dst_record_id


def test_relation_uses_canonical_id_not_node_id() -> None:
    """The boundary contract: handler reads ``id`` (canonical), not ``node_id``."""
    env_id = uuid4()
    src_canonical = uuid4()
    src_registry = uuid4()  # different from canonical — would be a bug if used
    payload = {
        "type": "mentions",
        "properties": {},
        "src": {
            "kind": "entity",
            "id": str(src_canonical),
            "node_id": str(src_registry),
        },
        "dst": {
            "kind": "entity",
            "id": str(uuid4()),
            "node_id": str(uuid4()),
        },
    }
    event = _event(
        aggregate_type=OutboxAggregateType.relation.value,
        op=OutboxOp.upsert.value,
        env_id=env_id,
        payload=payload,
    )
    store = AsyncMock()
    asyncio.run(handle_neo4j_event(event, graph_store=store))

    (edge,), _ = store.upsert_edge.call_args
    assert edge.src.record_id == src_canonical
    assert edge.src.record_id != src_registry


def test_relation_tombstone_raises_not_implemented() -> None:
    event = _event(
        aggregate_type=OutboxAggregateType.relation.value,
        op=OutboxOp.tombstone.value,
        payload={"type": "x", "src": {}, "dst": {}},
    )
    store = AsyncMock()
    with pytest.raises(NotImplementedError, match="tombstone op not yet supported"):
        asyncio.run(handle_neo4j_event(event, graph_store=store))


def test_task_upsert_calls_upsert_node_with_attrs() -> None:
    env_id = uuid4()
    task_id = uuid4()
    event = _event(
        aggregate_type=OutboxAggregateType.task.value,
        op=OutboxOp.upsert.value,
        aggregate_id=task_id,
        env_id=env_id,
        payload={"title": "B1", "status": "pending", "priority": 10, "version": 3},
    )
    store = AsyncMock()
    asyncio.run(handle_neo4j_event(event, graph_store=store))
    store.upsert_node.assert_awaited_once()
    (node,), kwargs = store.upsert_node.call_args
    assert node == GraphNodeRef(env_id=env_id, kind="task", record_id=task_id)
    assert kwargs["attrs"] == {
        "title": "B1",
        "status": "pending",
        "priority": 10,
        "version": 3,
    }


def test_task_tombstone_logs_and_skips(caplog: pytest.LogCaptureFixture) -> None:
    logging.disable(logging.NOTSET)
    logging.getLogger("projection_worker.handlers.neo4j").disabled = False
    caplog.set_level(logging.WARNING, logger="projection_worker.handlers.neo4j")
    event = _event(
        aggregate_type=OutboxAggregateType.task.value,
        op=OutboxOp.tombstone.value,
    )
    store = AsyncMock()
    asyncio.run(handle_neo4j_event(event, graph_store=store))
    store.upsert_node.assert_not_called()
    store.delete_subgraph.assert_not_called()
    assert "task tombstone skipped" in caplog.text


def test_unknown_aggregate_type_raises() -> None:
    event = _event(aggregate_type="memory", op=OutboxOp.upsert.value)
    store = AsyncMock()
    with pytest.raises(ValueError, match="expected 'entity' or 'relation'"):
        asyncio.run(handle_neo4j_event(event, graph_store=store))
