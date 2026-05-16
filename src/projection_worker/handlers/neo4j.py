"""Neo4j handler for the projection worker.

Translates leased ``LeasedEvent`` rows for ``aggregate_type ∈ {entity,
relation, task}`` into ``GraphStore`` operations. The handler is backend-agnostic
— it calls into ``GraphStore`` via the protocol, so the same code path
works against ``Neo4jGraphStore`` (default) and ``PostgresGraphStore``
(no-op for upsert/delete; canonical IS the projection).

Routing rules
-------------

* ``aggregate_type=entity``, ``op ∈ {upsert, update}`` → ``upsert_node``
  with attribute map drawn from the payload (canonical_name,
  normalized_name, kind, aliases as a comma-joined string for indexing).
* ``aggregate_type=entity``, ``op=tombstone`` → ``delete_subgraph`` for
  the merged entity. Note: ``ent_merge`` orchestration is responsible
  for emitting relation-update events BEFORE the entity-tombstone so
  the kept entity's edges are re-pointed first.
* ``aggregate_type=relation``, ``op ∈ {upsert, update}`` → ``upsert_edge``.
  Endpoint stub-creation is handled inside ``upsert_edge`` (Neo4j
  MERGE on both endpoints), so memory endpoints don't need a prior
  ``upsert_node`` event.
* ``aggregate_type=relation``, ``op=tombstone`` → relation deletion is
  not yet a Phase 1 capability (no ``rel_delete`` tool); guard with
  a runtime check that raises ``NotImplementedError`` so a future
  payload-only addition surfaces clearly.
* Any other aggregate type is a routing bug — raises ``ValueError``.

The handler is **idempotent on retry**: ``GraphStore.upsert_*`` MERGE
on identity, and ``delete_subgraph`` is no-op-on-missing.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from memory_mcp.db.graph.base import (
    GraphEdge,
    GraphNodeRef,
    GraphStore,
    NodeKind,
)
from memory_mcp.db.types import OutboxAggregateType, OutboxOp
from projection_worker.lease import LeasedEvent

log = logging.getLogger(__name__)


def _entity_attrs_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the entity payload to the small attribute set we replicate.

    Identity fields (``env_id``, ``id``, ``record_id``) are NOT included
    — the Protocol forbids their presence in ``attrs`` (they live on
    :class:`GraphNodeRef`). Only fields useful for graph-side filtering
    or display are projected; the canonical truth lives in Postgres.
    """
    return {
        "kind_tag": payload.get("kind"),
        "canonical_name": payload.get("canonical_name"),
        "normalized_name": payload.get("normalized_name"),
        # Aliases are flattened into a Neo4j-friendly array; we use the
        # raw list so backends can index/filter as appropriate.
        "aliases": list(payload.get("aliases") or []),
        "version": payload.get("version"),
        "updated_at": payload.get("updated_at"),
    }


def _task_attrs_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the task payload to the small attribute set we replicate."""
    return {
        "title": payload.get("title"),
        "status": payload.get("status"),
        "priority": payload.get("priority"),
        "version": payload.get("version"),
    }


def _node_ref_from_endpoint(env_id: UUID, endpoint: dict[str, Any]) -> GraphNodeRef:
    """Translate a relation-payload endpoint to a :class:`GraphNodeRef`.

    Endpoint shape: ``{"kind": "entity"|"memory", "id": "<uuid>",
    "node_id": "<uuid>"}``. We use ``id`` (the canonical record id),
    not ``node_id`` (the ``graph_nodes`` registry id) — the latter is
    a Postgres impl detail.
    """
    kind: NodeKind = endpoint["kind"]
    return GraphNodeRef(
        env_id=env_id,
        kind=kind,
        record_id=UUID(str(endpoint["id"])),
    )


async def handle_neo4j_event(
    event: LeasedEvent,
    *,
    graph_store: GraphStore,
) -> None:
    """Apply a single leased event to the graph projection."""
    if event.aggregate_type == OutboxAggregateType.entity.value:
        await _handle_entity_event(event, graph_store=graph_store)
        return
    if event.aggregate_type == OutboxAggregateType.relation.value:
        await _handle_relation_event(event, graph_store=graph_store)
        return
    if event.aggregate_type == OutboxAggregateType.task.value:
        await _handle_task_event(event, graph_store=graph_store)
        return
    raise ValueError(
        f"neo4j handler received aggregate_type={event.aggregate_type!r}; "
        "expected 'entity' or 'relation' or 'task'"
    )


async def _handle_entity_event(
    event: LeasedEvent,
    *,
    graph_store: GraphStore,
) -> None:
    payload = event.payload
    node = GraphNodeRef(
        env_id=event.env_id,
        kind="entity",
        record_id=event.aggregate_id,
    )
    if event.op == OutboxOp.tombstone.value:
        await graph_store.delete_subgraph(env_id=event.env_id, nodes=[node])
        log.debug(
            "neo4j entity tombstone applied: event_id=%s entity_id=%s",
            event.event_id, event.aggregate_id,
        )
        return

    if event.op not in (OutboxOp.upsert.value, OutboxOp.update.value):
        raise ValueError(
            f"neo4j entity handler: unexpected op={event.op!r} "
            f"for event_id={event.event_id}"
        )

    attrs = _entity_attrs_from_payload(payload)
    await graph_store.upsert_node(node, attrs=attrs)
    log.debug(
        "neo4j entity upsert applied: event_id=%s entity_id=%s op=%s",
        event.event_id, event.aggregate_id, event.op,
    )


async def _handle_relation_event(
    event: LeasedEvent,
    *,
    graph_store: GraphStore,
) -> None:
    payload = event.payload
    if event.op == OutboxOp.tombstone.value:
        # ``rel_delete`` is not a Phase 1 capability; a tombstone reaching
        # this handler indicates a future payload addition. Surface
        # loudly so the dead-letter path catches it.
        raise NotImplementedError(
            "neo4j relation handler: tombstone op not yet supported "
            f"(event_id={event.event_id}, relation_id={event.aggregate_id})"
        )

    if event.op not in (OutboxOp.upsert.value, OutboxOp.update.value):
        raise ValueError(
            f"neo4j relation handler: unexpected op={event.op!r} "
            f"for event_id={event.event_id}"
        )

    src = _node_ref_from_endpoint(event.env_id, payload["src"])
    dst = _node_ref_from_endpoint(event.env_id, payload["dst"])
    edge = GraphEdge(
        env_id=event.env_id,
        src=src,
        dst=dst,
        edge_type=str(payload["type"]),
        properties=dict(payload.get("properties") or {}),
    )
    await graph_store.upsert_edge(edge)
    log.debug(
        "neo4j relation upsert applied: event_id=%s relation_id=%s "
        "src=%s/%s dst=%s/%s type=%s",
        event.event_id, event.aggregate_id,
        src.kind, src.record_id, dst.kind, dst.record_id, edge.edge_type,
    )


async def _handle_task_event(
    event: LeasedEvent,
    *,
    graph_store: GraphStore,
) -> None:
    if event.op == OutboxOp.tombstone.value:
        log.warning(
            "neo4j task tombstone skipped: event_id=%s task_id=%s",
            event.event_id, event.aggregate_id,
        )
        return

    if event.op not in (OutboxOp.upsert.value, OutboxOp.update.value):
        raise ValueError(
            f"neo4j task handler: unexpected op={event.op!r} "
            f"for event_id={event.event_id}"
        )

    node = GraphNodeRef(env_id=event.env_id, kind="task", record_id=event.aggregate_id)
    await graph_store.upsert_node(node, attrs=_task_attrs_from_payload(event.payload))
    log.debug(
        "neo4j task upsert applied: event_id=%s task_id=%s op=%s",
        event.event_id, event.aggregate_id, event.op,
    )


__all__ = ["handle_neo4j_event"]
