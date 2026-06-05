"""Postgres-backed :class:`GraphStore` — recursive-CTE fallback.

For deployments that pin ``GRAPH_BACKEND=postgres``, canonical Postgres
**is** the projection: rows in ``graph_nodes`` + ``relations`` are
written by the originating tools (``ent_upsert``, ``rel_link``, …) in
the same transaction as their outbox events. The graph projection then
becomes a read view over those tables — there is no separate cache to
keep in sync.

Therefore:

* :meth:`upsert_node` and :meth:`upsert_edge` are **no-ops**. The
  projection-worker still calls them when ``GRAPH_BACKEND=postgres``
  so the worker's loop is uniform across backends; they just return
  immediately. Outbox-delivery rows for the ``neo4j`` sink are still
  marked done by the worker — this gives operators a consistent view
  of "no projection lag" even when no separate process exists.
* :meth:`delete_subgraph` is also a no-op. Canonical deletion from
  ``graph_nodes``/``relations`` happens at the originating tool's
  transaction (e.g. ``ent_merge`` cascading delete); there is no cache
  to clean up.
* :meth:`neighbors` runs a recursive CTE bounded by ``hops``, fetching
  paths up to depth N. Cursor pagination uses the same SKIP/LIMIT
  contract as the Neo4j impl, with deterministic ordering.

The Postgres impl is lighter than the Neo4j one — fewer features —
because canonical Postgres is the source of truth and there's no
projection lag to manage.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import text

from memory_mcp.config import Settings
from memory_mcp.db import postgres as pg
from memory_mcp.db.graph.base import (
    RESERVED_ATTRS,
    GraphEdge,
    GraphNodeRef,
    GraphPathStep,
    NeighborHit,
    NodeKind,
    TraversalDirection,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cursor encoding (mirrored from Neo4j impl — kept in sync via tests)
# ---------------------------------------------------------------------------


def _encode_cursor(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str) -> dict[str, Any]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        return json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cursor: {exc}") from exc


def _query_shape_key(
    *,
    node: GraphNodeRef,
    hops: int,
    direction: TraversalDirection,
    edge_types: Sequence[str] | None,
    kinds: Sequence[NodeKind] | None,
    limit: int,
) -> dict[str, Any]:
    return {
        "n": [str(node.env_id), node.kind, str(node.record_id)],
        "h": hops,
        "d": direction,
        "e": sorted(edge_types) if edge_types else None,
        "k": sorted(kinds) if kinds else None,
        "l": limit,
    }


# ---------------------------------------------------------------------------
# PostgresGraphStore
# ---------------------------------------------------------------------------


class PostgresGraphStore:
    """:class:`GraphStore` impl over canonical ``graph_nodes`` + ``relations``."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def init_schema(self) -> None:
        # Schema is owned by alembic; nothing to do.
        return

    async def close(self) -> None:
        # Engine lifecycle is owned by `db.postgres.dispose_engine`.
        return

    # -- Mutations (no-ops; canonical writes already happened) -----------

    async def upsert_node(
        self,
        node: GraphNodeRef,
        *,
        attrs: Mapping[str, Any],
    ) -> None:
        bad = RESERVED_ATTRS & attrs.keys()
        if bad:
            raise ValueError(
                f"upsert_node: attrs must not contain reserved keys {sorted(bad)}; identity is fixed by GraphNodeRef"
            )
        # No-op: canonical Postgres is the projection. The originating
        # tool already wrote `graph_nodes` + node-type rows in the
        # same transaction as the outbox event.
        return

    async def upsert_edge(self, edge: GraphEdge) -> None:
        # Same rationale as upsert_node — canonical write happened in
        # the originating tool. The same-env invariant is enforced by
        # GraphEdge's __post_init__, so we just return.
        _ = edge
        return

    async def delete_subgraph(
        self,
        *,
        env_id: UUID,
        nodes: Sequence[GraphNodeRef],
    ) -> None:
        for n in nodes:
            if n.env_id != env_id:
                raise ValueError(
                    f"delete_subgraph: node env_id {n.env_id} != {env_id}; cross-env deletion is not supported"
                )
        # No-op: canonical deletion (entity_merge / memory_delete_hard)
        # cascades through graph_nodes + relations FKs.
        return

    # -- Reads -----------------------------------------------------------

    async def neighbors(
        self,
        node: GraphNodeRef,
        *,
        hops: int = 1,
        direction: TraversalDirection = "both",
        edge_types: Sequence[str] | None = None,
        kinds: Sequence[NodeKind] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[list[NeighborHit], str | None]:
        if hops < 1:
            raise ValueError(f"hops must be >= 1, got {hops}")
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        shape = _query_shape_key(
            node=node,
            hops=hops,
            direction=direction,
            edge_types=edge_types,
            kinds=kinds,
            limit=limit,
        )
        offset = 0
        if cursor is not None:
            decoded = _decode_cursor(cursor)
            if decoded.get("shape") != shape:
                raise ValueError("cursor query-shape mismatch — cursors are not portable across calls")
            offset = int(decoded.get("offset", 0))

        # Build the recursive-CTE.  Each row in `walk` is a path of
        # length `plen` ending at `term_node_id` (a graph_nodes.id).
        # We then JOIN graph_nodes once for the terminal kind/record,
        # and project the path step-by-step.
        #
        # Direction-aware edge join:
        #   out:  src=current, dst=next
        #   in:   dst=current, src=next
        #   both: union of the two
        edge_join_clauses = {
            "out": "JOIN relations e ON e.src_node_id = w.term_node_id JOIN graph_nodes nx ON nx.id = e.dst_node_id ",
            "in": "JOIN relations e ON e.dst_node_id = w.term_node_id JOIN graph_nodes nx ON nx.id = e.src_node_id ",
        }

        if direction == "both":
            recursive_step = (
                "SELECT w.start_node_id, "
                "       nx.id AS term_node_id, "
                "       w.plen + 1 AS plen, "
                "       w.path_node_ids || nx.id, "
                "       w.path_edge_types || e.type, "
                "       w.path_src_ids || e.src_node_id, "
                "       w.path_dst_ids || e.dst_node_id "
                "FROM walk w "
                "JOIN relations e ON (e.src_node_id = w.term_node_id "
                "                     OR e.dst_node_id = w.term_node_id) "
                "JOIN graph_nodes nx ON nx.id = CASE "
                "    WHEN e.src_node_id = w.term_node_id THEN e.dst_node_id "
                "    ELSE e.src_node_id END "
                "WHERE w.plen < :hops "
                "AND nx.env_id = :env_id "
                "AND nx.id <> ALL(w.path_node_ids) "
            )
        else:
            recursive_step = (
                "SELECT w.start_node_id, "
                "       nx.id AS term_node_id, "
                "       w.plen + 1 AS plen, "
                "       w.path_node_ids || nx.id, "
                "       w.path_edge_types || e.type, "
                "       w.path_src_ids || e.src_node_id, "
                "       w.path_dst_ids || e.dst_node_id "
                "FROM walk w "
                f"{edge_join_clauses[direction]} "
                "WHERE w.plen < :hops "
                "AND nx.env_id = :env_id "
                "AND nx.id <> ALL(w.path_node_ids) "
            )

        # Edge-type filter applied as a WHERE on every path step's
        # edge_type — implemented post-walk via UNNEST + ALL.
        etype_filter = ""
        if edge_types:
            etype_filter = (
                "AND NOT EXISTS (  SELECT 1 FROM unnest(w.path_edge_types) AS et   WHERE et <> ALL(:edge_types) ) "
            )

        # Terminal-kind filter applied via JOIN on graph_nodes (we
        # already have the row).
        kind_filter = ""
        if kinds:
            kind_filter = "AND gn_term.node_type = ANY(:kinds) "

        sql = (
            "WITH RECURSIVE walk AS ( "
            "  SELECT gn0.id AS start_node_id, "
            "         gn0.id AS term_node_id, "
            "         0 AS plen, "
            "         ARRAY[gn0.id]::uuid[] AS path_node_ids, "
            "         ARRAY[]::text[] AS path_edge_types, "
            "         ARRAY[]::uuid[] AS path_src_ids, "
            "         ARRAY[]::uuid[] AS path_dst_ids "
            "  FROM graph_nodes gn0 "
            "  WHERE gn0.env_id = :env_id "
            "  AND gn0.node_type = :start_kind "
            "  AND CASE WHEN :start_kind = 'entity' THEN gn0.entity_id "
            "           WHEN :start_kind = 'memory' THEN gn0.memory_id "
            "           ELSE gn0.task_id END "
            "      = :start_record_id "
            "  UNION ALL "
            f"  {recursive_step} "
            ") "
            "SELECT gn_term.node_type AS term_kind, "
            "       CASE WHEN gn_term.node_type = 'entity' THEN gn_term.entity_id "
            "            WHEN gn_term.node_type = 'memory' THEN gn_term.memory_id "
            "            ELSE gn_term.task_id END AS term_record_id, "
            "       w.plen, "
            "       w.path_edge_types, "
            "       w.path_src_ids, "
            "       w.path_dst_ids "
            "FROM walk w "
            "JOIN graph_nodes gn_term ON gn_term.id = w.term_node_id "
            "WHERE w.plen >= 1 "
            f"{etype_filter}"
            f"{kind_filter}"
            "ORDER BY w.plen ASC, gn_term.node_type ASC, term_record_id ASC "
            "OFFSET :skip LIMIT :lim"
        )

        params: dict[str, Any] = {
            "env_id": str(node.env_id),
            "start_kind": node.kind,
            "start_record_id": str(node.record_id),
            "hops": hops,
            "skip": offset,
            "lim": limit + 1,
        }
        if edge_types:
            params["edge_types"] = list(edge_types)
        if kinds:
            params["kinds"] = list(kinds)

        async with pg.session_scope() as session:
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()

        has_more = len(rows) > limit
        rows = rows[:limit]

        # Resolve every node id we touched along the paths back to
        # (kind, record_id) in one bulk lookup.
        node_id_set: set[UUID] = set()
        for r in rows:
            for nid in r["path_src_ids"]:
                node_id_set.add(nid)
            for nid in r["path_dst_ids"]:
                node_id_set.add(nid)
        id_to_ref: dict[UUID, GraphNodeRef] = {}
        if node_id_set:
            async with pg.session_scope() as session:
                lookup = await session.execute(
                    text(
                        "SELECT id, node_type, "
                        "       COALESCE(entity_id, memory_id, task_id) AS record_id "
                        "FROM graph_nodes WHERE id = ANY(:ids)"
                    ),
                    {"ids": list(node_id_set)},
                )
                for row in lookup.mappings().all():
                    id_to_ref[row["id"]] = GraphNodeRef(
                        env_id=node.env_id,
                        kind=row["node_type"],
                        record_id=row["record_id"],
                    )

        hits: list[NeighborHit] = []
        for r in rows:
            term_kind: NodeKind = r["term_kind"]
            terminal = GraphNodeRef(
                env_id=node.env_id,
                kind=term_kind,
                record_id=r["term_record_id"],
            )
            steps: list[GraphPathStep] = []
            for i, etype in enumerate(r["path_edge_types"]):
                src_ref = id_to_ref.get(r["path_src_ids"][i])
                dst_ref = id_to_ref.get(r["path_dst_ids"][i])
                if src_ref is None or dst_ref is None:
                    continue
                steps.append(GraphPathStep(src=src_ref, dst=dst_ref, edge_type=etype))
            hits.append(
                NeighborHit(
                    node=terminal,
                    path_length=int(r["plen"]),
                    path=tuple(steps),
                )
            )

        next_cursor = None
        if has_more:
            next_cursor = _encode_cursor({"shape": shape, "offset": offset + limit})
        return hits, next_cursor


__all__ = ["PostgresGraphStore"]
