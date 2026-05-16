"""``GraphStore`` protocol — backend-agnostic graph projection surface.

Implemented by:

* :class:`memory_mcp.db.graph.neo4j.Neo4jGraphStore` (default, v1)
* :class:`memory_mcp.db.graph.postgres.PostgresGraphStore` — recursive-CTE
  fallback over ``graph_nodes`` + ``relations`` for small deployments

The projection-worker calls into a ``GraphStore`` to upsert / delete
graph nodes and edges from the canonical Postgres representation.
``memory_search`` calls ``neighbors`` for the graph stage of hybrid
ranking. ``ent_neighbors`` MCP tool calls ``neighbors`` directly.

Identity & env-scoping
----------------------

The canonical identity of a node is the triple
``(env_id, kind, record_id)``: ``kind ∈ {entity, memory, task}`` matches
``graph_nodes.node_type`` in Postgres, and ``record_id`` is the entity-
or memory-row id. Backends may keep their own internal ids (Neo4j's
internal id, the ``graph_nodes.id`` registry uuid for Postgres) but
never expose them to callers — the triple is the boundary contract.

Every method is **env-scoped**. ``neighbors`` traversal does not cross
``env_id`` boundaries. Cross-env edges are rejected: ``GraphEdge``
validates that ``edge.env_id == src.env_id == dst.env_id``.

Identity fields (``env_id``, ``kind``, ``record_id``) are reserved.
Implementations must preserve them; the ``attrs`` argument to
:meth:`GraphStore.upsert_node` MUST NOT contain any of these keys —
backends raise ``ValueError`` on conflict.

Memory lifecycle propagation
----------------------------

Outbox routing for v1 (Phase 2.1) sends only ``entity`` + ``relation``
events to the ``neo4j`` sink. Memory events (status changes, hard-delete)
do **not** flow to Neo4j. Graph search is therefore expected to
**post-filter** memory hits against canonical Postgres
``memories.status``: a ``Memory`` node may persist in the graph after
the canonical row was archived/retired. The cost is one short
Postgres query per page of graph hits — acceptable for v1.

Idempotency & MERGE-on-write
----------------------------

All mutations are idempotent. ``upsert_node`` MERGEs by
``(env_id, kind, record_id)``; ``upsert_edge`` MERGEs by
``(env_id, src, dst, edge_type)``. Implementations must tolerate
re-delivery from the outbox without producing duplicate nodes /
edges. ``delete_subgraph`` is also idempotent — deleting an absent
node is a no-op.

``upsert_edge`` MUST ensure both endpoints exist as stub nodes before
linking them — i.e., a MERGE on each endpoint with identity-only
attrs precedes the edge MERGE. This handles the Phase 2.1 design rule
that **memory nodes are projected lazily on first relation reference**:
the projection worker need not emit a separate ``upsert_node`` event
for memory endpoints. Real attribute backfill for those memory nodes
remains optional in v1 (graph search post-filters by canonical
Postgres).

Pagination & ordering
---------------------

``neighbors`` returns an opaque ``cursor`` for the next page. The cursor
is a base64-encoded JSON object owned by the implementation; callers
treat it as a string. Cursors are **bound to the query shape**
(source node, hops, edge_types, kinds, direction, limit). Implementations
SHOULD reject cursors whose embedded query shape mismatches the new
call — returning ``ValueError`` rather than undefined behaviour.

Within a page, results MUST follow a **deterministic total order**:

    (path_length ASC, score DESC NULLS LAST,
     node.kind ASC, node.record_id ASC)

The ordering is required so SKIP/LIMIT pagination does not duplicate or
drop rows under no-write conditions. Stability across mutations is
**not** guaranteed — concurrent writes may shift results between pages.
Stable seek-method cursors land if/when v1.5 demands it.

Entity resolution lives elsewhere
---------------------------------

The previous draft placed ``match_entities`` on this Protocol. After
review it has moved to :mod:`memory_mcp.search.entity_resolution`
because both the Neo4j and Postgres impls would simply delegate to
``entity_aliases`` — name resolution is a Postgres concern, not a
graph concern. The graph leg of ``mem_search`` calls the resolver
first, then ``GraphStore.neighbors`` for traversal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

NodeKind = Literal["entity", "memory", "task"]
TraversalDirection = Literal["out", "in", "both"]

# Reserved keys that may not appear in ``upsert_node(..., attrs=...)``.
RESERVED_ATTRS: frozenset[str] = frozenset({"env_id", "kind", "record_id", "id"})


@dataclass(frozen=True)
class GraphNodeRef:
    """Stable reference to a node in the projection.

    Two refs compare equal iff they point at the same canonical record
    in the same env. Backends translate this to whatever internal
    addressing they need (``graph_nodes.id`` for Postgres,
    ``MATCH (n {env_id, id})`` for Neo4j).
    """

    env_id: UUID
    kind: NodeKind
    record_id: UUID


@dataclass(frozen=True)
class GraphEdge:
    """A typed, directed edge between two ``GraphNodeRef``s.

    ``edge_type`` is a server-side allowlist value (see
    ``p2.1-cypher-safety``) — never caller-supplied literal text.
    ``properties`` is a free-form key-value map; backends serialize as
    appropriate (Neo4j relation properties, JSONB column for Postgres).

    **Same-env invariant**: ``env_id``, ``src.env_id``, and
    ``dst.env_id`` MUST be equal. The schema enforces this on
    canonical Postgres; we mirror the constraint here so the graph
    backends never silently drift across envs.
    """

    env_id: UUID
    src: GraphNodeRef
    dst: GraphNodeRef
    edge_type: str
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.src.env_id != self.env_id or self.dst.env_id != self.env_id:
            raise ValueError(
                f"GraphEdge env mismatch: edge.env_id={self.env_id}, "
                f"src.env_id={self.src.env_id}, dst.env_id={self.dst.env_id}"
            )


@dataclass(frozen=True)
class GraphPathStep:
    """One real-edge step along a path returned by :meth:`GraphStore.neighbors`.

    ``src`` and ``dst`` carry the **actual relation orientation**, NOT
    the traversal direction. For ``direction="in"`` walks the path
    sequence reads from terminal back to the start node in real-edge
    terms; clients can chain steps by matching nodes pairwise. Both
    backends (Neo4j ``startNode``/``endNode``; Postgres recursive CTE
    storing ``e.src_node_id`` / ``e.dst_node_id``) honor this contract.

    Property bags are typically empty or very small — full canonical
    relation properties live in Postgres.
    """

    src: GraphNodeRef
    dst: GraphNodeRef
    edge_type: str
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeighborHit:
    """A single neighbor returned by :meth:`GraphStore.neighbors`.

    ``path`` materializes the full edge sequence from the source node
    to ``node``; its length equals ``path_length``. For 1-hop queries
    this is exactly one step (``src == query_node``, ``dst == node``).

    ``score`` is backend-defined and may be ``None`` (Postgres CTE has
    no score; Neo4j may attach pattern-match scoring). Sort uses
    ``(path_length ASC, score DESC NULLS LAST, kind ASC, record_id ASC)``.
    """

    node: GraphNodeRef
    path_length: int
    path: tuple[GraphPathStep, ...]
    score: float | None = None


@runtime_checkable
class GraphStore(Protocol):
    """Backend-agnostic graph projection surface."""

    async def upsert_node(
        self,
        node: GraphNodeRef,
        *,
        attrs: Mapping[str, Any],
    ) -> None:
        """Idempotently MERGE the node, replacing its attribute map.

        ``attrs`` is the projected attribute set (e.g.
        ``{"name": ..., "normalized_name": ..., "kind_tag": ...}`` for
        entities; small for memory nodes). Callers should keep this
        set small — it is replicated; the canonical truth lives in
        Postgres.

        Reserved keys (``env_id``, ``kind``, ``record_id``, ``id``)
        MUST NOT appear in ``attrs`` — implementations raise
        ``ValueError`` if they do. Identity is fixed by the
        :class:`GraphNodeRef` argument.
        """

    async def upsert_edge(
        self,
        edge: GraphEdge,
    ) -> None:
        """Idempotently MERGE the edge.

        Both endpoints are MERGEd as stub nodes (identity-only attrs)
        before the edge is created — implementations must not assume
        endpoints already exist. This supports lazy memory-node
        projection: a relation referencing a memory id can be
        delivered without a prior ``upsert_node`` for that memory.

        A subsequent ``upsert_node`` call on the same node is allowed
        and overwrites its attribute map; the stub created here
        survives until that happens.
        """

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
        """Return up to ``limit`` neighbors within ``hops`` of ``node``.

        Traversal does not cross ``env_id`` — only the source node's
        env is searched. ``direction`` controls edge orientation:

        * ``out`` — follow ``(node)-[r]->(?)``
        * ``in``  — follow ``(?)-[r]->(node)``
        * ``both`` — both (default)

        ``edge_types``, when provided, restricts to relations of those
        types. ``kinds`` filters the **terminal** (returned) node kind,
        not intermediate path nodes — so an entity-to-entity-to-memory
        path is reachable when ``kinds=["memory"]``.

        Results are deterministically ordered as
        ``(path_length ASC, score DESC NULLS LAST,
        node.kind ASC, node.record_id ASC)``.

        Returns ``(hits, next_cursor)``. ``next_cursor`` is ``None``
        when the page exhausted the result set. Cursors are bound to
        the calling query shape; passing a cursor produced by a
        different ``hops``/``direction``/``edge_types``/``kinds``/
        ``limit`` SHOULD raise ``ValueError``.
        """

    async def delete_subgraph(
        self,
        *,
        env_id: UUID,
        nodes: Sequence[GraphNodeRef],
    ) -> None:
        """Idempotently delete nodes + all incident edges in one env.

        Used by entity-merge cleanup, memory hard-delete, and the
        admin-rebuild path. Deleting an absent node is a no-op.

        **Invariant**: every ``node.env_id`` MUST equal ``env_id`` —
        implementations raise ``ValueError`` on mismatch. Cross-env
        deletion is not a v1 capability.

        Note: ``delete_subgraph`` is the *raw* deletion primitive. For
        entity merges that re-point edges from a merged entity onto
        a kept entity, the orchestration layer (``ent_merge``) is
        responsible for emitting relation-update events to the outbox
        BEFORE the entity-tombstone event. The graph backend treats
        each event individually.
        """

    async def close(self) -> None:
        """Release any held resources (driver pools, sessions, …)."""


__all__ = [
    "RESERVED_ATTRS",
    "GraphEdge",
    "GraphNodeRef",
    "GraphPathStep",
    "GraphStore",
    "NeighborHit",
    "NodeKind",
    "TraversalDirection",
]
