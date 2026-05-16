"""Neo4j async driver wrapper + ``GraphStore`` implementation.

Phase 2.1 — owns the connection pool + schema-init helper, and implements
:class:`memory_mcp.db.graph.base.GraphStore` for Neo4j.

The driver is held on the class instance, not as a module-level global,
so server, projection-worker, and tests can construct independent
clients.  Lifetimes are bound to the lifespan of whichever process owns
them — close with :meth:`close`.

The schema-init helper is **idempotent** — calling it on an already-set-
up database is a no-op aside from CREATE CONSTRAINT IF NOT EXISTS noise
in Neo4j logs.  It is invoked from the projection-worker startup so the
first time the worker drains an entity / relation event, the constraints
are already in place.

Edge representation
-------------------

All edges in the Memory MCP graph projection use a single Neo4j
relationship type ``:RELATED`` with a property ``type`` that carries
the canonical edge type (``describes``, ``mentioned_by``, …). This
sidesteps Cypher's "relationship type must be a literal" rule and
removes the entire string-interpolation attack surface — every
operation is fully parameterized.

The cost is that ``MATCH (n)-[r:RELATED]->(m) WHERE r.type IN $types``
is slightly slower than typed-relationship MATCH on a large graph; we
add an index on ``r.type`` to compensate. Phase 2.1 is well within
limits where this matters.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from neo4j import AsyncDriver, AsyncGraphDatabase

from memory_mcp.config import Settings
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

# Idempotent constraints + indexes for the Memory MCP graph projection.
#
# We use ``IF NOT EXISTS`` everywhere so the schema-init can be called on
# every process startup without raising. Each (label, property) pair the
# projection writes corresponds to a constraint here.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    # Entity nodes — keyed by (env_id, id). env_id is part of the key so
    # the same entity-id in two envs is a separate node, preserving the
    # env-scoping invariant.
    "CREATE CONSTRAINT entity_id_per_env IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.env_id, e.id) IS UNIQUE",
    # Memory nodes — projected lazily when a relation references one.
    "CREATE CONSTRAINT memory_id_per_env IF NOT EXISTS "
    "FOR (m:Memory) REQUIRE (m.env_id, m.id) IS UNIQUE",
    # Task nodes — projected directly from task aggregate events.
    "CREATE CONSTRAINT task_id_per_env IF NOT EXISTS "
    "FOR (t:Task) REQUIRE (t.env_id, t.id) IS UNIQUE",
    # Helpful read indexes
    "CREATE INDEX entity_kind IF NOT EXISTS FOR (e:Entity) ON (e.kind)",
    "CREATE INDEX entity_normalized_name IF NOT EXISTS "
    "FOR (e:Entity) ON (e.normalized_name)",
    "CREATE INDEX task_status IF NOT EXISTS FOR (t:Task) ON (t.status)",
    # Relationship-property index for typed traversal — see edge
    # representation note in the module docstring.
    "CREATE INDEX related_type IF NOT EXISTS FOR ()-[r:RELATED]-() ON (r.type)",
)

# Map from `NodeKind` to Cypher node label.  Both labels are listed as
# constants in `_SCHEMA_STATEMENTS` so callers/tests can verify parity.
_LABEL_BY_KIND: dict[NodeKind, str] = {"entity": "Entity", "memory": "Memory", "task": "Task"}


class Neo4jDriver:
    """Owns a single :class:`neo4j.AsyncDriver` instance + lifecycle.

    Cheap to construct, expensive to close — the driver maintains its
    own connection pool internally. Each long-lived process should hold
    one instance for the duration of its lifespan.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._driver: AsyncDriver | None = None

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_url,
                auth=(self._settings.neo4j_user, self._settings.neo4j_password),
            )
        return self._driver

    async def verify(self, *, timeout: float = 2.0) -> None:  # noqa: ASYNC109 — pass-through to asyncio.wait_for
        """Ping the server. Raises on failure or timeout.

        Wrapped in :func:`asyncio.wait_for` because :meth:`AsyncDriver.
        verify_connectivity` does not honor a per-call timeout in older
        driver versions.
        """
        await asyncio.wait_for(self.driver.verify_connectivity(), timeout=timeout)

    async def init_schema(self) -> None:
        """Run idempotent CREATE CONSTRAINT / CREATE INDEX statements.

        Safe to call from multiple processes concurrently — Neo4j
        serializes constraint creation server-side.
        """
        async with self.driver.session() as session:
            for stmt in _SCHEMA_STATEMENTS:
                result = await session.run(stmt)
                await result.consume()

    async def close(self) -> None:
        if self._driver is not None:
            try:
                await self._driver.close()
            finally:
                self._driver = None


async def probe_neo4j(settings: Settings, *, timeout: float = 2.0) -> dict[str, Any]:  # noqa: ASYNC109 — pass-through to verify/wait_for
    """Best-effort liveness probe for ``/readyz``.

    Returns ``{"status": "skipped"}`` when ``GRAPH_BACKEND != "neo4j"``;
    Postgres-graph deployments do not need a Neo4j server. Otherwise
    constructs a one-shot driver, calls :meth:`verify`, and reports the
    outcome with a short error string on failure.
    """
    if settings.graph_backend != "neo4j":
        return {
            "status": "skipped",
            "reason": f"graph_backend={settings.graph_backend!r}",
        }

    drv = Neo4jDriver(settings)
    try:
        await drv.verify(timeout=timeout)
        return {"status": "ok"}
    except TimeoutError:
        return {"status": "error", "error": f"probe timed out after {timeout}s"}
    except Exception as exc:  # noqa: BLE001 — probe is best-effort
        return {"status": "error", "error": str(exc)[:200]}
    finally:
        await drv.close()


# ---------------------------------------------------------------------------
# Cursor encoding
# ---------------------------------------------------------------------------

# Cursors are base64-encoded JSON of a small dict capturing query shape +
# the SKIP offset of the next page.  Because they're opaque to callers,
# encoding details may change in v1.5; the contract is just "round-trips
# through ``GraphStore.neighbors``".


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
# Neo4jGraphStore
# ---------------------------------------------------------------------------


class Neo4jGraphStore:
    """:class:`GraphStore` implementation backed by Neo4j.

    Wraps a :class:`Neo4jDriver` and provides MERGE-on-write idempotent
    upserts, recursive variable-length traversal, and the
    :func:`asyncio.wait_for`-bounded probe.

    Constructed once per process; share across requests.  The underlying
    driver maintains its own session pool — methods open and dispose
    short-lived sessions per call.
    """

    def __init__(self, settings: Settings) -> None:
        self._driver = Neo4jDriver(settings)

    @property
    def driver(self) -> Neo4jDriver:
        return self._driver

    async def init_schema(self) -> None:
        await self._driver.init_schema()

    async def close(self) -> None:
        await self._driver.close()

    # -- Mutations --------------------------------------------------------

    async def upsert_node(
        self,
        node: GraphNodeRef,
        *,
        attrs: Mapping[str, Any],
    ) -> None:
        bad = RESERVED_ATTRS & attrs.keys()
        if bad:
            raise ValueError(
                f"upsert_node: attrs must not contain reserved keys {sorted(bad)}; "
                "identity is fixed by GraphNodeRef"
            )
        label = _LABEL_BY_KIND[node.kind]
        # Cypher cannot parameterize labels — but ``label`` is selected
        # from a closed Literal-typed dict, never from caller input.
        cypher = (
            f"MERGE (n:{label} {{env_id: $env_id, id: $rid}}) "
            "SET n += $attrs "
            "RETURN n"
        )
        async with self._driver.driver.session() as session:
            result = await session.run(
                cypher,
                env_id=str(node.env_id),
                rid=str(node.record_id),
                attrs=dict(attrs),
            )
            # ``consume()`` forces server-side execution and surfaces any
            # errors. Without it, errors are silently swallowed when the
            # session context manager exits.
            await result.consume()

    async def upsert_edge(self, edge: GraphEdge) -> None:
        src_label = _LABEL_BY_KIND[edge.src.kind]
        dst_label = _LABEL_BY_KIND[edge.dst.kind]
        # Edge properties are stored as a JSON-encoded string because
        # Neo4j relationships only accept primitive scalar/array values
        # (a map property like ``r.properties = {}`` is rejected at
        # write time). The canonical truth lives in Postgres ``relations.
        # properties``; the graph projection just needs a queryable
        # round-trip.
        cypher = (
            f"MERGE (s:{src_label} {{env_id: $env_id, id: $src_id}}) "
            f"MERGE (d:{dst_label} {{env_id: $env_id, id: $dst_id}}) "
            "MERGE (s)-[r:RELATED {type: $etype}]->(d) "
            "SET r.properties_json = $props_json "
            "RETURN r"
        )
        async with self._driver.driver.session() as session:
            result = await session.run(
                cypher,
                env_id=str(edge.env_id),
                src_id=str(edge.src.record_id),
                dst_id=str(edge.dst.record_id),
                etype=edge.edge_type,
                props_json=json.dumps(dict(edge.properties), sort_keys=True),
            )
            await result.consume()

    async def delete_subgraph(
        self,
        *,
        env_id: UUID,
        nodes: Sequence[GraphNodeRef],
    ) -> None:
        for n in nodes:
            if n.env_id != env_id:
                raise ValueError(
                    f"delete_subgraph: node env_id {n.env_id} != {env_id}; "
                    "cross-env deletion is not supported"
                )
        if not nodes:
            return
        # Group by kind so we can issue one MATCH ... DETACH DELETE per
        # label.  Labels come from the closed `_LABEL_BY_KIND` mapping.
        by_kind: dict[NodeKind, list[str]] = {"entity": [], "memory": []}
        for n in nodes:
            by_kind[n.kind].append(str(n.record_id))
        async with self._driver.driver.session() as session:
            for kind, ids in by_kind.items():
                if not ids:
                    continue
                label = _LABEL_BY_KIND[kind]
                result = await session.run(
                    f"MATCH (n:{label} {{env_id: $env_id}}) "
                    "WHERE n.id IN $ids "
                    "DETACH DELETE n",
                    env_id=str(env_id),
                    ids=ids,
                )
                await result.consume()

    # -- Reads ------------------------------------------------------------

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
            node=node, hops=hops, direction=direction,
            edge_types=edge_types, kinds=kinds, limit=limit,
        )

        offset = 0
        if cursor is not None:
            decoded = _decode_cursor(cursor)
            if decoded.get("shape") != shape:
                raise ValueError(
                    "cursor query-shape mismatch — cursors are not portable across calls"
                )
            offset = int(decoded.get("offset", 0))

        # Direction is rendered as the relationship arrow in the MATCH
        # clause.  All three forms use the same `:RELATED` label so the
        # caller-facing edge_type filter happens via WHERE.
        arrow = {
            "out": "-[r:RELATED*1..{h}]->",
            "in": "<-[r:RELATED*1..{h}]-",
            "both": "-[r:RELATED*1..{h}]-",
        }[direction].format(h=hops)

        src_label = _LABEL_BY_KIND[node.kind]
        # We accept any terminal label and filter post-MATCH so the
        # variable-length pattern stays simple.  `kinds` is then applied
        # via WHERE on the terminal label name.
        #
        # NOTE: Neo4j 5 requires ``length()`` to be called on a Path, not
        # on a list of relationships.  We bind the path to ``p`` and
        # use ``size(r)`` for the relationship count (equal to ``length(p)``
        # for our variable-length pattern).
        cypher = (
            f"MATCH p = (s:{src_label} {{env_id: $env_id, id: $rid}})"
            f"{arrow}(t) "
            "WHERE labels(t)[0] IN $term_labels "
            "AND ALL(rel IN relationships(p) "
            "        WHERE $allow_all_types OR rel.type IN $etypes) "
            "WITH t, relationships(p) AS r, length(p) AS plen "
            "RETURN labels(t)[0] AS term_label, t.id AS term_id, "
            "       plen, "
            "       [rel IN r | rel.type] AS rel_types, "
            "       [rel IN r | startNode(rel).id] AS src_ids, "
            "       [rel IN r | endNode(rel).id] AS dst_ids, "
            "       [rel IN r | labels(startNode(rel))[0]] AS src_labels, "
            "       [rel IN r | labels(endNode(rel))[0]] AS dst_labels "
            "ORDER BY plen ASC, term_label ASC, term_id ASC "
            "SKIP $skip LIMIT $lim"
        )

        term_labels = (
            [_LABEL_BY_KIND[k] for k in kinds]
            if kinds
            else list(_LABEL_BY_KIND.values())
        )
        params = {
            "env_id": str(node.env_id),
            "rid": str(node.record_id),
            "term_labels": term_labels,
            "allow_all_types": edge_types is None,
            "etypes": list(edge_types) if edge_types else [],
            "skip": offset,
            # Fetch limit+1 to detect a next page.
            "lim": limit + 1,
        }

        async with self._driver.driver.session() as session:
            result = await session.run(cypher, **params)
            records = [dict(r) async for r in result]

        has_more = len(records) > limit
        records = records[:limit]

        hits: list[NeighborHit] = []
        for rec in records:
            term_kind: NodeKind = (
                "entity" if rec["term_label"] == "Entity" else "memory"
            )
            terminal = GraphNodeRef(
                env_id=node.env_id,
                kind=term_kind,
                record_id=UUID(rec["term_id"]),
            )
            steps: list[GraphPathStep] = []
            for i, etype in enumerate(rec["rel_types"]):
                src_kind: NodeKind = (
                    "entity" if rec["src_labels"][i] == "Entity" else "memory"
                )
                dst_kind: NodeKind = (
                    "entity" if rec["dst_labels"][i] == "Entity" else "memory"
                )
                steps.append(
                    GraphPathStep(
                        src=GraphNodeRef(
                            env_id=node.env_id,
                            kind=src_kind,
                            record_id=UUID(rec["src_ids"][i]),
                        ),
                        dst=GraphNodeRef(
                            env_id=node.env_id,
                            kind=dst_kind,
                            record_id=UUID(rec["dst_ids"][i]),
                        ),
                        edge_type=etype,
                    )
                )
            hits.append(
                NeighborHit(
                    node=terminal,
                    path_length=int(rec["plen"]),
                    path=tuple(steps),
                )
            )

        next_cursor = None
        if has_more:
            next_cursor = _encode_cursor({"shape": shape, "offset": offset + limit})
        return hits, next_cursor


__all__ = [
    "Neo4jDriver",
    "Neo4jGraphStore",
    "probe_neo4j",
]
