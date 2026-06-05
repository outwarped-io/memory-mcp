"""Graph backends.

Two implementations behind a common :class:`GraphStore` Protocol:

* :class:`memory_mcp.db.graph.neo4j.Neo4jGraphStore` — default, full
  Cypher graph traversal
* :class:`memory_mcp.db.graph.postgres.PostgresGraphStore` —
  recursive-CTE fallback over the canonical ``relations`` table for
  small deployments (no Neo4j container required)

Selected at runtime by ``Settings.graph_backend`` via :func:`get_graph_store`.
The Protocol contract lives in :mod:`memory_mcp.db.graph.base`.
"""

from memory_mcp.db.graph.base import (
    RESERVED_ATTRS,
    GraphEdge,
    GraphNodeRef,
    GraphPathStep,
    GraphStore,
    NeighborHit,
    NodeKind,
    TraversalDirection,
)

__all__ = [
    "RESERVED_ATTRS",
    "GraphEdge",
    "GraphNodeRef",
    "GraphPathStep",
    "GraphStore",
    "NeighborHit",
    "NodeKind",
    "TraversalDirection",
    "get_graph_store",
]


def get_graph_store(settings: "Settings") -> GraphStore:  # noqa: F821
    """Construct the configured :class:`GraphStore` for ``settings``.

    Late-imported impls so the Postgres fallback never imports the
    ``neo4j`` driver and vice-versa — keeps cold-start lighter for
    deployments that pin one backend.
    """
    from memory_mcp.config import Settings  # noqa: F401  (forward ref)

    if settings.graph_backend == "neo4j":
        from memory_mcp.db.graph.neo4j import Neo4jGraphStore

        return Neo4jGraphStore(settings)
    if settings.graph_backend == "postgres":
        from memory_mcp.db.graph.postgres import PostgresGraphStore

        return PostgresGraphStore(settings)
    raise ValueError(f"unsupported graph_backend={settings.graph_backend!r} (expected 'neo4j' or 'postgres')")
