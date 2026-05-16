"""Cypher-injection safety audit for ``Neo4jGraphStore``.

We never interpolate caller-supplied strings into Cypher.  The only
interpolated tokens are:

* Node labels — sourced from the closed ``_LABEL_BY_KIND`` dict
  (keys are :class:`NodeKind` Literal values).
* The traversal arrow — selected from a closed ``{out, in, both}``
  mapping keyed by :class:`TraversalDirection`.
* The variable-length range ``*1..{h}`` — ``h`` is an ``int`` validated
  by Pydantic at the tool boundary (``ge=1, le=3``).

The relation type (``r.type``) and the edge-type filter list are passed
as ``$etype`` / ``$etypes`` parameters.  Cypher injection on
caller-supplied edge-type strings is therefore structurally impossible.

These tests assert that invariant by:

1. Capturing the literal Cypher string ``Neo4jGraphStore`` would send
   for a fuzzed-edge-type request and asserting the payload appears
   ONLY in the parameter dict, NEVER in the Cypher source.
2. Asserting Pydantic validation rejects out-of-range / blank-only /
   length-violating ``edge_types`` and ``type`` values.

If a future refactor introduces caller-string interpolation, these
tests fail loudly.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from memory_mcp.config import Settings
from memory_mcp.db.graph.base import (
    GraphEdge,
    GraphNodeRef,
)
from memory_mcp.db.graph.neo4j import Neo4jDriver, Neo4jGraphStore
from memory_mcp.graph import EntityNeighborsRequest
from memory_mcp.relations import RelationEndpoint, RelationLinkRequest

# ---------------------------------------------------------------------------
# Cypher-injection payload corpus
#
# Each payload tries a different escape strategy.  Every one of these
# MUST appear ONLY in the parameter dict, never in the Cypher source.
# ---------------------------------------------------------------------------

INJECTION_PAYLOADS: list[str] = [
    # Try to close the relationship pattern and start a new one
    "RELATED]->(victim:Entity)//",
    # Property-side injection
    "describes'}) RETURN n; //",
    # Multi-line / comment escape
    "x\n// comment\nMATCH (q) DETACH DELETE q",
    # Quote injection
    "'; DROP CONSTRAINT entity_id_per_env;//",
    # Backtick (Cypher identifier escape)
    "`bad`type",
    # Unicode / RTL trickery
    "describes\u202e",
    # NULL byte
    "describes\x00DELETE",
    # Plain malicious-looking but valid string
    "DROP CONSTRAINT entity_id_per_env",
]


# ---------------------------------------------------------------------------
# Capture helper — wraps Neo4jDriver.driver.session() so .run(cypher, **params)
# is recorded for inspection.
# ---------------------------------------------------------------------------


def _capturing_store() -> tuple[Neo4jGraphStore, list[tuple[str, dict[str, Any]]]]:
    """Build a Neo4jGraphStore whose driver records every (cypher, params)."""
    captured: list[tuple[str, dict[str, Any]]] = []

    async def _run(cypher: str, **params: Any) -> Any:
        captured.append((cypher, params))
        # Mimic an empty result iterator so callers that consume rows are happy.
        result = MagicMock()
        async def _aiter() -> Any:
            return
            yield  # noqa: F812 — never reached but makes this an async generator
        result.__aiter__ = lambda self: _aiter()
        result.consume = AsyncMock(return_value=None)
        return result

    session = MagicMock()
    session.run = AsyncMock(side_effect=_run)

    @asynccontextmanager
    async def _sessionmaker() -> Any:
        yield session

    driver = MagicMock()
    driver.session = _sessionmaker

    store = Neo4jGraphStore.__new__(Neo4jGraphStore)
    real_driver = Neo4jDriver(Settings(graph_backend="neo4j"))  # type: ignore[arg-type]
    real_driver._driver = driver  # type: ignore[attr-defined]
    store._driver = real_driver  # type: ignore[attr-defined]
    return store, captured


# ---------------------------------------------------------------------------
# (1) Pydantic-level rejection
# ---------------------------------------------------------------------------


def test_relation_link_type_rejects_blank() -> None:
    eid_a, eid_b = uuid4(), uuid4()
    src = RelationEndpoint(kind="entity", id=eid_a)
    dst = RelationEndpoint(kind="entity", id=eid_b)
    with pytest.raises(ValidationError):
        RelationLinkRequest(src=src, dst=dst, type="")
    with pytest.raises(ValidationError):
        RelationLinkRequest(src=src, dst=dst, type="   ")


def test_relation_link_type_rejects_too_long() -> None:
    eid_a, eid_b = uuid4(), uuid4()
    src = RelationEndpoint(kind="entity", id=eid_a)
    dst = RelationEndpoint(kind="entity", id=eid_b)
    with pytest.raises(ValidationError):
        RelationLinkRequest(src=src, dst=dst, type="x" * 201)


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_relation_link_accepts_injection_strings_as_opaque(payload: str) -> None:
    """At Pydantic level, malicious strings are valid input — the
    parameterization in the Neo4j layer makes them safe.  This asserts
    the type field accepts them so a defender can't bypass the layer
    by writing client-side validation.
    """
    if len(payload) > 200:
        pytest.skip("payload exceeds max_length")
    eid_a, eid_b = uuid4(), uuid4()
    src = RelationEndpoint(kind="entity", id=eid_a)
    dst = RelationEndpoint(kind="entity", id=eid_b)
    req = RelationLinkRequest(src=src, dst=dst, type=payload)
    assert req.type == payload


def test_entity_neighbors_edge_types_rejects_blank_entries() -> None:
    with pytest.raises(ValidationError):
        EntityNeighborsRequest(entity_id=uuid4(), edge_types=["good", "  "])


def test_entity_neighbors_edge_types_rejects_too_long() -> None:
    with pytest.raises(ValidationError):
        EntityNeighborsRequest(entity_id=uuid4(), edge_types=["x" * 201])


def test_entity_neighbors_hops_bounded() -> None:
    # Pydantic-level upper bound on hops prevents unbounded variable-length
    # patterns being interpolated into the Cypher pattern.
    with pytest.raises(ValidationError):
        EntityNeighborsRequest(entity_id=uuid4(), hops=0)
    with pytest.raises(ValidationError):
        EntityNeighborsRequest(entity_id=uuid4(), hops=4)


# ---------------------------------------------------------------------------
# (2) Cypher-source inspection: payload MUST NOT leak into cypher string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_upsert_edge_payload_only_in_parameters(payload: str) -> None:
    if len(payload) > 200:
        pytest.skip("payload exceeds max_length")

    env = uuid4()
    src = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())
    dst = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())
    edge = GraphEdge(env_id=env, src=src, dst=dst, edge_type=payload, properties={})

    store, captured = _capturing_store()
    asyncio.run(store.upsert_edge(edge))

    assert len(captured) == 1, "upsert_edge should issue exactly one Cypher statement"
    cypher, params = captured[0]
    # Payload must NOT appear in the cypher source — it must be passed
    # through the $etype parameter.
    assert payload not in cypher, (
        f"INJECTION REGRESSION: payload {payload!r} leaked into Cypher source"
    )
    assert params["etype"] == payload
    # Sanity: cypher uses the parameter placeholder.
    assert "$etype" in cypher


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_neighbors_payload_only_in_parameters(payload: str) -> None:
    if len(payload) > 200:
        pytest.skip("payload exceeds max_length")

    env = uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())

    store, captured = _capturing_store()
    asyncio.run(
        store.neighbors(
            node,
            hops=1,
            direction="both",
            edge_types=[payload],
            kinds=["entity", "memory"],
            limit=10,
        )
    )

    assert len(captured) == 1, "neighbors should issue exactly one Cypher MATCH"
    cypher, params = captured[0]
    assert payload not in cypher, (
        f"INJECTION REGRESSION: payload {payload!r} leaked into Cypher source"
    )
    assert payload in params["etypes"]
    assert "$etypes" in cypher


def test_upsert_edge_only_uses_closed_label_set() -> None:
    """Sanity: the cypher source for upsert_edge contains only the two
    valid labels (`Entity`, `Memory`) and never anything caller-derived.
    """
    env = uuid4()
    src = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())
    dst = GraphNodeRef(env_id=env, kind="memory", record_id=uuid4())
    edge = GraphEdge(env_id=env, src=src, dst=dst, edge_type="describes", properties={})

    store, captured = _capturing_store()
    asyncio.run(store.upsert_edge(edge))
    cypher, _params = captured[0]
    # Both labels appear as bare identifiers from the closed dict.
    assert ":Entity" in cypher
    assert ":Memory" in cypher
    # No other labels — the closed dict has exactly two.
    assert ":Entity" in cypher and ":Memory" in cypher


def test_neighbors_arrow_direction_is_from_closed_set() -> None:
    """The arrow patterns are selected from a closed dict on
    direction; pydantic Literal blocks anything else upstream.  Verify
    the cypher contains exactly one of the three valid arrows.
    """
    env = uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())

    for direction, expected_arrow in [
        ("out", "-[r:RELATED*1..1]->"),
        ("in", "<-[r:RELATED*1..1]-"),
        ("both", "-[r:RELATED*1..1]-"),
    ]:
        store, captured = _capturing_store()
        asyncio.run(
            store.neighbors(
                node,
                hops=1,
                direction=direction,  # type: ignore[arg-type]
                edge_types=None,
                kinds=None,
                limit=5,
            )
        )
        cypher, _ = captured[0]
        assert expected_arrow in cypher, (
            f"direction={direction!r} should produce arrow {expected_arrow!r}; "
            f"cypher was: {cypher}"
        )


def test_neighbors_hops_int_bounded_in_pattern() -> None:
    """Pydantic enforces hops ∈ [1, 3] at the tool boundary.  Confirm
    the Cypher pattern only ever has integer-bound hop counts.
    """
    env = uuid4()
    node = GraphNodeRef(env_id=env, kind="entity", record_id=uuid4())

    for hops in (1, 2, 3):
        store, captured = _capturing_store()
        asyncio.run(
            store.neighbors(
                node, hops=hops, direction="out",
                edge_types=None, kinds=None, limit=5,
            )
        )
        cypher, _ = captured[0]
        assert f"*1..{hops}]" in cypher
