"""Unit tests for ``memory_mcp.graph`` — schema, projection helpers, singleton.

The DB-bound parts of :func:`entity_neighbors` are exercised by the
integration suite (Phase 2.1 ``p2.1-tests``). Here we cover:

* :class:`EntityNeighborsRequest` schema validation, including alias
  acceptance (``id``/``entity_id``, ``types``/``edge_types``).
* :func:`_collect_record_ids` bucketing entity / memory ids correctly
  across terminal + path nodes.
* :func:`_project_hits` projection rules:
  - lifecycle filter on terminals
  - lifecycle filter on path-transit memories
  - self-as-neighbor suppression
  - missing canonical row → skipped
  - real-edge orientation preserved in path steps
* Singleton (`_get_default_graph_store`, `_close_default_graph_store`)
  with a stubbed factory.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from memory_mcp.db.graph.base import GraphNodeRef, GraphPathStep, NeighborHit
from memory_mcp.graph import (
    EntityNeighborsRequest,
    NeighborHitResponse,
    _close_default_graph_store,
    _collect_record_ids,
    _get_default_graph_store,
    _project_hits,
    _reset_default_graph_store_for_tests,
)

# ---------------------------------------------------------------------------
# EntityNeighborsRequest schema
# ---------------------------------------------------------------------------


class TestEntityNeighborsRequestSchema:
    def test_minimal_request(self) -> None:
        eid = uuid4()
        req = EntityNeighborsRequest(entity_id=eid)
        assert req.entity_id == eid
        assert req.hops == 1
        assert req.kind == "both"
        assert req.direction == "both"
        assert req.limit == 20
        assert req.cursor is None
        assert req.edge_types is None

    def test_id_alias_accepted(self) -> None:
        """Plan's public name ``id`` should resolve to ``entity_id``."""
        eid = uuid4()
        req = EntityNeighborsRequest.model_validate({"id": str(eid)})
        assert req.entity_id == eid

    def test_types_alias_accepted(self) -> None:
        eid = uuid4()
        req = EntityNeighborsRequest.model_validate({"entity_id": str(eid), "types": ["mentions", "describes"]})
        assert req.edge_types == ["mentions", "describes"]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), bogus=1)  # type: ignore[call-arg]

    def test_hops_below_minimum(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), hops=0)

    def test_hops_above_maximum(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), hops=4)

    def test_limit_above_maximum(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), limit=101)

    def test_edge_types_blank_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), edge_types=["   "])

    def test_edge_types_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), edge_types=["x" * 201])

    def test_edge_types_max_count_enforced(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), edge_types=["t"] * 21)

    def test_edge_types_strips_whitespace(self) -> None:
        req = EntityNeighborsRequest(entity_id=uuid4(), edge_types=["  mentions  "])
        assert req.edge_types == ["mentions"]

    def test_kind_must_be_known(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), kind="other")  # type: ignore[arg-type]

    def test_direction_must_be_known(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), direction="sideways")  # type: ignore[arg-type]

    def test_cursor_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityNeighborsRequest(entity_id=uuid4(), cursor="x" * 4097)


# ---------------------------------------------------------------------------
# _collect_record_ids
# ---------------------------------------------------------------------------


def _node(env_id: UUID, kind: str, record_id: UUID) -> GraphNodeRef:
    return GraphNodeRef(env_id=env_id, kind=kind, record_id=record_id)  # type: ignore[arg-type]


def test_collect_record_ids_buckets_by_kind() -> None:
    env_id = uuid4()
    e1, e2, m1 = uuid4(), uuid4(), uuid4()
    hit = NeighborHit(
        node=_node(env_id, "memory", m1),
        path_length=2,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", e1),
                dst=_node(env_id, "entity", e2),
                edge_type="rel_a",
            ),
            GraphPathStep(
                src=_node(env_id, "entity", e2),
                dst=_node(env_id, "memory", m1),
                edge_type="rel_b",
            ),
        ),
    )
    e_ids, m_ids = _collect_record_ids([hit])
    assert e_ids == {e1, e2}
    assert m_ids == {m1}


def test_collect_record_ids_handles_empty() -> None:
    e_ids, m_ids = _collect_record_ids([])
    assert e_ids == set()
    assert m_ids == set()


# ---------------------------------------------------------------------------
# _project_hits — lifecycle, self-cycle, real-edge orientation
# ---------------------------------------------------------------------------


def _one_hop_entity_hit(env_id: UUID, src: UUID, dst: UUID, edge_type: str = "rel") -> NeighborHit:
    return NeighborHit(
        node=_node(env_id, "entity", dst),
        path_length=1,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", src),
                dst=_node(env_id, "entity", dst),
                edge_type=edge_type,
            ),
        ),
    )


def test_project_hits_resolves_entity_terminal_name() -> None:
    env_id, start, neighbor = uuid4(), uuid4(), uuid4()
    hit = _one_hop_entity_hit(env_id, start, neighbor)
    out = _project_hits(
        hits=[hit],
        start_entity_id=start,
        env_id=env_id,
        entities_by_id={start: "Start", neighbor: "Neighbor"},
        memories_by_id={},
    )
    assert len(out) == 1
    assert out[0].node.name == "Neighbor"
    assert out[0].node.id == neighbor
    assert out[0].path_length == 1


def test_project_hits_drops_self_cycle_back_to_start() -> None:
    """A 2-hop walk returning to the start entity must be filtered."""
    env_id, start, mid = uuid4(), uuid4(), uuid4()
    hit = NeighborHit(
        node=_node(env_id, "entity", start),  # back to start
        path_length=2,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", start),
                dst=_node(env_id, "entity", mid),
                edge_type="x",
            ),
            GraphPathStep(
                src=_node(env_id, "entity", mid),
                dst=_node(env_id, "entity", start),
                edge_type="y",
            ),
        ),
    )
    out = _project_hits(
        hits=[hit],
        start_entity_id=start,
        env_id=env_id,
        entities_by_id={start: "S", mid: "M"},
        memories_by_id={},
    )
    assert out == []


def test_project_hits_drops_terminal_retired_memory() -> None:
    env_id, start, mem = uuid4(), uuid4(), uuid4()
    hit = NeighborHit(
        node=_node(env_id, "memory", mem),
        path_length=1,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", start),
                dst=_node(env_id, "memory", mem),
                edge_type="describes",
            ),
        ),
    )
    out = _project_hits(
        hits=[hit],
        start_entity_id=start,
        env_id=env_id,
        entities_by_id={start: "S"},
        memories_by_id={mem: ("retired memo", "retired")},
    )
    assert out == []


def test_project_hits_drops_path_transit_archived_memory() -> None:
    """An archived memory in path transit should hide the entire hit."""
    env_id, start, mem, terminal = uuid4(), uuid4(), uuid4(), uuid4()
    hit = NeighborHit(
        node=_node(env_id, "entity", terminal),
        path_length=2,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", start),
                dst=_node(env_id, "memory", mem),
                edge_type="a",
            ),
            GraphPathStep(
                src=_node(env_id, "memory", mem),
                dst=_node(env_id, "entity", terminal),
                edge_type="b",
            ),
        ),
    )
    out = _project_hits(
        hits=[hit],
        start_entity_id=start,
        env_id=env_id,
        entities_by_id={start: "S", terminal: "T"},
        memories_by_id={mem: ("memo", "archived")},
    )
    assert out == []


def test_project_hits_keeps_active_memory_terminal() -> None:
    env_id, start, mem = uuid4(), uuid4(), uuid4()
    hit = NeighborHit(
        node=_node(env_id, "memory", mem),
        path_length=1,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", start),
                dst=_node(env_id, "memory", mem),
                edge_type="describes",
            ),
        ),
    )
    out = _project_hits(
        hits=[hit],
        start_entity_id=start,
        env_id=env_id,
        entities_by_id={start: "S"},
        memories_by_id={mem: ("Active memo", "active")},
    )
    assert len(out) == 1
    assert out[0].node.kind == "memory"
    assert out[0].node.name == "Active memo"


def test_project_hits_drops_terminal_with_missing_canonical_entity() -> None:
    env_id, start, ghost = uuid4(), uuid4(), uuid4()
    hit = _one_hop_entity_hit(env_id, start, ghost)
    out = _project_hits(
        hits=[hit],
        start_entity_id=start,
        env_id=env_id,
        entities_by_id={start: "S"},  # ghost intentionally absent
        memories_by_id={},
    )
    assert out == []


def test_project_hits_drops_missing_canonical_memory() -> None:
    """Memory in graph but not in canonical Postgres → skipped (treated as hidden)."""
    env_id, start, ghost = uuid4(), uuid4(), uuid4()
    hit = NeighborHit(
        node=_node(env_id, "memory", ghost),
        path_length=1,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", start),
                dst=_node(env_id, "memory", ghost),
                edge_type="describes",
            ),
        ),
    )
    out = _project_hits(
        hits=[hit],
        start_entity_id=start,
        env_id=env_id,
        entities_by_id={start: "S"},
        memories_by_id={},
    )
    assert out == []


def test_project_hits_preserves_real_edge_orientation_in_path() -> None:
    """Path steps' src/dst should reflect actual relation direction."""
    env_id, start, neighbor = uuid4(), uuid4(), uuid4()
    # Backend reports an inbound edge: neighbor -> start (real direction).
    hit = NeighborHit(
        node=_node(env_id, "entity", neighbor),
        path_length=1,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", neighbor),
                dst=_node(env_id, "entity", start),
                edge_type="caused",
            ),
        ),
    )
    out = _project_hits(
        hits=[hit],
        start_entity_id=start,
        env_id=env_id,
        entities_by_id={start: "S", neighbor: "N"},
        memories_by_id={},
    )
    assert len(out) == 1
    step = out[0].path[0]
    assert step.src_id == neighbor
    assert step.dst_id == start
    assert step.edge_type == "caused"


def test_project_hits_keeps_score() -> None:
    env_id, start, neighbor = uuid4(), uuid4(), uuid4()
    hit = NeighborHit(
        node=_node(env_id, "entity", neighbor),
        path_length=1,
        path=(
            GraphPathStep(
                src=_node(env_id, "entity", start),
                dst=_node(env_id, "entity", neighbor),
                edge_type="rel",
            ),
        ),
        score=0.42,
    )
    out: list[NeighborHitResponse] = _project_hits(
        hits=[hit],
        start_entity_id=start,
        env_id=env_id,
        entities_by_id={start: "S", neighbor: "N"},
        memories_by_id={},
    )
    assert out[0].score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class _StubStore:
    """Minimal stand-in for GraphStore — only ``close`` is exercised."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_singleton_reuses_first_instance(monkeypatch) -> None:
    _reset_default_graph_store_for_tests()
    instances: list[_StubStore] = []

    def fake_factory(_settings):
        store = _StubStore()
        instances.append(store)
        return store

    monkeypatch.setattr("memory_mcp.graph.get_graph_store", fake_factory)

    settings_marker = object()  # truly opaque — never used by stub
    a = asyncio.run(_get_default_graph_store(settings_marker))
    b = asyncio.run(_get_default_graph_store(settings_marker))
    assert a is b
    assert len(instances) == 1
    _reset_default_graph_store_for_tests()


def test_close_resets_singleton(monkeypatch) -> None:
    _reset_default_graph_store_for_tests()
    monkeypatch.setattr(
        "memory_mcp.graph.get_graph_store",
        lambda _settings: _StubStore(),
    )
    store = asyncio.run(_get_default_graph_store(object()))
    asyncio.run(_close_default_graph_store())
    assert store.closed is True
    # After close the singleton is reset; next get returns a fresh store.
    new_store = asyncio.run(_get_default_graph_store(object()))
    assert new_store is not store
    _reset_default_graph_store_for_tests()


def test_close_is_idempotent_when_no_singleton() -> None:
    _reset_default_graph_store_for_tests()
    asyncio.run(_close_default_graph_store())
    asyncio.run(_close_default_graph_store())  # must not raise
