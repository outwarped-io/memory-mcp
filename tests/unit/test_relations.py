"""Unit tests for ``memory_mcp.relations`` (no DB).

Covers:

* ``RelationEndpoint`` / ``RelationLinkRequest`` Pydantic schema validation.
* ``_resolve_env_id`` env scoping behavior.
* ``_endpoint_for_node`` direction.
* ``_relation_payload`` snapshot shape (used for outbox + audit).
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from pydantic import ValidationError

from memory_mcp import relations as relation_api
from memory_mcp.db.models import GraphNode, Relation
from memory_mcp.errors import CycleDetectedError, EnvAmbiguousError
from memory_mcp.identity import AgentContext
from memory_mcp.relations import (
    RelationEndpoint,
    RelationLinkRequest,
    _endpoint_for_node,
    _relation_payload,
    _resolve_env_id,
)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestRelationEndpointSchema:
    def test_accepts_entity_kind(self) -> None:
        ep = RelationEndpoint(kind="entity", id=uuid4())
        assert ep.kind == "entity"

    def test_accepts_memory_kind(self) -> None:
        ep = RelationEndpoint(kind="memory", id=uuid4())
        assert ep.kind == "memory"

    def test_accepts_task_kind(self) -> None:
        ep = RelationEndpoint(kind="task", id=uuid4())
        assert ep.kind == "task"

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            RelationEndpoint(kind="unknown", id=uuid4())  # type: ignore[arg-type]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RelationEndpoint(kind="entity", id=uuid4(), extra="forbid")  # type: ignore[call-arg]


class TestRelationLinkRequestSchema:
    def _ep(self) -> RelationEndpoint:
        return RelationEndpoint(kind="entity", id=uuid4())

    def test_accepts_minimal_request(self) -> None:
        req = RelationLinkRequest(src=self._ep(), dst=self._ep(), type="relates_to")
        assert req.properties == {}
        assert req.env_id is None
        assert req.expected_version is None

    def test_rejects_blank_type(self) -> None:
        with pytest.raises(ValidationError):
            RelationLinkRequest(src=self._ep(), dst=self._ep(), type="   ")

    def test_rejects_empty_type(self) -> None:
        with pytest.raises(ValidationError):
            RelationLinkRequest(src=self._ep(), dst=self._ep(), type="")

    def test_rejects_too_long_type(self) -> None:
        with pytest.raises(ValidationError):
            RelationLinkRequest(src=self._ep(), dst=self._ep(), type="x" * 201)

    def test_rejects_zero_expected_version(self) -> None:
        with pytest.raises(ValidationError):
            RelationLinkRequest(
                src=self._ep(),
                dst=self._ep(),
                type="x",
                expected_version=0,
            )

    def test_accepts_properties_and_env(self) -> None:
        env = uuid4()
        req = RelationLinkRequest(
            src=self._ep(),
            dst=self._ep(),
            type="depends_on",
            properties={"weight": 0.7, "tag": "hot"},
            env_id=env,
            expected_version=2,
        )
        assert req.properties == {"weight": 0.7, "tag": "hot"}
        assert req.env_id == env
        assert req.expected_version == 2

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RelationLinkRequest(
                src=self._ep(),
                dst=self._ep(),
                type="x",
                bogus=1,  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# _resolve_env_id
# ---------------------------------------------------------------------------


class TestResolveEnvId:
    def test_explicit_wins(self) -> None:
        explicit = uuid4()
        ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[uuid4(), uuid4()])
        assert _resolve_env_id(explicit=explicit, ctx=ctx) == explicit

    def test_sole_attached_resolved(self) -> None:
        env = uuid4()
        ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env])
        assert _resolve_env_id(explicit=None, ctx=ctx) == env

    def test_dedup_attached_collapses_to_one(self) -> None:
        env = uuid4()
        ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env, env, env])
        assert _resolve_env_id(explicit=None, ctx=ctx) == env

    def test_ambiguous_attached_raises(self) -> None:
        ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[uuid4(), uuid4()])
        with pytest.raises(EnvAmbiguousError):
            _resolve_env_id(explicit=None, ctx=ctx)

    def test_no_attached_raises(self) -> None:
        ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[])
        with pytest.raises(EnvAmbiguousError):
            _resolve_env_id(explicit=None, ctx=ctx)


# ---------------------------------------------------------------------------
# _endpoint_for_node
# ---------------------------------------------------------------------------


class TestEndpointForNode:
    def test_entity_node(self) -> None:
        rec = uuid4()
        node = GraphNode(env_id=uuid4(), node_type="entity", entity_id=rec)
        ep = _endpoint_for_node(node)
        assert ep.kind == "entity" and ep.id == rec

    def test_memory_node(self) -> None:
        rec = uuid4()
        node = GraphNode(env_id=uuid4(), node_type="memory", memory_id=rec)
        ep = _endpoint_for_node(node)
        assert ep.kind == "memory" and ep.id == rec

    def test_task_node(self) -> None:
        rec = uuid4()
        node = GraphNode(env_id=uuid4(), node_type="task", task_id=rec)
        ep = _endpoint_for_node(node)
        assert ep.kind == "task" and ep.id == rec


# ---------------------------------------------------------------------------
# _relation_payload (outbox / audit shape)
# ---------------------------------------------------------------------------


class TestRelationPayload:
    def test_payload_shape(self) -> None:
        env = uuid4()
        src_rec = uuid4()
        dst_rec = uuid4()
        src = GraphNode(
            id=uuid4(),
            env_id=env,
            node_type="entity",
            entity_id=src_rec,
        )
        dst = GraphNode(
            id=uuid4(),
            env_id=env,
            node_type="memory",
            memory_id=dst_rec,
        )
        rel = Relation(
            id=uuid4(),
            env_id=env,
            src_node_id=src.id,
            dst_node_id=dst.id,
            type="mentions",
            properties={"weight": 0.5},
            version=3,
            created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            updated_at=dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
        )
        p = _relation_payload(rel, src, dst)
        assert p["relation_id"] == str(rel.id)
        assert p["env_id"] == str(env)
        assert p["type"] == "mentions"
        assert p["properties"] == {"weight": 0.5}
        assert p["src"] == {
            "kind": "entity",
            "id": str(src_rec),
            "node_id": str(src.id),
        }
        assert p["dst"] == {
            "kind": "memory",
            "id": str(dst_rec),
            "node_id": str(dst.id),
        }
        assert p["version"] == 3
        assert p["created_at"] == "2026-01-01T00:00:00+00:00"
        assert p["updated_at"] == "2026-01-02T00:00:00+00:00"

    def test_payload_handles_none_properties(self) -> None:
        env = uuid4()
        src = GraphNode(
            id=uuid4(),
            env_id=env,
            node_type="entity",
            entity_id=uuid4(),
        )
        dst = GraphNode(
            id=uuid4(),
            env_id=env,
            node_type="entity",
            entity_id=uuid4(),
        )
        rel = Relation(
            id=uuid4(),
            env_id=env,
            src_node_id=src.id,
            dst_node_id=dst.id,
            type="x",
            properties=None,  # type: ignore[arg-type]
            version=1,
            created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            updated_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        )
        p = _relation_payload(rel, src, dst)
        assert p["properties"] == {}


class _FakeScalarResult:
    def scalar_one_or_none(self) -> object | None:
        return None


class _FakeRelationSession:
    def __init__(self, edges: set[tuple[object, object]]) -> None:
        self.edges = edges

    async def execute(self, _stmt: object) -> _FakeScalarResult:
        return _FakeScalarResult()

    def add(self, obj: object) -> None:
        if isinstance(obj, Relation):
            self.edges.add((obj.src_node_id, obj.dst_node_id))

    async def flush(self) -> None:
        return None

    async def refresh(self, obj: object) -> None:
        if isinstance(obj, Relation):
            obj.id = obj.id or uuid4()
            obj.version = 1
            obj.created_at = dt.datetime.now(dt.UTC)
            obj.updated_at = obj.created_at


@pytest.mark.asyncio
async def test_relation_link_depends_on_task_cycle_uses_task_cycle_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    env = uuid4()
    task_a = uuid4()
    task_b = uuid4()
    edges: set[tuple[object, object]] = set()

    @asynccontextmanager
    async def fake_scope():
        yield _FakeRelationSession(edges)

    async def fake_graph_node(_session: object, *, env_id, endpoint):
        return GraphNode(id=endpoint.id, env_id=env_id, node_type="task", task_id=endpoint.id)

    async def fake_would_cycle(_session: object, _env_id, src_task_id, dst_task_id) -> bool:
        return src_task_id == dst_task_id or (dst_task_id, src_task_id) in edges

    async def fake_lock(_session: object, _env_id) -> None:
        return None

    async def noop(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(relation_api, "session_scope", fake_scope)
    monkeypatch.setattr(relation_api, "_ensure_graph_node", fake_graph_node)
    monkeypatch.setattr(relation_api, "would_cycle", fake_would_cycle)
    monkeypatch.setattr(relation_api, "_acquire_dep_lock", fake_lock)
    monkeypatch.setattr(relation_api, "_record_relation_audit", noop)
    monkeypatch.setattr(relation_api, "enqueue_event", noop)

    ctx = AgentContext(agent_id=uuid4(), attached_env_ids=[env])
    await relation_api.relation_link(
        RelationLinkRequest(
            env_id=env,
            src=RelationEndpoint(kind="task", id=task_a),
            dst=RelationEndpoint(kind="task", id=task_b),
            type="depends_on",
        ),
        ctx=ctx,
    )

    with pytest.raises(CycleDetectedError):
        await relation_api.relation_link(
            RelationLinkRequest(
                env_id=env,
                src=RelationEndpoint(kind="task", id=task_b),
                dst=RelationEndpoint(kind="task", id=task_a),
                type="depends_on",
            ),
            ctx=ctx,
        )
