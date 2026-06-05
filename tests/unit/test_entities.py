"""Unit tests for :mod:`memory_mcp.entities` — pure-Python coverage.

DB-touching paths (upsert→update with optimistic-lock, alias dedupe,
merge atomicity) are covered by the integration smoke. These tests
exercise normalization + schema validation in isolation.
"""

from __future__ import annotations

import datetime as dt
from uuid import UUID, uuid4

import pytest

from memory_mcp.db.models import GraphNode, Relation
from memory_mcp.entities import (
    EntityMergeRequest,
    EntityResolveRequest,
    EntityUpsertRequest,
    _merge_entity_graph_nodes,
    _normalize_name,
    _plan_relation_node_repoint,
    _resolve_env_id,
)
from memory_mcp.errors import EnvAmbiguousError
from memory_mcp.identity import AgentContext


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Foo", "foo"),
            ("FOO BAR", "foo bar"),
            ("  trim  me  ", "trim me"),
            ("multiple    spaces", "multiple spaces"),
            ("Foo, Inc.", "foo inc"),
            ("re-entrancy", "re entrancy"),
            ("a/b\\c", "a b c"),
            ("Microsoft®", "microsoft"),
            ("café", "café"),  # unicode preserved (only punct stripped)
            # NFKC normalization: full-width Latin → ASCII Latin
            ("Ｈｅｌｌｏ", "hello"),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert _normalize_name(raw) == expected

    def test_empty_after_normalization_returns_empty(self) -> None:
        assert _normalize_name("   ...,,!!  ") == ""

    def test_empty_string(self) -> None:
        assert _normalize_name("") == ""


class TestEntityUpsertRequest:
    def test_basic(self) -> None:
        req = EntityUpsertRequest(kind="service", canonical_name="My Service")
        assert req.aliases == []
        assert req.metadata == {}
        assert req.expected_version is None

    def test_canonical_name_min_length(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            EntityUpsertRequest(kind="x", canonical_name="")

    def test_canonical_name_punct_only_rejected(self) -> None:
        # Normalizes to empty → rejected by the validator
        with pytest.raises(Exception):  # noqa: B017
            EntityUpsertRequest(kind="x", canonical_name=" ... ,, ")

    def test_alias_punct_only_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            EntityUpsertRequest(
                kind="x",
                canonical_name="ok",
                aliases=["valid", "...,,"],
            )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            EntityUpsertRequest(  # type: ignore[call-arg]
                kind="x",
                canonical_name="ok",
                bogus=True,  # type: ignore[arg-type]
            )

    def test_expected_version_must_be_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            EntityUpsertRequest(kind="x", canonical_name="ok", expected_version=0)


class TestEntityResolveRequest:
    def test_default_limit(self) -> None:
        req = EntityResolveRequest(name="Foo")
        assert req.limit == 20

    def test_limit_bounds(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            EntityResolveRequest(name="x", limit=0)
        with pytest.raises(Exception):  # noqa: B017
            EntityResolveRequest(name="x", limit=300)

    def test_name_required(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            EntityResolveRequest()  # type: ignore[call-arg]


class TestEntityMergeRequest:
    def test_basic(self) -> None:
        keep = uuid4()
        m1 = uuid4()
        m2 = uuid4()
        req = EntityMergeRequest(
            keep_id=keep,
            merge_ids=[m1, m2],
            expected_versions={keep: 1, m1: 1, m2: 2},
        )
        assert req.keep_id == keep

    def test_merge_ids_must_be_unique(self) -> None:
        m1 = uuid4()
        with pytest.raises(Exception):  # noqa: B017
            EntityMergeRequest(
                keep_id=uuid4(),
                merge_ids=[m1, m1],
                expected_versions={m1: 1},
            )

    def test_merge_ids_min_length_one(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            EntityMergeRequest(
                keep_id=uuid4(),
                merge_ids=[],
                expected_versions={},
            )


class TestEntityResolveEnvHandling:
    """``_resolve_env_id`` is the same shape as memories' resolver."""

    def _ctx(self, *envs: UUID) -> AgentContext:
        return AgentContext(agent_id=uuid4(), attached_env_ids=list(envs))

    def test_explicit_wins(self) -> None:
        a, b = uuid4(), uuid4()
        assert _resolve_env_id(explicit=a, ctx=self._ctx(b)) == a

    def test_sole_attached(self) -> None:
        a = uuid4()
        assert _resolve_env_id(explicit=None, ctx=self._ctx(a)) == a

    def test_ambiguous(self) -> None:
        a, b = uuid4(), uuid4()
        with pytest.raises(EnvAmbiguousError):
            _resolve_env_id(explicit=None, ctx=self._ctx(a, b))

    def test_none(self) -> None:
        with pytest.raises(EnvAmbiguousError):
            _resolve_env_id(explicit=None, ctx=self._ctx())


class _FakeExecuteResult:
    def __init__(self, *, scalar: object | None = None, scalars: list[object] | None = None, rows=None):
        self._scalar = scalar
        self._scalars = scalars or []
        self._rows = rows or []

    def scalar_one_or_none(self) -> object | None:
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return list(self._scalars or self._rows)


class _FakeGraphMergeSession:
    def __init__(self, results: list[_FakeExecuteResult], relations: list[Relation]):
        self.results = list(results)
        self.relations = {relation.id: relation for relation in relations}
        self.deleted_relation_ids: list[UUID] = []
        self.deleted_objects: list[object] = []
        self.flush_count = 0

    async def execute(self, statement):
        if self.results:
            return self.results.pop(0)

        table_name = getattr(getattr(statement, "table", None), "name", None)
        params = statement.compile().params
        if table_name == "relations" and statement.is_delete:
            ids = params["id_1"]
            self.deleted_relation_ids.extend(ids)
            for relation_id in ids:
                self.relations.pop(relation_id, None)
            return _FakeExecuteResult()
        if table_name == "relations" and statement.is_update:
            relation = self.relations[params["id_1"]]
            relation.src_node_id = params["src_node_id"]
            relation.dst_node_id = params["dst_node_id"]
            return _FakeExecuteResult()
        raise AssertionError(f"unexpected statement: {statement}")

    async def delete(self, obj: object) -> None:
        self.deleted_objects.append(obj)

    async def flush(self) -> None:
        self.flush_count += 1


def _relation(env: UUID, src: UUID, dst: UUID, type_: str = "relates_to") -> Relation:
    return Relation(
        id=uuid4(),
        env_id=env,
        src_node_id=src,
        dst_node_id=dst,
        type=type_,
        properties={},
        created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        updated_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
    )


class TestEntityMergeGraphNodes:
    @pytest.mark.asyncio
    async def test_ent_merge_with_both_graph_nodes_repoints_relations_and_deletes_orphan(self) -> None:
        """§17.8: both entity graph nodes must collapse without constraint violations."""
        env = uuid4()
        keep_id = uuid4()
        merge_id = uuid4()
        keep_node = GraphNode(id=uuid4(), env_id=env, node_type="entity", entity_id=keep_id)
        merge_node = GraphNode(id=uuid4(), env_id=env, node_type="entity", entity_id=merge_id)
        dst1, dst2, incoming = uuid4(), uuid4(), uuid4()
        keep_edges = [
            _relation(env, keep_node.id, uuid4(), "owns"),
            _relation(env, keep_node.id, uuid4(), "mentions"),
        ]
        merge_edges = [
            _relation(env, merge_node.id, dst1, "owns"),
            _relation(env, merge_node.id, dst2, "mentions"),
            _relation(env, incoming, merge_node.id, "depends_on"),
        ]
        session = _FakeGraphMergeSession(
            [
                _FakeExecuteResult(scalar=keep_node),
                _FakeExecuteResult(scalar=merge_node),
                _FakeExecuteResult(scalars=merge_edges),
                _FakeExecuteResult(rows=[(r.src_node_id, r.dst_node_id, r.type) for r in keep_edges]),
            ],
            keep_edges + merge_edges,
        )

        await _merge_entity_graph_nodes(
            session,
            env_id=env,
            keep_id=keep_id,
            merge_ids=[merge_id],
        )

        assert session.deleted_objects == [merge_node]
        assert len(session.relations) == 5
        assert all(relation.src_node_id != merge_node.id for relation in session.relations.values())
        assert all(relation.dst_node_id != merge_node.id for relation in session.relations.values())
        assert {r.id for r in merge_edges}.issubset(session.relations)
        assert merge_edges[0].src_node_id == keep_node.id
        assert merge_edges[1].src_node_id == keep_node.id
        assert merge_edges[2].dst_node_id == keep_node.id

    @pytest.mark.asyncio
    async def test_merge_graph_node_repoints_entity_id_when_keep_has_no_node(self) -> None:
        env = uuid4()
        keep_id = uuid4()
        merge_node = GraphNode(id=uuid4(), env_id=env, node_type="entity", entity_id=uuid4())
        session = _FakeGraphMergeSession(
            [
                _FakeExecuteResult(scalar=None),
                _FakeExecuteResult(scalar=merge_node),
            ],
            [],
        )

        await _merge_entity_graph_nodes(
            session,
            env_id=env,
            keep_id=keep_id,
            merge_ids=[merge_node.entity_id],
        )

        assert merge_node.entity_id == keep_id
        assert session.deleted_objects == []

    @pytest.mark.asyncio
    async def test_merge_graph_node_noops_when_merge_has_no_node(self) -> None:
        env = uuid4()
        keep_node = GraphNode(id=uuid4(), env_id=env, node_type="entity", entity_id=uuid4())
        session = _FakeGraphMergeSession(
            [
                _FakeExecuteResult(scalar=keep_node),
                _FakeExecuteResult(scalar=None),
            ],
            [],
        )

        await _merge_entity_graph_nodes(
            session,
            env_id=env,
            keep_id=keep_node.entity_id,
            merge_ids=[uuid4()],
        )

        assert session.deleted_objects == []
        assert session.flush_count == 0

    def test_relation_repoint_plan_deletes_collisions_before_update(self) -> None:
        env = uuid4()
        keep_node_id = uuid4()
        merge_node_id = uuid4()
        dst = uuid4()
        duplicate_existing = (keep_node_id, dst, "owns")
        first = _relation(env, merge_node_id, dst, "owns")
        second = _relation(env, merge_node_id, dst, "owns")
        incoming = _relation(env, uuid4(), merge_node_id, "depends_on")

        delete_ids, move_values = _plan_relation_node_repoint(
            [first, second, incoming],
            existing_keys={duplicate_existing},
            from_node_id=merge_node_id,
            to_node_id=keep_node_id,
        )

        assert delete_ids == [first.id, second.id]
        assert move_values == {incoming.id: (incoming.src_node_id, keep_node_id)}
