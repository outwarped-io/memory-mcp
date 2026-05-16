from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest

from memory_mcp.db.models import Entity, Environment, GraphNode, Memory, Relation, Tag
from memory_mcp.env_ops import diff as diff_mod
from memory_mcp.errors import NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp_schemas.env_ops import DiffGranularity, EnvDiffRequest


class _ScalarResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def first(self) -> object | None:
        return self._rows[0] if self._rows else None


class _ExecuteResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows

    def scalars(self) -> _ScalarResult:
        return _ScalarResult(self._rows)


class _Session:
    def __init__(self, fixture: "_Fixture") -> None:
        self._fixture = fixture

    async def connection(self, **_kwargs: object) -> None:
        return None

    async def execute(self, stmt: object) -> _ExecuteResult:
        compiled = stmt.compile()
        sql = str(compiled)
        params = list(compiled.params.values())
        wants_count = "count(" in sql.lower()

        if "FROM environments" in sql:
            if wants_count:
                return _ExecuteResult([(env.id, 1) for env in self._fixture.envs])
            wanted = next((value for value in params if isinstance(value, UUID)), None)
            return _ExecuteResult([env for env in self._fixture.envs if env.id == wanted])

        if "FROM memory_sources" in sql:
            return _ExecuteResult([])
        if "FROM memory_lineage" in sql:
            return _ExecuteResult([])

        if wants_count:
            table = _table_name(sql)
            rows = getattr(self._fixture, table, [])
            counts: dict[UUID, int] = {}
            for row in rows:
                env_id = row.env_id
                counts[env_id] = counts.get(env_id, 0) + 1
            return _ExecuteResult(list(counts.items()))

        if "FROM memories" in sql:
            return _ExecuteResult(
                [
                    (row.id, row.env_id, row.kind, row.body, row.metadata_)
                    for row in self._fixture.memories
                ]
            )
        if "FROM entities" in sql:
            return _ExecuteResult([(row.env_id, row.normalized_name) for row in self._fixture.entities])
        if "FROM relations" in sql:
            return _ExecuteResult([self._fixture.relation_tuple(row) for row in self._fixture.relations])
        if "FROM graph_nodes" in sql:
            return _ExecuteResult(
                [
                    (row.env_id, row.node_type, row.memory_id, row.entity_id, row.task_id)
                    for row in self._fixture.graph_nodes
                ]
            )
        if "FROM tags" in sql:
            return _ExecuteResult([(row.env_id, row.name) for row in self._fixture.tags])
        if "FROM tasks" in sql:
            return _ExecuteResult([])
        return _ExecuteResult([])


class _Fixture:
    def __init__(self, *, deleted_b: bool = False) -> None:
        now = dt.datetime.now(dt.UTC)
        self.env_a = _env("a", now)
        self.env_b = _env("b", now, deleted=deleted_b)
        self.envs = [self.env_a, self.env_b]
        self.memories: list[Memory] = []
        self.tags: list[Tag] = []
        self.memory_tags: list[object] = []
        self.entities: list[Entity] = []
        self.entity_aliases: list[object] = []
        self.relations: list[Relation] = []
        self.graph_nodes: list[GraphNode] = []
        self.tasks: list[object] = []
        self.env_grants: list[object] = []
        self.dream_runs: list[object] = []
        self.dream_proposals: list[object] = []
        self.audit_log: list[object] = []

    def add_memory(self, env_id: UUID, body: str, *, kind: str = "fact") -> Memory:
        memory = Memory(
            id=uuid4(),
            env_id=env_id,
            kind=kind,
            status="active",
            title=body[:20],
            body=body,
            salience=0.5,
            confidence=0.5,
            pinned=False,
            metadata_={},
            version=1,
        )
        self.memories.append(memory)
        return memory

    def add_entity(self, env_id: UUID, key: str) -> Entity:
        entity = Entity(
            id=uuid4(),
            env_id=env_id,
            kind="thing",
            canonical_name=key,
            normalized_name=key,
            metadata_={},
            version=1,
        )
        self.entities.append(entity)
        return entity

    def add_tag(self, env_id: UUID, name: str) -> None:
        self.tags.append(Tag(id=uuid4(), env_id=env_id, name=name))

    def add_relation(self, env_id: UUID, src_entity_id: UUID, dst_entity_id: UUID, rel_type: str) -> None:
        src = GraphNode(id=uuid4(), env_id=env_id, node_type="entity", entity_id=src_entity_id)
        dst = GraphNode(id=uuid4(), env_id=env_id, node_type="entity", entity_id=dst_entity_id)
        self.graph_nodes.extend([src, dst])
        self.relations.append(
            Relation(
                id=uuid4(),
                env_id=env_id,
                src_node_id=src.id,
                dst_node_id=dst.id,
                type=rel_type,
                properties={},
                version=1,
            )
        )

    def relation_tuple(self, relation: Relation) -> tuple[object, ...]:
        src = next(row for row in self.graph_nodes if row.id == relation.src_node_id)
        dst = next(row for row in self.graph_nodes if row.id == relation.dst_node_id)
        return (
            relation.env_id,
            src.node_type,
            src.memory_id,
            src.entity_id,
            src.task_id,
            relation.type,
            dst.node_type,
            dst.memory_id,
            dst.entity_id,
            dst.task_id,
        )


def _env(name: str, now: dt.datetime, *, deleted: bool = False) -> Environment:
    return Environment(
        id=uuid4(),
        name=name,
        kind=None,
        retention_policy={},
        default_embedding_model_id="test-model",
        created_at=now,
        status="deleted" if deleted else "active",
        deleted_at=now if deleted else None,
    )


def _table_name(sql: str) -> str:
    for table in (
        "memory_tags",
        "entity_aliases",
        "env_grants",
        "dream_runs",
        "dream_proposals",
        "audit_log",
        "memories",
        "tags",
        "entities",
        "relations",
        "graph_nodes",
        "tasks",
    ):
        if f"FROM {table}" in sql:
            return table
    return "missing"


def _install(monkeypatch: pytest.MonkeyPatch, fixture: _Fixture) -> None:
    @asynccontextmanager
    async def fake_session_scope() -> AsyncIterator[_Session]:
        yield _Session(fixture)

    monkeypatch.setattr(diff_mod, "session_scope", fake_session_scope)


def _ctx() -> AgentContext:
    return AgentContext(agent_id=uuid4())


async def _run(fixture: _Fixture, granularity: DiffGranularity):
    return await diff_mod.diff_envs(
        EnvDiffRequest(env_a_id=fixture.env_a.id, env_b_id=fixture.env_b.id, granularity=granularity),
        ctx=_ctx(),
    )


@pytest.mark.asyncio
async def test_diff_counts_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    for index in range(3):
        fixture.add_memory(fixture.env_a.id, f"a-{index}")
    for index in range(5):
        fixture.add_memory(fixture.env_b.id, f"b-{index}")
    _install(monkeypatch, fixture)

    out = await _run(fixture, DiffGranularity.counts)

    assert out.counts["memories"]["a"] == 3
    assert out.counts["memories"]["b"] == 5


@pytest.mark.asyncio
async def test_diff_entity_keys_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    for key in ("x", "y"):
        fixture.add_entity(fixture.env_a.id, key)
    for key in ("y", "z"):
        fixture.add_entity(fixture.env_b.id, key)
    _install(monkeypatch, fixture)

    out = await _run(fixture, DiffGranularity.entity_keys)

    assert set(out.entity_keys["only_in_a"]) == {"x"}  # type: ignore[index]
    assert set(out.entity_keys["only_in_b"]) == {"z"}  # type: ignore[index]
    assert set(out.entity_keys["in_both"]) == {"y"}  # type: ignore[index]


@pytest.mark.asyncio
async def test_diff_memory_hashes_identical_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    for index in range(4):
        body = f"same-{index}"
        fixture.add_memory(fixture.env_a.id, body)
        fixture.add_memory(fixture.env_b.id, body)
    _install(monkeypatch, fixture)

    out = await _run(fixture, DiffGranularity.memory_hashes)

    assert out.memory_hashes["identical"]["count"] == 4  # type: ignore[index]
    assert out.memory_hashes["only_in_a"]["count"] == 0  # type: ignore[index]
    assert out.memory_hashes["only_in_b"]["count"] == 0  # type: ignore[index]


@pytest.mark.asyncio
async def test_diff_memory_hashes_disjoint(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    for index in range(3):
        fixture.add_memory(fixture.env_a.id, f"a-{index}")
        fixture.add_memory(fixture.env_b.id, f"b-{index}")
    _install(monkeypatch, fixture)

    out = await _run(fixture, DiffGranularity.memory_hashes)

    assert out.memory_hashes["only_in_a"]["count"] == 3  # type: ignore[index]
    assert out.memory_hashes["only_in_b"]["count"] == 3  # type: ignore[index]
    assert out.memory_hashes["identical"]["count"] == 0  # type: ignore[index]


@pytest.mark.asyncio
async def test_diff_full_includes_relations_and_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    fixture.add_tag(fixture.env_a.id, "shared")
    fixture.add_tag(fixture.env_a.id, "a-only")
    fixture.add_tag(fixture.env_b.id, "shared")
    fixture.add_tag(fixture.env_b.id, "b-only")
    src_id = uuid4()
    dst_id = uuid4()
    fixture.add_relation(fixture.env_a.id, src_id, dst_id, "related_to")
    fixture.add_relation(fixture.env_b.id, src_id, dst_id, "related_to")
    fixture.add_relation(fixture.env_b.id, src_id, uuid4(), "mentions")
    _install(monkeypatch, fixture)

    out = await _run(fixture, DiffGranularity.full)

    assert out.full["tags"]["only_in_a"]["count"] == 1  # type: ignore[index]
    assert out.full["tags"]["only_in_b"]["count"] == 1  # type: ignore[index]
    assert out.full["relations"]["in_both_count"] == 1  # type: ignore[index]
    assert out.full["relations"]["only_in_b"]["count"] == 1  # type: ignore[index]


@pytest.mark.asyncio
async def test_diff_rejects_deleted_env(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture(deleted_b=True)
    _install(monkeypatch, fixture)

    with pytest.raises(NotFoundError) as exc:
        await _run(fixture, DiffGranularity.counts)

    assert exc.value.code == "ENV_DELETED"


@pytest.mark.asyncio
async def test_diff_counts_when_empty_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    _install(monkeypatch, fixture)

    out = await _run(fixture, DiffGranularity.counts)

    assert all(values["a"] == 0 and values["b"] == 0 for name, values in out.counts.items() if name != "environments")


@pytest.mark.asyncio
async def test_diff_truncation_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    for index in range(501):
        fixture.add_entity(fixture.env_a.id, f"x-{index:03d}")
    _install(monkeypatch, fixture)

    out = await _run(fixture, DiffGranularity.entity_keys)

    assert out.truncated is True
    assert len(out.entity_keys["only_in_a"]) == 500  # type: ignore[index]
