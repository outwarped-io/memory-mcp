"""Environment diff implementation for v0.8 env operations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from memory_mcp_schemas.env_ops import DiffGranularity, EnvDiffRequest, EnvDiffResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from memory_mcp import rbac
from memory_mcp.db.models import (
    AuditLog,
    DreamProposal,
    DreamRun,
    Entity,
    EntityAlias,
    EnvGrant,
    Environment,
    GraphNode,
    Memory,
    MemoryLineage,
    MemorySource,
    MemoryTag,
    Relation,
    Tag,
    Task,
)
from memory_mcp.db.postgres import session_scope
from memory_mcp.errors import NotFoundError
from memory_mcp.identity import AgentContext

ENTITY_LIMIT = 500
MEMORY_SAMPLE_LIMIT = 100
FULL_SAMPLE_LIMIT = 50


@dataclass(frozen=True)
class _TableSpec:
    name: str
    model: type[Any]
    env_column: Any


_DIRECT_TABLES = (
    _TableSpec("memories", Memory, Memory.env_id),
    _TableSpec("tags", Tag, Tag.env_id),
    _TableSpec("memory_tags", MemoryTag, MemoryTag.env_id),
    _TableSpec("entities", Entity, Entity.env_id),
    _TableSpec("entity_aliases", EntityAlias, EntityAlias.env_id),
    _TableSpec("relations", Relation, Relation.env_id),
    _TableSpec("graph_nodes", GraphNode, GraphNode.env_id),
    _TableSpec("tasks", Task, Task.env_id),
    _TableSpec("env_grants", EnvGrant, EnvGrant.env_id),
    _TableSpec("dream_runs", DreamRun, DreamRun.env_id),
    _TableSpec("dream_proposals", DreamProposal, DreamProposal.env_id),
    _TableSpec("audit_log", AuditLog, AuditLog.env_id),
)


async def diff_envs(request: EnvDiffRequest, *, ctx: AgentContext) -> EnvDiffResponse:
    """Compare two environments at the requested granularity.

    ``counts`` deliberately returns only total row counts per table
    (``{"a": int, "b": int}``) because row identity across environments is
    table-specific. ``entity_keys`` adds bounded entity-key set slices,
    ``memory_hashes`` adds multiset content-hash counts and samples, and
    ``full`` adds bounded tag/relation/task/graph-node/lineage samples.
    ``body_changed`` under ``memory_hashes`` is a conservative heuristic based
    on ``(kind, first_100_body_chars)`` matching while content hashes differ.
    """

    rbac.require("read", request.env_a_id, ctx)
    rbac.require("read", request.env_b_id, ctx)

    async with session_scope() as session:
        if request.granularity in {DiffGranularity.memory_hashes, DiffGranularity.full}:
            await _set_repeatable_read(session)

        await _load_environment(session, request.env_a_id)
        await _load_environment(session, request.env_b_id)

        counts = await _counts(session, request.env_a_id, request.env_b_id)
        response = EnvDiffResponse(granularity=request.granularity, counts=counts)

        if request.granularity in {
            DiffGranularity.entity_keys,
            DiffGranularity.memory_hashes,
            DiffGranularity.full,
        }:
            entity_diff, truncated = await _entity_keys(session, request.env_a_id, request.env_b_id)
            response = response.model_copy(
                update={
                    "entity_keys": entity_diff,
                    "truncated": response.truncated or truncated,
                    "entity_keys_a_only": entity_diff["only_in_a"],
                    "entity_keys_b_only": entity_diff["only_in_b"],
                }
            )

        memory_index: _MemoryIndex | None = None
        if request.granularity in {DiffGranularity.memory_hashes, DiffGranularity.full}:
            memory_index = await _memory_hashes(session, request.env_a_id, request.env_b_id)
            response = response.model_copy(
                update={
                    "memory_hashes": memory_index.report,
                    "memory_hashes_a_only": memory_index.report["only_in_a"]["sample"],
                    "memory_hashes_b_only": memory_index.report["only_in_b"]["sample"],
                    "memory_hashes_changed": memory_index.report["body_changed"]["sample"],
                }
            )

        if request.granularity == DiffGranularity.full:
            assert memory_index is not None
            full = await _full_diff(session, request.env_a_id, request.env_b_id, memory_index.id_to_hash)
            response = response.model_copy(
                update={
                    "full": full,
                    "per_table_a_only": {name: table["only_in_a"]["count"] for name, table in full.items()},
                    "per_table_b_only": {name: table["only_in_b"]["count"] for name, table in full.items()},
                    "per_table_both": {name: table["in_both_count"] for name, table in full.items()},
                }
            )

        return response


async def _set_repeatable_read(session: AsyncSession) -> None:
    try:
        await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
    except AttributeError:
        return


async def _load_environment(session: AsyncSession, env_id: UUID) -> Environment:
    env = (await session.execute(select(Environment).where(Environment.id == env_id))).scalars().first()
    if env is None:
        raise NotFoundError(f"environment {env_id} not found", env_id=str(env_id))
    if getattr(env, "status", "active") == "deleted":
        exc = NotFoundError(f"env {env_id} is deleted", env_id=str(env_id))
        exc.code = "ENV_DELETED"
        raise exc
    return env


async def _counts(session: AsyncSession, env_a_id: UUID, env_b_id: UUID) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {
        "environments": await _environment_count(session, env_a_id, env_b_id),
        "memory_lineage": await _memory_lineage_count(session, env_a_id, env_b_id),
        "memory_sources": await _memory_sources_count(session, env_a_id, env_b_id),
    }
    for spec in _DIRECT_TABLES:
        out[spec.name] = await _direct_count(session, spec, env_a_id, env_b_id)
    return out


async def _direct_count(
    session: AsyncSession,
    spec: _TableSpec,
    env_a_id: UUID,
    env_b_id: UUID,
) -> dict[str, int]:
    rows = (
        await session.execute(
            select(spec.env_column, func.count())
            .select_from(spec.model)
            .where(spec.env_column.in_([env_a_id, env_b_id]))
            .group_by(spec.env_column)
        )
    ).all()
    return _count_pair(rows, env_a_id, env_b_id)


async def _environment_count(session: AsyncSession, env_a_id: UUID, env_b_id: UUID) -> dict[str, int]:
    rows = (
        await session.execute(
            select(Environment.id, func.count())
            .where(Environment.id.in_([env_a_id, env_b_id]))
            .group_by(Environment.id)
        )
    ).all()
    return _count_pair(rows, env_a_id, env_b_id)


async def _memory_sources_count(session: AsyncSession, env_a_id: UUID, env_b_id: UUID) -> dict[str, int]:
    rows = (
        await session.execute(
            select(Memory.env_id, func.count())
            .select_from(MemorySource)
            .join(Memory, MemorySource.memory_id == Memory.id)
            .where(Memory.env_id.in_([env_a_id, env_b_id]))
            .group_by(Memory.env_id)
        )
    ).all()
    return _count_pair(rows, env_a_id, env_b_id)


async def _memory_lineage_count(session: AsyncSession, env_a_id: UUID, env_b_id: UUID) -> dict[str, int]:
    parent = aliased(Memory)
    rows = (
        await session.execute(
            select(parent.env_id, func.count())
            .select_from(MemoryLineage)
            .join(parent, MemoryLineage.parent_memory_id == parent.id)
            .where(parent.env_id.in_([env_a_id, env_b_id]))
            .group_by(parent.env_id)
        )
    ).all()
    return _count_pair(rows, env_a_id, env_b_id)


def _count_pair(rows: list[tuple[Any, int]], env_a_id: UUID, env_b_id: UUID) -> dict[str, int]:
    counts = {str(env_id): int(count) for env_id, count in rows}
    return {"a": counts.get(str(env_a_id), 0), "b": counts.get(str(env_b_id), 0)}


async def _entity_keys(
    session: AsyncSession,
    env_a_id: UUID,
    env_b_id: UUID,
) -> tuple[dict[str, list[str]], bool]:
    rows = (
        await session.execute(
            select(Entity.env_id, Entity.normalized_name)
            .where(Entity.env_id.in_([env_a_id, env_b_id]))
            .order_by(Entity.normalized_name)
        )
    ).all()
    by_env = _set_by_env(rows, env_a_id, env_b_id)
    return _bounded_set_diff(by_env["a"], by_env["b"], ENTITY_LIMIT)


@dataclass(frozen=True)
class _MemoryIndex:
    report: dict[str, Any]
    id_to_hash: dict[UUID, str]


async def _memory_hashes(session: AsyncSession, env_a_id: UUID, env_b_id: UUID) -> _MemoryIndex:
    ids_by_hash: dict[str, dict[str, list[UUID]]] = defaultdict(lambda: {"a": [], "b": []})
    hashes_by_stable_key: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: {"a": set(), "b": set()})
    ids_by_stable_key: dict[tuple[str, str], dict[str, list[UUID]]] = defaultdict(lambda: {"a": [], "b": []})
    id_to_hash: dict[UUID, str] = {}

    stmt = (
        select(Memory.id, Memory.env_id, Memory.kind, Memory.body, Memory.metadata_)
        .where(Memory.env_id.in_([env_a_id, env_b_id]))
        .order_by(Memory.id)
        .execution_options(yield_per=5000)
    )
    async for memory_id, env_id, kind, body, payload in _iter_rows(session, stmt):
        side = "a" if env_id == env_a_id else "b"
        digest = _memory_hash(kind, body, payload)
        id_to_hash[memory_id] = digest
        ids_by_hash[digest][side].append(memory_id)
        stable_key = (kind, (body or "")[:100])
        hashes_by_stable_key[stable_key][side].add(digest)
        ids_by_stable_key[stable_key][side].append(memory_id)

    counter_a = Counter({digest: len(sides["a"]) for digest, sides in ids_by_hash.items()})
    counter_b = Counter({digest: len(sides["b"]) for digest, sides in ids_by_hash.items()})
    only_a, sample_a = _counter_only(counter_a, counter_b, ids_by_hash, "a", MEMORY_SAMPLE_LIMIT)
    only_b, sample_b = _counter_only(counter_b, counter_a, ids_by_hash, "b", MEMORY_SAMPLE_LIMIT)
    identical = sum(min(counter_a[digest], counter_b[digest]) for digest in set(counter_a) | set(counter_b))

    changed_ids: list[UUID] = []
    changed_count = 0
    for stable_key, hashes in hashes_by_stable_key.items():
        if hashes["a"] and hashes["b"] and hashes["a"] != hashes["b"]:
            changed_count += 1
            if len(changed_ids) < MEMORY_SAMPLE_LIMIT:
                changed_ids.extend(ids_by_stable_key[stable_key]["a"][: MEMORY_SAMPLE_LIMIT - len(changed_ids)])

    return _MemoryIndex(
        report={
            "only_in_a": {"count": only_a, "sample": sample_a},
            "only_in_b": {"count": only_b, "sample": sample_b},
            "identical": {"count": identical},
            "body_changed": {"count": changed_count, "sample": changed_ids[:MEMORY_SAMPLE_LIMIT]},
        },
        id_to_hash=id_to_hash,
    )


async def _iter_rows(session: AsyncSession, stmt: Any):
    if hasattr(session, "stream"):
        result = await session.stream(stmt)
        async for row in result:
            yield row
        return
    for row in (await session.execute(stmt)).all():
        yield row


def _memory_hash(kind: str, body: str, payload: Any) -> str:
    normalized = json.dumps(payload or {}, sort_keys=True, separators=(",", ":"), default=str)
    content = f"{kind}\x00{body or ''}\x00{normalized}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _counter_only(
    lhs: Counter[str],
    rhs: Counter[str],
    ids_by_hash: dict[str, dict[str, list[UUID]]],
    side: str,
    sample_limit: int,
) -> tuple[int, list[UUID]]:
    total = 0
    sample: list[UUID] = []
    for digest in sorted(lhs):
        extra = max(lhs[digest] - rhs.get(digest, 0), 0)
        if extra == 0:
            continue
        total += extra
        if len(sample) < sample_limit:
            sample.extend(ids_by_hash[digest][side][: min(extra, sample_limit - len(sample))])
    return total, sample


async def _full_diff(
    session: AsyncSession,
    env_a_id: UUID,
    env_b_id: UUID,
    memory_hashes: dict[UUID, str],
) -> dict[str, Any]:
    tags = await _key_rows(
        session,
        select(Tag.env_id, Tag.name).where(Tag.env_id.in_([env_a_id, env_b_id])),
        env_a_id,
        env_b_id,
    )
    tasks = await _key_rows(
        session,
        select(Task.env_id, Task.title).where(Task.env_id.in_([env_a_id, env_b_id])),
        env_a_id,
        env_b_id,
    )
    graph_nodes = await _graph_node_keys(session, env_a_id, env_b_id)
    relations = await _relation_keys(session, env_a_id, env_b_id)
    lineage = await _lineage_keys(session, env_a_id, env_b_id, memory_hashes)
    return {
        "tags": _sampled_diff(tags["a"], tags["b"]),
        "relations": _sampled_diff(relations["a"], relations["b"]),
        "tasks": _sampled_diff(tasks["a"], tasks["b"]),
        "graph_nodes": _sampled_diff(graph_nodes["a"], graph_nodes["b"]),
        "memory_lineage": _sampled_diff(lineage["a"], lineage["b"]),
    }


async def _key_rows(
    session: AsyncSession,
    stmt: Any,
    env_a_id: UUID,
    env_b_id: UUID,
) -> dict[str, set[tuple[Any, ...]]]:
    rows = (await session.execute(stmt)).all()
    out = {"a": set(), "b": set()}
    for row in rows:
        if row[0] == env_a_id:
            out["a"].add(tuple(row[1:]))
        elif row[0] == env_b_id:
            out["b"].add(tuple(row[1:]))
    return out


async def _graph_node_keys(session: AsyncSession, env_a_id: UUID, env_b_id: UUID) -> dict[str, set[tuple[str, str]]]:
    rows = (
        await session.execute(
            select(
                GraphNode.env_id, GraphNode.node_type, GraphNode.memory_id, GraphNode.entity_id, GraphNode.task_id
            ).where(GraphNode.env_id.in_([env_a_id, env_b_id]))
        )
    ).all()
    out = {"a": set(), "b": set()}
    for env_id, node_type, memory_id, entity_id, task_id in rows:
        side = "a" if env_id == env_a_id else "b"
        out[side].add((node_type, str(memory_id or entity_id or task_id)))
    return out


async def _relation_keys(session: AsyncSession, env_a_id: UUID, env_b_id: UUID) -> dict[str, set[tuple[str, str, str]]]:
    src = aliased(GraphNode)
    dst = aliased(GraphNode)
    rows = (
        await session.execute(
            select(
                Relation.env_id,
                src.node_type,
                src.memory_id,
                src.entity_id,
                src.task_id,
                Relation.type,
                dst.node_type,
                dst.memory_id,
                dst.entity_id,
                dst.task_id,
            )
            .join(src, Relation.src_node_id == src.id)
            .join(dst, Relation.dst_node_id == dst.id)
            .where(Relation.env_id.in_([env_a_id, env_b_id]))
        )
    ).all()
    out = {"a": set(), "b": set()}
    for row in rows:
        side = "a" if row[0] == env_a_id else "b"
        src_key = f"{row[1]}:{row[2] or row[3] or row[4]}"
        dst_key = f"{row[6]}:{row[7] or row[8] or row[9]}"
        out[side].add((src_key, row[5], dst_key))
    return out


async def _lineage_keys(
    session: AsyncSession,
    env_a_id: UUID,
    env_b_id: UUID,
    memory_hashes: dict[UUID, str],
) -> dict[str, set[tuple[str, str, str]]]:
    parent = aliased(Memory)
    rows = (
        await session.execute(
            select(parent.env_id, MemoryLineage.parent_memory_id, MemoryLineage.child_memory_id, MemoryLineage.relation)
            .select_from(MemoryLineage)
            .join(parent, MemoryLineage.parent_memory_id == parent.id)
            .where(parent.env_id.in_([env_a_id, env_b_id]))
        )
    ).all()
    out = {"a": set(), "b": set()}
    for env_id, parent_id, child_id, relation in rows:
        parent_hash = memory_hashes.get(parent_id)
        child_hash = memory_hashes.get(child_id)
        if parent_hash is None or child_hash is None:
            continue
        side = "a" if env_id == env_a_id else "b"
        out[side].add((parent_hash, child_hash, relation))
    return out


def _set_by_env(rows: list[tuple[Any, str]], env_a_id: UUID, env_b_id: UUID) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"a": set(), "b": set()}
    for env_id, value in rows:
        if value is None:
            continue
        if env_id == env_a_id:
            out["a"].add(value)
        elif env_id == env_b_id:
            out["b"].add(value)
    return out


def _bounded_set_diff(a_values: set[str], b_values: set[str], limit: int) -> tuple[dict[str, list[str]], bool]:
    only_a = sorted(a_values - b_values)
    only_b = sorted(b_values - a_values)
    both = sorted(a_values & b_values)
    truncated = len(only_a) > limit or len(only_b) > limit or len(both) > limit
    return {
        "only_in_a": only_a[:limit],
        "only_in_b": only_b[:limit],
        "in_both": both[:limit],
    }, truncated


def _sampled_diff(a_values: set[tuple[Any, ...]], b_values: set[tuple[Any, ...]]) -> dict[str, Any]:
    only_a = sorted(a_values - b_values)
    only_b = sorted(b_values - a_values)
    both = a_values & b_values
    return {
        "only_in_a": {"count": len(only_a), "sample": [_sample_row(row) for row in only_a[:FULL_SAMPLE_LIMIT]]},
        "only_in_b": {"count": len(only_b), "sample": [_sample_row(row) for row in only_b[:FULL_SAMPLE_LIMIT]]},
        "in_both_count": len(both),
    }


def _sample_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {f"key_{index}": value for index, value in enumerate(row)}


__all__ = ["diff_envs"]
