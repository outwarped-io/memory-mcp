"""v0.10 memory statistics snapshot and scrape-time metric helpers."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any
from uuid import UUID

from memory_mcp_schemas.stats import (
    AccessCountStats,
    AgeStats,
    BucketStats,
    ChainDepthStats,
    DistributionStats,
    EnvMemoryStats,
    EnvStats,
    MemoriesStats,
    MemStatsRequest,
    MemStatsResponse,
    OutboxStats,
    PercentileStats,
    ProcessStats,
    ProjectionLagEntry,
    SubstrateStats,
    TagCount,
    TagsPerMemoryStats,
)
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.postgres import session_scope
from memory_mcp.identity import AgentContext

_STATEMENT_TIMEOUT_MS = 1_500
_PROCESS_START = time.monotonic()


@dataclass(slots=True)
class MetricsSnapshot:
    memory_counts: list[tuple[str, str, str, int]] = field(default_factory=list)
    pinned_by_env: dict[str, int] = field(default_factory=dict)
    body_bytes_by_env: dict[str, int] = field(default_factory=dict)
    tasks: dict[str, int] = field(default_factory=dict)
    playbooks: dict[str, int] = field(default_factory=dict)
    decisions: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[float]] = field(default_factory=dict)
    rss_bytes: int | None = None


__all__ = [
    "MemStatsRequest",
    "MemStatsResponse",
    "MetricsSnapshot",
    "compute_mem_stats",
    "compute_metrics_snapshot",
    "read_process_rss",
]


def _dedupe_env_ids(env_ids: Sequence[UUID] | None) -> list[UUID]:
    if not env_ids:
        return []
    return list(dict.fromkeys(env_ids))


def _scope_env_ids(req: MemStatsRequest, ctx: AgentContext) -> list[UUID] | None:
    if req.global_:
        rbac.require("admin", None, ctx)
        return None
    env_ids = _dedupe_env_ids(req.env_ids) or _dedupe_env_ids(ctx.attached_env_ids)
    for env_id in env_ids:
        rbac.require("read", env_id, ctx)
    return env_ids


def _sql_filter(env_ids: Sequence[UUID] | None, *, alias: str = "") -> tuple[str, dict[str, Any]]:
    if env_ids is None:
        return "", {}
    if not env_ids:
        return "WHERE false", {}
    prefix = f"{alias}." if alias else ""
    return f"WHERE {prefix}env_id IN :env_ids", {"env_ids": list(env_ids)}


def _bind_envs(stmt: Any, env_ids: Sequence[UUID] | None) -> Any:
    if env_ids:
        return stmt.bindparams(bindparam("env_ids", expanding=True))
    return stmt


async def _execute(
    session: AsyncSession, sql: str, params: dict[str, Any] | None = None, *, env_ids: Sequence[UUID] | None = None
):
    stmt = _bind_envs(text(sql), env_ids)
    return await session.execute(stmt, params or {})


def _percentiles(values: Iterable[float | int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"p50": None, "p90": None, "p99": None, "max": None}

    def pick(p: float) -> float | int:
        idx = max(0, min(len(ordered) - 1, ceil(p * len(ordered)) - 1))
        return ordered[idx]

    return {"p50": pick(0.50), "p90": pick(0.90), "p99": pick(0.99), "max": ordered[-1]}


def _chain_depth_stats(depths: Iterable[int]) -> ChainDepthStats:
    vals = list(depths)
    buckets = {"1": 0, "2": 0, "3": 0, "4+": 0}
    for depth in vals:
        buckets["4+" if depth >= 4 else str(depth)] += 1
    p = _percentiles(vals)
    return ChainDepthStats(buckets=buckets, p50=p["p50"], p90=p["p90"], p99=p["p99"], max=p["max"])


def _salience_stats(values: Iterable[float]) -> BucketStats:
    buckets = {"0.0-0.2": 0, "0.2-0.5": 0, "0.5-0.8": 0, "0.8-1.0": 0}
    for value in values:
        if value < 0.2:
            buckets["0.0-0.2"] += 1
        elif value < 0.5:
            buckets["0.2-0.5"] += 1
        elif value < 0.8:
            buckets["0.5-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    return BucketStats(buckets=buckets)


def _access_stats(values: Iterable[int]) -> AccessCountStats:
    vals = list(values)
    buckets = {"never": 0, "1-5": 0, "6-50": 0, "51+": 0}
    for value in vals:
        if value <= 0:
            buckets["never"] += 1
        elif value <= 5:
            buckets["1-5"] += 1
        elif value <= 50:
            buckets["6-50"] += 1
        else:
            buckets["51+"] += 1
    p = _percentiles(vals)
    return AccessCountStats(buckets=buckets, p50=p["p50"], p90=p["p90"], p99=p["p99"])


def _tags_per_memory_stats(values: Iterable[int], *, total_memories: int) -> TagsPerMemoryStats:
    vals = list(values)
    untagged = max(0, total_memories - len(vals))
    p = _percentiles(vals + ([0] * untagged))
    return TagsPerMemoryStats(p50=p["p50"], p90=p["p90"], p99=p["p99"], max=p["max"], untagged=untagged)


def read_process_rss() -> ProcessStats:
    if os.name != "posix" or not os.path.exists("/proc/self/statm"):
        return ProcessStats(
            rss_bytes=None, rss_reason="unsupported_os", uptime_seconds=time.monotonic() - _PROCESS_START
        )
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            parts = handle.read().split()
        rss_pages = int(parts[1])
        return ProcessStats(
            rss_bytes=rss_pages * os.sysconf("SC_PAGE_SIZE"),
            rss_reason=None,
            uptime_seconds=time.monotonic() - _PROCESS_START,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort observability
        return ProcessStats(rss_bytes=None, rss_reason=str(exc)[:120], uptime_seconds=time.monotonic() - _PROCESS_START)


async def _env_stats(session: AsyncSession, env_ids: Sequence[UUID] | None) -> tuple[EnvStats, dict[UUID, str]]:
    where, params = _sql_filter(env_ids)
    rows = (
        await _execute(session, f"SELECT id, name, status FROM environments {where}", params, env_ids=env_ids)
    ).all()
    names = {row.id: row.name for row in rows}
    active = sum(1 for row in rows if row.status == "active")
    deleted = sum(1 for row in rows if row.status == "deleted")
    return EnvStats(total=len(rows), active=active, deleted=deleted), names


async def _memory_stats(
    session: AsyncSession,
    env_ids: Sequence[UUID] | None,
    env_names: dict[UUID, str],
    *,
    include_body_bytes: bool,
    tag_top_k: int,
) -> MemoriesStats:
    where, params = _sql_filter(env_ids, alias="m")
    body_expr = "SUM(octet_length(m.body))" if include_body_bytes else "NULL"
    agg = (
        await _execute(
            session,
            f"""
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE m.status = 'active') AS active,
                   COUNT(*) FILTER (WHERE m.status = 'superseded') AS superseded,
                   COUNT(*) FILTER (WHERE m.status = 'retired') AS retired,
                   COUNT(*) FILTER (WHERE m.pinned) AS pinned,
                   {body_expr} AS body_bytes
            FROM memories m
            {where}
            """,
            params,
            env_ids=env_ids,
        )
    ).one()
    by_env_rows = (
        await _execute(
            session,
            f"""
            SELECT m.env_id, COUNT(*) AS count, {body_expr} AS body_bytes
            FROM memories m
            {where}
            GROUP BY m.env_id
            """,
            params,
            env_ids=env_ids,
        )
    ).all()
    by_env = {
        row.env_id: EnvMemoryStats(
            name=env_names.get(row.env_id),
            count=int(row.count),
            body_bytes=int(row.body_bytes) if row.body_bytes is not None else None,
        )
        for row in by_env_rows
    }
    by_kind = {
        str(row.kind): int(row.count)
        for row in (
            await _execute(
                session,
                f"SELECT m.kind, COUNT(*) AS count FROM memories m {where} GROUP BY m.kind",
                params,
                env_ids=env_ids,
            )
        ).all()
    }
    by_status = {
        str(row.status): int(row.count)
        for row in (
            await _execute(
                session,
                f"SELECT m.status, COUNT(*) AS count FROM memories m {where} GROUP BY m.status",
                params,
                env_ids=env_ids,
            )
        ).all()
    }
    top_tags: list[TagCount] = []
    if tag_top_k > 0:
        top_tags = [
            TagCount(tag=str(row.name), count=int(row.count))
            for row in (
                await _execute(
                    session,
                    f"""
                    SELECT t.name, COUNT(*) AS count
                    FROM memories m
                    JOIN memory_tags mt ON mt.memory_id = m.id AND mt.env_id = m.env_id
                    JOIN tags t ON t.id = mt.tag_id AND t.env_id = mt.env_id
                    {where}
                    GROUP BY t.name
                    ORDER BY count DESC, t.name ASC
                    LIMIT :limit
                    """,
                    {**params, "limit": tag_top_k},
                    env_ids=env_ids,
                )
            ).all()
        ]
    return MemoriesStats(
        total=int(agg.total or 0),
        active=int(agg.active or 0),
        superseded=int(agg.superseded or 0),
        retired=int(agg.retired or 0),
        pinned=int(agg.pinned or 0),
        total_body_bytes=int(agg.body_bytes) if agg.body_bytes is not None else None,
        by_env=by_env,
        by_kind=by_kind,
        by_status=by_status,
        top_tags=top_tags,
    )


async def _group_counts(
    session: AsyncSession, table_expr: str, status_expr: str, env_ids: Sequence[UUID] | None
) -> dict[str, int]:
    where, params = _sql_filter(env_ids)
    return {
        str(row.status): int(row.count)
        for row in (
            await _execute(
                session,
                f"SELECT {status_expr} AS status, COUNT(*) AS count FROM {table_expr} {where} GROUP BY {status_expr}",
                params,
                env_ids=env_ids,
            )
        ).all()
    }


async def _memory_kind_status_counts(
    session: AsyncSession,
    env_ids: Sequence[UUID] | None,
    *,
    kind: str,
) -> dict[str, int]:
    where, params = _sql_filter(env_ids)
    prefix = "AND" if where else "WHERE"
    return {
        str(row.status): int(row.count)
        for row in (
            await _execute(
                session,
                f"SELECT status, COUNT(*) AS count FROM memories {where} {prefix} kind = :kind GROUP BY status",
                {**params, "kind": kind},
                env_ids=env_ids,
            )
        ).all()
    }


async def _playbooks(session: AsyncSession, env_ids: Sequence[UUID] | None) -> dict[str, int]:
    return await _memory_kind_status_counts(session, env_ids, kind="playbook")


async def _decisions(session: AsyncSession, env_ids: Sequence[UUID] | None) -> dict[str, object]:
    by_status = await _memory_kind_status_counts(session, env_ids, kind="decision")
    return {"total": sum(by_status.values()), "by_status": by_status}


async def _distributions(
    session: AsyncSession, env_ids: Sequence[UUID] | None, total_memories: int
) -> DistributionStats:
    where, params = _sql_filter(env_ids)
    await session.execute(text(f"SET LOCAL statement_timeout = {_STATEMENT_TIMEOUT_MS}"))
    chain_rows = (
        await _execute(
            session,
            f"""
            WITH RECURSIVE chain(root_id, id, env_id, depth) AS (
                SELECT id, id, env_id, 1 FROM memories {where} AND superseded_by IS NULL
                UNION ALL
                SELECT c.root_id, m.id, m.env_id, c.depth + 1
                FROM chain c
                JOIN memories m ON m.superseded_by = c.id AND m.env_id = c.env_id
                WHERE c.depth < 1000
            ), depths AS (
                SELECT root_id, MAX(depth) AS depth FROM chain GROUP BY root_id
            )
            SELECT depth FROM depths
            """
            if where
            else """
            WITH RECURSIVE chain(root_id, id, env_id, depth) AS (
                SELECT id, id, env_id, 1 FROM memories WHERE superseded_by IS NULL
                UNION ALL
                SELECT c.root_id, m.id, m.env_id, c.depth + 1
                FROM chain c
                JOIN memories m ON m.superseded_by = c.id AND m.env_id = c.env_id
                WHERE c.depth < 1000
            ), depths AS (
                SELECT root_id, MAX(depth) AS depth FROM chain GROUP BY root_id
            )
            SELECT depth FROM depths
            """,
            params,
            env_ids=env_ids,
        )
    ).all()
    body_rows = (
        await _execute(session, f"SELECT octet_length(body) AS v FROM memories {where}", params, env_ids=env_ids)
    ).all()
    age_rows = (
        await _execute(
            session,
            f"SELECT EXTRACT(EPOCH FROM (now() - created_at)) AS v FROM memories {where} {'AND' if where else 'WHERE'} status = 'active'",
            params,
            env_ids=env_ids,
        )
    ).all()
    salience_rows = (
        await _execute(session, f"SELECT salience::float AS v FROM memories {where}", params, env_ids=env_ids)
    ).all()
    access_rows = (
        await _execute(session, f"SELECT access_count AS v FROM memories {where}", params, env_ids=env_ids)
    ).all()
    tag_where, tag_params = _sql_filter(env_ids, alias="mt")
    tag_rows = (
        await _execute(
            session,
            f"""
            SELECT mt.memory_id, COUNT(*) AS v
            FROM memory_tags mt
            {tag_where}
            GROUP BY mt.memory_id
            """,
            tag_params,
            env_ids=env_ids,
        )
    ).all()
    body_p = _percentiles([int(row.v) for row in body_rows])
    age_p = _percentiles([float(row.v) for row in age_rows])
    return DistributionStats(
        chain_depth=_chain_depth_stats([int(row.depth) for row in chain_rows]),
        body_length=PercentileStats(**body_p),
        age_seconds=AgeStats(p50=age_p["p50"], p90=age_p["p90"], p99=age_p["p99"], oldest=age_p["max"]),
        salience=_salience_stats([float(row.v) for row in salience_rows]),
        access_count=_access_stats([int(row.v) for row in access_rows]),
        tags_per_memory=_tags_per_memory_stats([int(row.v) for row in tag_rows], total_memories=total_memories),
    )


async def _projection_and_outbox(
    session: AsyncSession, env_ids: Sequence[UUID] | None
) -> tuple[list[ProjectionLagEntry], OutboxStats]:
    where, params = _sql_filter(env_ids)
    rows = (
        await _execute(
            session,
            f"SELECT sink, env_id, lag_seconds, last_event_id, status FROM projection_state {where}",
            params,
            env_ids=env_ids,
        )
    ).all()
    distinct_env_ids = {row.env_id for row in rows if row.env_id is not None}
    name_map: dict[UUID, str] = {}
    if distinct_env_ids:
        name_rows = (
            await session.execute(
                text("SELECT id, name FROM environments WHERE id = ANY(:ids)"),
                {"ids": list(distinct_env_ids)},
            )
        ).all()
        name_map = {r.id: r.name for r in name_rows}
    lag = [
        ProjectionLagEntry(
            sink=str(row.sink),
            env_id=row.env_id,
            env_name=name_map.get(row.env_id) if row.env_id is not None else None,
            lag_seconds=float(row.lag_seconds) if row.lag_seconds is not None else None,
            last_event_id=int(row.last_event_id) if row.last_event_id is not None else None,
            status=row.status,
        )
        for row in rows
    ]
    by_sink: dict[str, dict[str, int]] = {}
    for row in (
        await session.execute(text("SELECT sink, status, COUNT(*) AS count FROM outbox_delivery GROUP BY sink, status"))
    ).all():
        by_sink.setdefault(str(row.sink), {})[str(row.status)] = int(row.count)
    return lag, OutboxStats(by_sink=by_sink)


async def _substrate_snapshot(
    session: AsyncSession, env_ids: Sequence[UUID] | None, settings: Settings
) -> tuple[SubstrateStats, list[str]]:
    degraded: list[str] = []
    postgres: dict[str, int | str | None] | None = None
    qdrant: dict[str, int | str | None] | None = None
    neo4j: dict[str, int | str | None] | None = None
    try:
        size = (await session.execute(text("SELECT pg_database_size(current_database()) AS size"))).one().size
        postgres = {"db_size_bytes": int(size)}
    except Exception as exc:  # noqa: BLE001
        degraded.append("postgres")
        postgres = {"error": str(exc)[:200]}
    try:
        if settings.vector_backend != "qdrant":
            qdrant = {"status": "skipped"}
        else:
            from memory_mcp.db.vector.qdrant import QdrantVectorStore, _collection_name

            store = QdrantVectorStore(settings)
            try:
                points = 0
                target_envs = env_ids
                if target_envs is None:
                    env_rows = (await session.execute(text("SELECT id FROM environments"))).all()
                    target_envs = [row.id for row in env_rows]
                for env_id in target_envs:
                    result = await asyncio.wait_for(
                        store.client.count(collection_name=_collection_name(env_id), exact=True),
                        timeout=2.0,
                    )
                    points += int(getattr(result, "count", 0))
                qdrant = {"points": points}
            finally:
                await store.close()
    except Exception as exc:  # noqa: BLE001
        degraded.append("qdrant")
        qdrant = {"error": str(exc)[:200]}
    try:
        if settings.graph_backend != "neo4j":
            neo4j = {"status": "skipped"}
        else:
            from memory_mcp.db.graph.neo4j import Neo4jDriver

            drv = Neo4jDriver(settings)
            try:
                env_filter = "" if env_ids is None else "WHERE n.env_id IN $env_ids"
                params = {} if env_ids is None else {"env_ids": [str(e) for e in env_ids]}
                async with drv.driver.session() as neo_session:
                    nodes = await asyncio.wait_for(
                        (await neo_session.run(f"MATCH (n) {env_filter} RETURN count(n) AS count", **params)).single(),
                        timeout=2.0,
                    )
                    rels = await asyncio.wait_for(
                        (await neo_session.run("MATCH ()-[r]->() RETURN count(r) AS count")).single(),
                        timeout=2.0,
                    )
                neo4j = {
                    "nodes": int(nodes["count"] if nodes else 0),
                    "relationships": int(rels["count"] if rels else 0),
                }
            finally:
                await drv.close()
    except Exception as exc:  # noqa: BLE001
        degraded.append("neo4j")
        neo4j = {"error": str(exc)[:200]}
    return SubstrateStats(postgres=postgres, qdrant=qdrant, neo4j=neo4j), degraded


async def compute_mem_stats(
    req: MemStatsRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemStatsResponse:
    settings = settings or get_settings()
    env_ids = _scope_env_ids(req, ctx)
    if env_ids == []:
        return MemStatsResponse(process=read_process_rss())
    degraded_sections: list[str] = []
    async with session_scope() as session:
        await session.execute(text(f"SET LOCAL statement_timeout = {_STATEMENT_TIMEOUT_MS}"))
        envs, env_names = await _env_stats(session, env_ids)
        try:
            memories = await _memory_stats(
                session,
                env_ids,
                env_names,
                include_body_bytes=req.include_body_bytes,
                tag_top_k=req.tag_top_k,
            )
        except Exception:
            await session.rollback()
            memories = MemoriesStats(total_body_bytes=None, total_body_bytes_approximate=True)
            degraded_sections.append("memories")
        tasks = await _group_counts(session, "tasks", "status", env_ids)
        playbooks = await _playbooks(session, env_ids)
        decisions = await _decisions(session, env_ids)
        distributions: DistributionStats | None = None
        if req.include_distributions:
            try:
                distributions = await _distributions(session, env_ids, memories.total)
            except Exception:
                await session.rollback()
                degraded_sections.append("distributions")
        projection_lag, outbox = await _projection_and_outbox(session, env_ids)
        substrate = None
        degraded_substrates: list[str] = []
        if req.include_substrates:
            substrate, degraded_substrates = await _substrate_snapshot(session, env_ids, settings)
    return MemStatsResponse(
        memories=memories,
        envs=envs,
        distributions=distributions,
        tasks=tasks,
        playbooks=playbooks,
        decisions=decisions,
        substrate=substrate,
        projection_lag=projection_lag,
        outbox=outbox,
        process=read_process_rss(),
        degraded_substrates=degraded_substrates,
        degraded_sections=degraded_sections,
    )


async def compute_metrics_snapshot(session: AsyncSession) -> MetricsSnapshot:
    stats = await compute_mem_stats(
        MemStatsRequest(global_=True, include_substrates=False, include_body_bytes=True, include_distributions=True),
        ctx=AgentContext(agent_id=UUID("00000000-0000-0000-0000-000000000000")),
    )
    snapshot = MetricsSnapshot(rss_bytes=stats.process.rss_bytes)
    for env_id, env_stats in stats.memories.by_env.items():
        env_label = env_stats.name or str(env_id)
        snapshot.pinned_by_env[env_label] = stats.memories.pinned
        if env_stats.body_bytes is not None:
            snapshot.body_bytes_by_env[env_label] = env_stats.body_bytes
    for kind, count in stats.memories.by_kind.items():
        # Status-specific counts are populated by a lighter SQL query in observability; keep global fallback here.
        snapshot.memory_counts.append(("_all", kind, "_all", count))
    snapshot.tasks = dict(stats.tasks)
    snapshot.playbooks = dict(stats.playbooks)
    snapshot.decisions = (
        dict(stats.decisions.get("by_status", {})) if isinstance(stats.decisions.get("by_status"), dict) else {}
    )
    if stats.distributions:
        d = stats.distributions
        if d.chain_depth:
            snapshot.samples["chain_depth"] = [
                float(v)
                for label, count in d.chain_depth.buckets.items()
                for v in ([4.0 if label == "4+" else float(label)] * count)
            ]
        # Histogram samples use raw values from direct SQL below for better fidelity.
    return snapshot
