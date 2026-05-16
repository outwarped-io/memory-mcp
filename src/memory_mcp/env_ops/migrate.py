"""Filtered bulk memory migration between environments."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp.db.models import Environment, Memory, MemoryTag, Tag
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import MemoryStatus
from memory_mcp.errors import InvalidInputError, MemoryMCPError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.memories import mem_copy, mem_move

from memory_mcp_schemas.browse import MemBrowseRequest
from memory_mcp_schemas.env_ops import (
    BatchFailure,
    EnvMigrateRequest,
    EnvMigrateResponse,
    MemCopyRequest,
    MemMoveRequest,
    MigrationMode,
)


async def migrate_env(request: EnvMigrateRequest, *, ctx: AgentContext) -> EnvMigrateResponse:
    """Bulk migration of memories from src env to dst env via mem_copy or mem_move.

    Selects matching memories by BrowseFilter (or ALL if filter is None), then applies
    mem_copy (mode=copy) or mem_move (mode=move) to each. Reports per-memory success/failure.
    Atomicity is per-memory, NOT per-batch — partial successes are possible. The report
    enumerates which memories succeeded and which failed and why.
    """

    await _validate_request(request, ctx=ctx)

    async with session_scope() as session:
        seed_ids = await _seed_memory_ids(session, request)
        original_superseded_by = await _superseded_by_map(session, seed_ids)
        ordered_ids = list(seed_ids)
        if request.preserve_supersession_chain:
            chain_ids = await _expand_supersession_chain(
                session,
                env_id=request.src_env_id,
                memory_ids=set(seed_ids),
            )
            original_superseded_by = await _superseded_by_map(session, chain_ids)
            ordered_ids = await _order_for_chain(session, env_id=request.src_env_id, memory_ids=chain_ids)

    closure_inclusions = max(0, len(set(ordered_ids)) - len(set(seed_ids)))
    remap: dict[UUID, UUID] = {}
    failures: list[BatchFailure] = []
    pending_vector_rebuild = 0

    for memory_id in ordered_ids:
        try:
            if request.mode == MigrationMode.copy:
                out = await mem_copy(
                    MemCopyRequest(
                        memory_id=memory_id,
                        dst_env_id=request.dst_env_id,
                        copy_tags=request.copy_tags,
                        copy_provenance=request.copy_provenance,
                        create_lineage_edge=request.create_lineage_edges,
                        re_embed_if_model_mismatch=request.re_embed_if_model_mismatch,
                        preserve_timestamps=request.preserve_timestamps,
                    ),
                    ctx=ctx,
                )
            else:
                out = await mem_move(
                    MemMoveRequest(
                        memory_id=memory_id,
                        dst_env_id=request.dst_env_id,
                        copy_tags=request.copy_tags,
                        copy_provenance=request.copy_provenance,
                        create_lineage_edge=request.create_lineage_edges,
                        re_embed_if_model_mismatch=request.re_embed_if_model_mismatch,
                        preserve_timestamps=request.preserve_timestamps,
                        redirect_source=True,
                    ),
                    ctx=ctx,
                )
            remap[memory_id] = out.dst_memory_id
            pending_vector_rebuild += out.pending_vector_rebuild
        except Exception as exc:
            if request.fail_fast:
                raise
            failures.append(_batch_failure(memory_id, exc))

    if request.preserve_supersession_chain and remap:
        await _restore_dst_supersession_chain(
            remap=remap,
            original_superseded_by=original_superseded_by,
        )

    truncated = len(failures) > 100
    return EnvMigrateResponse(
        src_env_id=request.src_env_id,
        dst_env_id=request.dst_env_id,
        mode=request.mode,
        attempted=len(ordered_ids),
        succeeded=len(remap),
        failed=len(failures),
        remap=remap,
        failures=failures[:100],
        truncated=truncated,
        pending_vector_rebuild=pending_vector_rebuild,
        closure_inclusions=closure_inclusions,
    )


async def _validate_request(request: EnvMigrateRequest, *, ctx: AgentContext) -> None:
    if request.src_env_id == request.dst_env_id:
        raise InvalidInputError("src_env_id and dst_env_id must differ")

    src, dst = await _load_env_pair(request.src_env_id, request.dst_env_id)
    rbac.require("read", request.src_env_id, ctx)
    rbac.require("write", request.dst_env_id, ctx)
    if request.mode == MigrationMode.move:
        rbac.require("write", request.src_env_id, ctx)

    if (
        src.default_embedding_model_id != dst.default_embedding_model_id
        and not request.re_embed_if_model_mismatch
    ):
        exc = InvalidInputError(
            "source and destination default embedding models differ",
            src_env_id=str(request.src_env_id),
            dst_env_id=str(request.dst_env_id),
            src_default_embedding_model_id=src.default_embedding_model_id,
            dst_default_embedding_model_id=dst.default_embedding_model_id,
        )
        exc.code = "EMBEDDING_MODEL_MISMATCH"
        raise exc


async def _load_env_pair(src_env_id: UUID, dst_env_id: UUID) -> tuple[Environment, Environment]:
    async with session_scope() as session:
        rows = (await session.execute(
            select(Environment).where(Environment.id.in_([src_env_id, dst_env_id])),
        )).scalars().all()

    by_id = {row.id: row for row in rows}
    for env_id in (src_env_id, dst_env_id):
        env = by_id.get(env_id)
        if env is None:
            raise NotFoundError(f"environment {env_id} not found", env_id=str(env_id))
        if getattr(env, "status", "active") == "deleted":
            exc = NotFoundError(f"environment {env_id} is deleted", env_id=str(env_id))
            exc.code = "ENV_DELETED"
            raise exc
    return by_id[src_env_id], by_id[dst_env_id]


async def _seed_memory_ids(session: AsyncSession, request: EnvMigrateRequest) -> list[UUID]:
    if request.filter is None:
        statuses = [MemoryStatus.active.value]
        if request.include_superseded:
            statuses.append(MemoryStatus.superseded.value)
        rows = await session.execute(
            select(Memory.id)
            .where(Memory.env_id == request.src_env_id)
            .where(Memory.status.in_(statuses))
            .order_by(Memory.created_at.asc(), Memory.id.asc()),
        )
        return list(rows.scalars().all())

    browse_filter = _filter_for_src(request.filter, request.src_env_id, request.include_superseded)
    statuses = [s.value for s in browse_filter.statuses] if browse_filter.statuses else [MemoryStatus.active.value]
    stmt = (
        select(Memory.id)
        .where(Memory.env_id == request.src_env_id)
        .where(Memory.status.in_(statuses))
    )
    if browse_filter.kinds:
        stmt = stmt.where(Memory.kind.in_([k.value for k in browse_filter.kinds]))
    if browse_filter.created_after is not None:
        stmt = stmt.where(Memory.created_at >= browse_filter.created_after)
    if browse_filter.created_before is not None:
        stmt = stmt.where(Memory.created_at < browse_filter.created_before)
    if browse_filter.updated_after is not None:
        stmt = stmt.where(Memory.updated_at >= browse_filter.updated_after)
    if browse_filter.tags:
        stmt = stmt.where(
            select(MemoryTag.memory_id)
            .join(Tag, Tag.id == MemoryTag.tag_id)
            .where(
                MemoryTag.env_id == request.src_env_id,
                MemoryTag.memory_id == Memory.id,
                Tag.name.in_(browse_filter.tags),
            )
            .exists(),
        )
    rows = await session.execute(stmt.order_by(Memory.created_at.asc(), Memory.id.asc()))
    return list(rows.scalars().all())


def _filter_for_src(raw: MemBrowseRequest, src_env_id: UUID, include_superseded: bool) -> MemBrowseRequest:
    statuses = raw.statuses
    if statuses is None and include_superseded:
        statuses = [MemoryStatus.active, MemoryStatus.superseded]
    return raw.model_copy(update={"env_ids": [src_env_id], "cursor": None, "statuses": statuses})


async def _expand_supersession_chain(
    session: AsyncSession,
    *,
    env_id: UUID,
    memory_ids: set[UUID],
) -> set[UUID]:
    expanded = set(memory_ids)
    frontier = set(memory_ids)
    while frontier:
        rows = (await session.execute(
            select(Memory.id, Memory.superseded_by)
            .where(Memory.env_id == env_id)
            .where((Memory.id.in_(frontier)) | (Memory.superseded_by.in_(frontier))),
        )).all()
        discovered: set[UUID] = set()
        for memory_id, superseded_by in rows:
            discovered.add(memory_id)
            if superseded_by is not None:
                discovered.add(superseded_by)
        frontier = discovered - expanded
        expanded.update(discovered)
    return expanded


async def _order_for_chain(session: AsyncSession, *, env_id: UUID, memory_ids: Iterable[UUID]) -> list[UUID]:
    ids = set(memory_ids)
    if not ids:
        return []
    rows = (await session.execute(
        select(Memory.id, Memory.superseded_by, Memory.created_at)
        .where(Memory.env_id == env_id)
        .where(Memory.id.in_(ids)),
    )).all()
    superseded_by = {memory_id: target for memory_id, target, _created_at in rows}
    created_order = {memory_id: index for index, (memory_id, _target, _created_at) in enumerate(
        sorted(rows, key=lambda row: (row[2], row[0])),
    )}

    def depth(memory_id: UUID) -> int:
        seen: set[UUID] = set()
        cur = memory_id
        count = 0
        while cur not in seen and (nxt := superseded_by.get(cur)) in ids:
            seen.add(cur)
            cur = nxt
            count += 1
        return count

    return sorted(ids, key=lambda memory_id: (-depth(memory_id), created_order.get(memory_id, 0), memory_id))


async def _superseded_by_map(session: AsyncSession, memory_ids: Iterable[UUID]) -> dict[UUID, UUID | None]:
    ids = set(memory_ids)
    if not ids:
        return {}
    rows = await session.execute(select(Memory.id, Memory.superseded_by).where(Memory.id.in_(ids)))
    return {memory_id: superseded_by for memory_id, superseded_by in rows.all()}


async def _restore_dst_supersession_chain(
    *,
    remap: dict[UUID, UUID],
    original_superseded_by: dict[UUID, UUID | None],
) -> None:
    async with session_scope() as session:
        for src_id, dst_id in remap.items():
            src_target = original_superseded_by.get(src_id)
            if src_target is None or src_target not in remap:
                continue
            await session.execute(
                update(Memory)
                .where(Memory.id == dst_id)
                .values(status=MemoryStatus.superseded.value, superseded_by=remap[src_target]),
            )


def _batch_failure(memory_id: UUID, exc: Exception) -> BatchFailure:
    code = getattr(exc, "code", exc.__class__.__name__)
    if isinstance(exc, MemoryMCPError):
        code = exc.code
    return BatchFailure(
        id=str(memory_id),
        error_code=str(code),
        memory_id=memory_id,
        code=str(code),
        message=str(exc),
    )
