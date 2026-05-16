"""Entity tools — canonical entity + alias management for v1.

Entities are first-class records (people, services, repos, machines,
documents). Each entity lives in exactly one environment and is identified
by its canonical name. Aliases are a separate row type so the same entity
can be discovered via multiple lexical forms.

Phase 1 surface:

* :func:`entity_upsert` — create-or-update by ``(env_id, normalized_name)``;
  also accepts a list of aliases that are upserted idempotently.
* :func:`entity_resolve` — case/punctuation-normalized lookup by name or
  alias across one or more environments.
* :func:`entity_merge` — admin-only: merge ``merge_ids`` into ``keep_id``,
  rewiring aliases + graph_nodes, deleting the merged rows.

Outbox routing for ``aggregate_type=entity`` is determined by the
``graph_backend`` setting — when ``GRAPH_BACKEND=postgres`` it's a no-op
(no row written). v1.5 will project to Neo4j.

Name normalization
------------------

``_normalize_name`` runs NFKC unicode normalization, lowercases, strips
punctuation (anything not ``[\\w\\s]``), and collapses internal whitespace.
The same routine is used for canonical names and aliases — so an alias
with the same normalized form as the entity's canonical name is treated
as a no-op (already matched).

Optimistic concurrency
----------------------

Same pattern as ``memory_update``: SELECT, version-check, UPDATE WHERE
version=:expected_version, raise ``VersionConflictError`` on rowcount=0.
Idempotent re-upserts (same kind/canonical_name/metadata) skip the UPDATE
and don't bump version.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import unicodedata
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Select, delete, func, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from memory_mcp import rbac
from memory_mcp.config import Settings, get_settings
from memory_mcp.db.models import (
    AuditLog,
    Entity,
    EntityAlias,
    GraphNode,
    Relation,
)
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import OutboxAggregateType, OutboxOp
from memory_mcp.errors import (
    EnvAmbiguousError,
    InvalidCursorError,
    InvalidInputError,
    NotFoundError,
    VersionConflictError,
)
from memory_mcp.identity import AgentContext
from memory_mcp.pagination import (
    Direction,
    compute_filter_fingerprint,
    decode_cursor,
    encode_cursor,
)

from memory_mcp_schemas.entities import (
    _normalize_name,
    EntityBrowseRequest,
    EntityBrowseResponse,
    EntityMergeRequest,
    EntityResolveRequest,
    EntityResponse,
    EntityUpsertRequest,
)

log = logging.getLogger(__name__)

__all__ = [
    "EntityBrowseRequest",
    "EntityBrowseResponse",
    "EntityMergeRequest",
    "EntityResolveRequest",
    "EntityResponse",
    "EntityUpsertRequest",
    "entity_browse",
    "entity_merge",
    "entity_resolve",
    "entity_upsert",
]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_env_id(*, explicit: UUID | None, ctx: AgentContext) -> UUID:
    if explicit is not None:
        return explicit
    attached = list(dict.fromkeys(ctx.attached_env_ids))
    if len(attached) == 1:
        return attached[0]
    raise EnvAmbiguousError(
        "no unambiguous env to write to; pass env_id or attach exactly one env",
        attached=[str(e) for e in attached],
    )


async def _load_aliases(session, entity_id: UUID) -> list[str]:
    rows = await session.execute(
        select(EntityAlias.alias)
        .where(EntityAlias.entity_id == entity_id)
        .order_by(EntityAlias.alias)
    )
    return [a for (a,) in rows.all()]


def _entity_payload(entity: Entity, aliases: list[str]) -> dict[str, Any]:
    return {
        "entity_id": str(entity.id),
        "env_id": str(entity.env_id),
        "kind": entity.kind,
        "canonical_name": entity.canonical_name,
        "normalized_name": entity.normalized_name,
        "aliases": aliases,
        "metadata": dict(entity.metadata_ or {}),
        "version": entity.version,
        "created_at": entity.created_at.isoformat() if entity.created_at else None,
        "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
    }


def _entity_to_response(entity: Entity, aliases: list[str]) -> EntityResponse:
    return EntityResponse(
        id=entity.id,
        env_id=entity.env_id,
        kind=entity.kind,
        canonical_name=entity.canonical_name,
        normalized_name=entity.normalized_name,
        aliases=list(aliases),
        metadata=dict(entity.metadata_ or {}),
        version=entity.version,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


async def _record_entity_audit(
    session,
    *,
    op: str,
    entity_id: UUID,
    env_id: UUID,
    by_agent_id: UUID | None,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> None:
    audit = AuditLog(
        record_type="entity",
        record_id=entity_id,
        env_id=env_id,
        op=op,
        by_agent_id=by_agent_id,
        before=before,
        after=after,
    )
    session.add(audit)
    await session.flush()


# ---------------------------------------------------------------------------
# entity_upsert
# ---------------------------------------------------------------------------


async def entity_upsert(
    request: EntityUpsertRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> EntityResponse:
    """Create or update an entity by canonical name (normalized, env-scoped).

    Behavior:

    * No existing row in this env with the same normalized canonical name
      → INSERT new entity (version=1) + alias rows + audit (op=create) +
      outbox (op=upsert).
    * Existing row found:

      - If ``expected_version`` is provided and mismatches the current
        version → :class:`VersionConflictError`.
      - If any of ``kind`` / ``canonical_name`` / ``metadata`` differ
        from the stored row → UPDATE WHERE version=:expected; bump
        version; emit outbox (op=update).
      - If nothing differs → no UPDATE, no version bump, no outbox row.
        (Idempotent re-upsert.)

    Aliases are always idempotently upserted (``ON CONFLICT DO NOTHING``)
    against ``UNIQUE(env_id, normalized_alias)``. Aliases whose
    normalized form equals the canonical's normalized form are skipped.
    """
    settings = settings or get_settings()
    env_id = _resolve_env_id(explicit=request.env_id, ctx=ctx)
    rbac.require("write", env_id, ctx)

    normalized_canonical = _normalize_name(request.canonical_name)

    async with session_scope() as s:
        existing = (await s.execute(
            select(Entity).where(
                Entity.env_id == env_id,
                Entity.normalized_name == normalized_canonical,
            )
        )).scalar_one_or_none()

        is_create = existing is None
        emit_outbox = False
        outbox_op = OutboxOp.upsert

        if existing is None:
            entity = Entity(
                env_id=env_id,
                kind=request.kind,
                canonical_name=request.canonical_name,
                normalized_name=normalized_canonical,
                metadata_=request.metadata,
            )
            s.add(entity)
            await s.flush()
            await s.refresh(entity)
            emit_outbox = True
            outbox_op = OutboxOp.upsert
        else:
            entity = existing
            if (
                request.expected_version is not None
                and entity.version != request.expected_version
            ):
                raise VersionConflictError(
                    expected=request.expected_version,
                    actual=entity.version,
                )
            updates: dict[Any, Any] = {}
            if entity.kind != request.kind:
                updates[Entity.kind] = request.kind
            if entity.canonical_name != request.canonical_name:
                updates[Entity.canonical_name] = request.canonical_name
            if dict(entity.metadata_ or {}) != request.metadata:
                # Use ORM attribute as key to avoid collision with the
                # `metadata` MetaData attribute that SQLAlchemy reserves.
                updates[Entity.metadata_] = request.metadata

            if updates:
                expected_v = entity.version
                updates[Entity.version] = expected_v + 1
                updates[Entity.updated_at] = func.now()
                result = await s.execute(
                    update(Entity)
                    .where(Entity.id == entity.id, Entity.version == expected_v)
                    .values(updates)
                )
                if result.rowcount == 0:  # type: ignore[attr-defined]
                    raise VersionConflictError(
                        expected=expected_v, actual=expected_v + 1
                    )
                await s.refresh(entity)
                emit_outbox = True
                outbox_op = OutboxOp.update

        # Aliases — idempotent upsert; skip aliases that normalize to the canonical.
        for alias in request.aliases:
            normalized = _normalize_name(alias)
            if normalized == normalized_canonical:
                continue
            await s.execute(
                pg_insert(EntityAlias.__table__)
                .values(
                    entity_id=entity.id,
                    env_id=env_id,
                    alias=alias,
                    normalized_alias=normalized,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        EntityAlias.__table__.c.entity_id,
                        EntityAlias.__table__.c.normalized_alias,
                    ]
                )
            )

        aliases_now = await _load_aliases(s, entity.id)

        # Audit
        before_snap = None if is_create else _entity_payload(existing, aliases_now)
        after_snap = _entity_payload(entity, aliases_now)
        if is_create or emit_outbox or before_snap != after_snap:
            await _record_entity_audit(
                s,
                op="create" if is_create else "update",
                entity_id=entity.id,
                env_id=env_id,
                by_agent_id=ctx.agent_id,
                before=before_snap,
                after=after_snap,
            )

        if emit_outbox:
            await enqueue_event(
                s,
                aggregate_type=OutboxAggregateType.entity,
                aggregate_id=entity.id,
                aggregate_version=entity.version,
                env_id=env_id,
                op=outbox_op,
                payload=after_snap,
                settings=settings,
            )

    return _entity_to_response(entity, aliases_now)


# ---------------------------------------------------------------------------
# entity_resolve
# ---------------------------------------------------------------------------


async def entity_resolve(
    request: EntityResolveRequest,
    *,
    ctx: AgentContext,
) -> list[EntityResponse]:
    """Find entities whose canonical name OR any alias matches the input.

    Matching is on normalized form (case/punctuation-insensitive).
    Search scope:

    * ``env_ids`` if provided (subset of caller's attached envs in v1.5).
    * Otherwise ``ctx.attached_env_ids``.
    * If both are empty, returns ``[]`` (no envs to search).

    Results sorted by ``(env_id, canonical_name)`` and capped at ``limit``.
    """
    normalized = _normalize_name(request.name)
    if not normalized:
        return []

    env_ids = request.env_ids if request.env_ids is not None else list(ctx.attached_env_ids)
    if not env_ids:
        return []

    for eid in env_ids:
        rbac.require("read", eid, ctx)

    async with session_scope() as s:
        # Match entities by canonical normalized name OR by alias.
        stmt = (
            select(Entity)
            .distinct()
            .outerjoin(EntityAlias, EntityAlias.entity_id == Entity.id)
            .where(
                Entity.env_id.in_(env_ids),
                (Entity.normalized_name == normalized)
                | (EntityAlias.normalized_alias == normalized),
            )
        )
        if request.kinds:
            stmt = stmt.where(Entity.kind.in_(request.kinds))
        stmt = stmt.order_by(Entity.env_id, Entity.canonical_name).limit(request.limit)

        rows = (await s.execute(stmt)).scalars().all()
        # Hydrate aliases
        out: list[EntityResponse] = []
        for entity in rows:
            aliases = await _load_aliases(s, entity.id)
            out.append(_entity_to_response(entity, aliases))
        return out


# ---------------------------------------------------------------------------
# entity_merge
# ---------------------------------------------------------------------------


def _plan_relation_node_repoint(
    relations: list[Relation],
    *,
    existing_keys: set[tuple[UUID, UUID, str]],
    from_node_id: UUID,
    to_node_id: UUID,
) -> tuple[list[UUID], dict[UUID, tuple[UUID, UUID]]]:
    """Plan relation deletes/updates for a graph-node merge."""
    delete_ids: list[UUID] = []
    move_values: dict[UUID, tuple[UUID, UUID]] = {}
    seen_new_keys: set[tuple[UUID, UUID, str]] = set()
    for relation in relations:
        new_src = to_node_id if relation.src_node_id == from_node_id else relation.src_node_id
        new_dst = to_node_id if relation.dst_node_id == from_node_id else relation.dst_node_id
        new_key = (new_src, new_dst, relation.type)
        if new_key in existing_keys or new_key in seen_new_keys:
            delete_ids.append(relation.id)
        else:
            seen_new_keys.add(new_key)
            move_values[relation.id] = (new_src, new_dst)
    return delete_ids, move_values


async def _merge_entity_graph_nodes(
    session,
    *,
    env_id: UUID,
    keep_id: UUID,
    merge_ids: list[UUID],
) -> None:
    """Move merge-side entity graph nodes onto ``keep_id`` without collisions."""
    keep_node = (await session.execute(
        select(GraphNode).where(GraphNode.entity_id == keep_id)
    )).scalar_one_or_none()

    for merge_id in merge_ids:
        merge_node = (await session.execute(
            select(GraphNode).where(GraphNode.entity_id == merge_id)
        )).scalar_one_or_none()
        if merge_node is None:
            continue

        if keep_node is None:
            merge_node.entity_id = keep_id
            keep_node = merge_node
            await session.flush()
            continue

        relations = (await session.execute(
            select(Relation)
            .where(
                Relation.env_id == env_id,
                or_(
                    Relation.src_node_id == merge_node.id,
                    Relation.dst_node_id == merge_node.id,
                ),
            )
            .order_by(Relation.created_at, Relation.id)
        )).scalars().all()

        relation_ids = [relation.id for relation in relations]
        existing_stmt = select(
            Relation.src_node_id,
            Relation.dst_node_id,
            Relation.type,
        ).where(Relation.env_id == env_id)
        if relation_ids:
            existing_stmt = existing_stmt.where(Relation.id.notin_(relation_ids))
        existing_keys = set((await session.execute(existing_stmt)).all())
        delete_ids, move_values = _plan_relation_node_repoint(
            relations,
            existing_keys=existing_keys,
            from_node_id=merge_node.id,
            to_node_id=keep_node.id,
        )

        if delete_ids:
            await session.execute(delete(Relation).where(Relation.id.in_(delete_ids)))
            await session.flush()

        for relation_id, (new_src, new_dst) in move_values.items():
            await session.execute(
                update(Relation)
                .where(Relation.id == relation_id)
                .values(src_node_id=new_src, dst_node_id=new_dst, updated_at=func.now())
            )

        await session.delete(merge_node)
        await session.flush()


async def entity_merge(
    request: EntityMergeRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> EntityResponse:
    """Merge ``merge_ids`` entities into ``keep_id``.

    Behavior (single transaction):

    1. Load all involved entities; verify each ``expected_version``.
    2. All entities must share the same ``env_id``. (Cross-env merge is
       admin-only and deferred — same posture as cross-env supersede.)
    3. Aliases of merged entities are re-pointed to ``keep_id``. Conflicts
       on ``UNIQUE(env_id, normalized_alias)`` are resolved by **dropping
       the duplicate** (keep_id already has that alias).
    4. ``graph_nodes`` for merged entities are either re-pointed to
       ``keep_id`` when keep lacks a node, or have their relations moved
       onto keep's node before deleting the merged graph node.
    5. Merged entity rows are DELETEd.
    6. ``keep_id``'s version is bumped (transformative change → outbox
       update); each merged entity emits an outbox tombstone.
    7. Audit log: one ``merge`` entry per merged id (before=merged,
       after=null) + one ``update`` for keep.

    Returns the post-merge :class:`EntityResponse` for ``keep_id``.
    """
    settings = settings or get_settings()
    rbac.require("admin", env_id=None, ctx=ctx)

    if request.keep_id in request.merge_ids:
        raise ValueError("keep_id cannot appear in merge_ids")

    expected = dict(request.expected_versions)
    required_ids = {request.keep_id, *request.merge_ids}
    missing = required_ids - expected.keys()
    if missing:
        raise ValueError(
            f"expected_versions missing entries for: {sorted(str(m) for m in missing)}"
        )

    async with session_scope() as s:
        rows = (await s.execute(
            select(Entity).where(Entity.id.in_(required_ids))
        )).scalars().all()
        by_id = {e.id: e for e in rows}

        for eid in required_ids:
            if eid not in by_id:
                raise NotFoundError(f"entity {eid} not found", entity_id=str(eid))
            if by_id[eid].version != expected[eid]:
                raise VersionConflictError(
                    expected=expected[eid], actual=by_id[eid].version
                )

        keep = by_id[request.keep_id]
        env_id = keep.env_id

        # Same-env constraint
        for mid in request.merge_ids:
            if by_id[mid].env_id != env_id:
                raise ValueError(
                    f"cross-env merge not allowed (entity {mid} in env "
                    f"{by_id[mid].env_id}, keep in env {env_id})"
                )
            rbac.require("write", env_id, ctx)

        # Pre-merge snapshots for audit
        merged_before: dict[UUID, dict[str, Any]] = {}
        for mid in request.merge_ids:
            aliases = await _load_aliases(s, mid)
            merged_before[mid] = _entity_payload(by_id[mid], aliases)
        keep_before = _entity_payload(keep, await _load_aliases(s, keep.id))

        # Re-point aliases. Use the canonical normalized form of keep to
        # also drop any alias-of-merged that equals keep's canonical (it'd
        # be redundant).
        keep_normalized = keep.normalized_name
        # Step A: drop alias rows on merge_ids whose normalized form would
        # collide with an alias already on keep_id (or is keep's canonical).
        existing_keep_aliases = (await s.execute(
            select(EntityAlias.normalized_alias).where(EntityAlias.entity_id == keep.id)
        )).scalars().all()
        existing_keep_set = set(existing_keep_aliases) | {keep_normalized}
        if existing_keep_set:
            await s.execute(
                delete(EntityAlias).where(
                    EntityAlias.entity_id.in_(request.merge_ids),
                    EntityAlias.normalized_alias.in_(existing_keep_set),
                )
            )
        # Step B: re-point remaining merge alias rows.
        await s.execute(
            update(EntityAlias)
            .where(EntityAlias.entity_id.in_(request.merge_ids))
            .values(entity_id=keep.id)
        )

        await _merge_entity_graph_nodes(
            s,
            env_id=env_id,
            keep_id=keep.id,
            merge_ids=request.merge_ids,
        )

        # Bump keep version (transformative); persist via optimistic UPDATE.
        new_version = keep.version + 1
        result = await s.execute(
            update(Entity)
            .where(Entity.id == keep.id, Entity.version == keep.version)
            .values(version=new_version, updated_at=func.now())
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]
            raise VersionConflictError(
                expected=keep.version, actual=keep.version + 1
            )
        await s.refresh(keep)

        # DELETE merged entity rows (cascades remaining aliases / graph_nodes).
        await s.execute(delete(Entity).where(Entity.id.in_(request.merge_ids)))

        # Audit + outbox per merged
        for mid in request.merge_ids:
            await _record_entity_audit(
                s,
                op="merge",
                entity_id=mid,
                env_id=env_id,
                by_agent_id=ctx.agent_id,
                before=merged_before[mid],
                after={"merged_into": str(keep.id)},
            )
            # We use the merged entity's PRE-merge version as aggregate_version
            # (the version they were last seen at). This keeps outbox version
            # monotonic-per-aggregate.
            await enqueue_event(
                s,
                aggregate_type=OutboxAggregateType.entity,
                aggregate_id=mid,
                aggregate_version=expected[mid] + 1,
                env_id=env_id,
                op=OutboxOp.tombstone,
                payload={
                    "entity_id": str(mid),
                    "env_id": str(env_id),
                    "merged_into": str(keep.id),
                },
                settings=settings,
            )

        # Audit + outbox for keep
        keep_aliases_after = await _load_aliases(s, keep.id)
        keep_after = _entity_payload(keep, keep_aliases_after)
        await _record_entity_audit(
            s,
            op="update",
            entity_id=keep.id,
            env_id=env_id,
            by_agent_id=ctx.agent_id,
            before=keep_before,
            after=keep_after,
        )
        await enqueue_event(
            s,
            aggregate_type=OutboxAggregateType.entity,
            aggregate_id=keep.id,
            aggregate_version=keep.version,
            env_id=env_id,
            op=OutboxOp.update,
            payload=keep_after,
            settings=settings,
        )

    return _entity_to_response(keep, keep_aliases_after)


# ---------------------------------------------------------------------------
# entity_browse (Sprint A exploration API)
# ---------------------------------------------------------------------------


from typing import Any as _Any  # noqa: E402 — local alias for forward-compat with future Select typing

"""Sort key for :func:`entity_browse`. ``updated_at`` is NOT supported
because the ``entities`` table does not track per-row update timestamps
distinct from ``created_at`` for our purposes (alias additions don't
touch the parent row's updated_at). Caller-visible chronology is
``created_at`` only.
"""


def _resolve_browse_env_ids(
    explicit: list[UUID] | None,
    ctx: AgentContext,
) -> list[UUID]:
    if explicit:
        return list(dict.fromkeys(explicit))
    return list(dict.fromkeys(ctx.attached_env_ids))


def _entity_browse_filter_dict(
    req: EntityBrowseRequest, env_ids: list[UUID], normalized_prefix: str | None,
) -> dict[str, _Any]:
    return {
        "env_ids": list(env_ids),
        "kinds": sorted(req.kinds) if req.kinds else None,
        "normalized_prefix": normalized_prefix,
        "order_by": req.order_by,
        "descending": req.descending,
    }


def _apply_entity_keyset(
    stmt: "Select[_Any]",
    *,
    order_by: EntityOrderField,
    descending: bool,
    cursor_value: _Any,
    cursor_id: UUID | None,
) -> "Select[_Any]":
    order_col = (
        Entity.canonical_name if order_by == "canonical_name" else Entity.created_at
    )
    if cursor_value is not None and cursor_id is not None:
        if descending:
            stmt = stmt.where(tuple_(order_col, Entity.id) < tuple_(cursor_value, cursor_id))
        else:
            stmt = stmt.where(tuple_(order_col, Entity.id) > tuple_(cursor_value, cursor_id))
    if descending:
        stmt = stmt.order_by(order_col.desc(), Entity.id.desc())
    else:
        stmt = stmt.order_by(order_col.asc(), Entity.id.asc())
    return stmt


async def entity_browse(
    request: EntityBrowseRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> EntityBrowseResponse:
    """Keyset-paginated listing of entities, optionally prefix-filtered.

    ``name_prefix`` is normalized via :func:`_normalize_name` and then
    matched against ``entities.normalized_name`` **or** any row in
    ``entity_aliases.normalized_alias`` via a SQL ``LIKE 'prefix%'``
    backed by ``text_pattern_ops`` indexes added in migration 0004.
    """
    _ = settings or get_settings()
    env_ids = _resolve_browse_env_ids(request.env_ids, ctx)
    for env_id in env_ids:
        rbac.require("read", env_id, ctx)

    if not env_ids:
        return EntityBrowseResponse(hits=[], next_cursor=None, has_more=False)

    normalized_prefix: str | None = None
    if request.name_prefix is not None:
        normalized_prefix = _normalize_name(request.name_prefix)
        if not normalized_prefix:
            raise InvalidInputError(
                "INVALID_INPUT: name_prefix is empty after normalization",
            )

    filter_dict = _entity_browse_filter_dict(request, env_ids, normalized_prefix)
    fingerprint = compute_filter_fingerprint(filter_dict)
    direction: Direction = "desc" if request.descending else "asc"

    cursor_value: _Any = None
    cursor_id: UUID | None = None
    if request.cursor:
        cur = decode_cursor(
            request.cursor,
            expected_fingerprint=fingerprint,
            expected_order_field=request.order_by,
            expected_direction=direction,
        )
        cursor_id = cur.tiebreak_id
        if request.order_by == "created_at":
            try:
                cursor_value = dt.datetime.fromisoformat(cur.order_value)
            except ValueError as exc:
                raise InvalidCursorError(
                    f"INVALID_CURSOR: cursor order_value is not ISO-8601: {cur.order_value!r}",
                ) from exc
        else:
            cursor_value = cur.order_value

    async with session_scope() as session:
        stmt: Select[_Any] = select(Entity).where(Entity.env_id.in_(env_ids))
        if request.kinds:
            stmt = stmt.where(Entity.kind.in_(list(request.kinds)))
        if normalized_prefix is not None:
            # Escape LIKE metacharacters. ``_normalize_name`` strips ``%``
            # via ``_PUNCT_RE`` (it's not a word char) but preserves ``_``
            # (``\w`` includes underscore), which is a LIKE single-char
            # wildcard. Escape both to be safe.
            escaped = (
                normalized_prefix
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            pattern = escaped + "%"
            alias_match = (
                select(EntityAlias.entity_id)
                .where(
                    EntityAlias.entity_id == Entity.id,
                    EntityAlias.normalized_alias.like(pattern, escape="\\"),
                )
            )
            stmt = stmt.where(
                or_(
                    Entity.normalized_name.like(pattern, escape="\\"),
                    alias_match.exists(),
                )
            )

        stmt = _apply_entity_keyset(
            stmt,
            order_by=request.order_by,
            descending=request.descending,
            cursor_value=cursor_value,
            cursor_id=cursor_id,
        )
        stmt = stmt.limit(request.limit + 1)

        rows = (await session.execute(stmt)).scalars().all()
        page = list(rows[: request.limit])
        has_more = len(rows) > request.limit

        hits: list[EntityResponse] = []
        for ent in page:
            aliases = await _load_aliases(session, ent.id)
            hits.append(_entity_to_response(ent, aliases))

    next_cursor: str | None = None
    if has_more and page:
        last = page[-1]
        if request.order_by == "canonical_name":
            order_value: _Any = last.canonical_name
        else:
            order_value = last.created_at
        next_cursor = encode_cursor(
            filter_fingerprint=fingerprint,
            order_field=request.order_by,
            order_value=order_value,
            tiebreak_id=last.id,
            direction=direction,
        )

    return EntityBrowseResponse(hits=hits, next_cursor=next_cursor, has_more=has_more)
