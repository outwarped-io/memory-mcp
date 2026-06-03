"""MCP Streamable HTTP transport — registers v1 tool surface.

Wires every domain-layer function in :mod:`memory_mcp.{memories,journal,
entities,relations,envs,search}` into a :class:`FastMCP` instance whose
``streamable_http_app()`` is mounted by :mod:`memory_mcp.server` at
``/mcp``.

Identity contract (v1, local-only)
----------------------------------

Tool inputs accept two optional identity fields:

* ``agent_id`` (UUID) — caller-managed identity. When omitted, the
  server-default agent is used (see :class:`memory_mcp.identity.IdentityResolver`).
* ``attached_env_ids`` (list[UUID]) — overrides the session's attached
  envs for THIS call. Useful when a single MCP session wants to scope
  individual reads to specific envs without round-tripping through
  ``env_attach``/``env_detach``.
* ``attached_env_names`` (list[str]) — friendly-name twin for
  ``attached_env_ids``; mutually exclusive and resolved before domain code.

These fields are NOT REQUIRED for any v1 tool. v1.5 will replace them
with token/header-based identity resolution; the fields are kept on the
schemas for forward-compat but the resolver will preferentially trust
auth context once enabled.

Error mapping
-------------

Any :class:`memory_mcp.errors.MemoryMCPError` raised by a domain
function is translated to a :class:`mcp.server.fastmcp.exceptions.ToolError`
whose message embeds the stable error code (e.g. ``[VERSION_CONFLICT]``)
and optional details JSON. Unexpected exceptions are wrapped with
``[INTERNAL]`` so callers always see a well-formed error.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any
from uuid import UUID

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BaseModel
from sqlalchemy import select

from memory_mcp.browse import (
    MemBrowseRequest,
    MemBrowseResponse,
    MemFacetsRequest,
    MemFacetsResponse,
    memory_browse,
    memory_facets,
)
from memory_mcp.top import (
    MemTopRequest,
    MemTopResponse,
    memory_top,
)
from memory_mcp.config import Settings, get_settings
from memory_mcp.context_pack import ContextPackResponse, pack as context_pack
from memory_mcp.decisions import AdrExportResponse, adr_export as adr_export_memory
from memory_mcp.dream.api import (
    DreamProposalsListRequest,
    DreamProposalsListResponse,
    DreamReviewRequest,
    DreamReviewResponse,
    DreamRunRequest,
    DreamRunResponse,
    DreamStatusRequest,
    DreamStatusResponse,
    dream_proposals_list,
    dream_review,
    dream_run,
    dream_status,
)
from memory_mcp.digest import DigestResponse, ResumeResponse, digest_for_env, resume_for_env
from memory_mcp.entities import (
    EntityBrowseRequest,
    EntityBrowseResponse,
    EntityMergeRequest,
    EntityResolveRequest,
    EntityResponse,
    EntityUpsertRequest,
    entity_browse,
    entity_merge,
    entity_resolve,
    entity_upsert,
)
from memory_mcp.env_resolve import _resolve_env_refs
from memory_mcp.envs import (
    AttachedEnvsResponse,
    EnvCreateRequest,
    EnvResponse,
    env_attach,
    env_create,
    env_detach,
    env_get,
    env_list,
)
from memory_mcp.env_ops.delete import delete_env
from memory_mcp.env_ops.diff import diff_envs
from memory_mcp.env_ops.clone import clone_env
from memory_mcp.env_ops.export import export_env
from memory_mcp.env_ops.import_ import import_env
from memory_mcp.env_ops.merge import merge_envs
from memory_mcp.env_ops.migrate import migrate_env
from memory_mcp.env_ops.snapshot import create_snapshot, restore_snapshot
from memory_mcp.env_ops.rename import rename_env
from memory_mcp_schemas.digest import DigestRequest, ResumeRequest
from memory_mcp_schemas.env_ops import (
    EnvCloneRequest,
    EnvCloneResponse,
    EnvDeleteRequest,
    EnvDeleteResponse,
    EnvDiffRequest,
    EnvDiffResponse,
    EnvExportRequest,
    EnvExportResponse,
    EnvImportRequest,
    EnvImportReport,
    EnvSnapshotResponse,
    EnvSnapshotRequest,
    EnvRestoreResponse,
    EnvRestoreRequest,
    EnvMergeRequest,
    EnvMergeResponse,
    EnvMigrateRequest,
    EnvMigrateResponse,
    EnvRenameResponse,
    EnvRenameRequest,
    MemCopyRequest,
    MemCopyResponse,
    MemMoveRequest,
    MemMoveResponse,
)
from memory_mcp import rbac
from memory_mcp.db.models import Memory, Task
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import TaskRelationKind, TaskStatus
from memory_mcp.errors import EnvNotAttachedError, InvalidInputError, MemoryMCPError, NotFoundError
from memory_mcp.graph import (
    EntityNeighborsRequest,
    EntityNeighborsResponse,
    MemNeighborsRequest,
    MemNeighborsResponse,
    MemRelatedRequest,
    MemRelatedResponse,
    entity_neighbors,
    memory_neighbors,
    memory_related,
)
from memory_mcp.identity import AgentContext, get_identity_resolver
from memory_mcp.journal import JournalRequest, memory_journal
from memory_mcp.composers import (
    MemComposeRequest,
    MemComposeResponse,
    memory_compose,
)
from memory_mcp.decomposers import (
    MemDecomposeRequest,
    MemDecomposeResponse,
    memory_decompose,
)
from memory_mcp import inbox as inbox_mod
from memory_mcp.inbox import (
    MemInboxOpenRequest,
    MemInboxOpenResponse,
    MemInboxRequest,
    MemInboxResponse,
    MemInboxSendRequest,
    MemInboxSendResponse,
)
from memory_mcp.memories import (
    MemoryHardDeleteRequest,
    MemoryHardDeleteResponse,
    MemoryResponse,
    MemorySupersedeRequest,
    MemoryUpdatePatch,
    MemoryWriteRequest,
    mem_copy,
    mem_move,
    memory_archive,
    memory_get,
    memory_get_many,
    memory_hard_delete,
    memory_retire,
    memory_supersede,
    memory_update,
    memory_write,
)
from memory_mcp.playbooks import PlaybookInvokeResponse, playbook_invoke as invoke_playbook
from memory_mcp.provenance import (
    MemLineageRequest,
    MemLineageResponse,
    MemSourcesBrowseRequest,
    MemSourcesBrowseResponse,
    memory_lineage,
    memory_sources_browse,
)
from memory_mcp.relations import (
    RelationBrowseRequest,
    RelationBrowseResponse,
    RelationLinkRequest,
    RelationResponse,
    relation_browse,
    relation_link,
)
from memory_mcp.search import (
    AutoContextResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    memory_auto_context,
    memory_search,
)
from memory_mcp.stats import MemStatsRequest, MemStatsResponse, compute_mem_stats
from memory_mcp.tasks import (
    TaskCreateRequest,
    TaskLinkMemoryRequest,
    TaskLinkMemoryResponse,
    TaskListRequest,
    TaskListResponse,
    TaskRelationRequest,
    TaskRelationResponse,
    TaskResponse,
    TaskTreeResponse,
    task_create as task_create_impl,
    task_dep_link as task_dep_link_impl,
    task_link_memory as task_link_memory_impl,
    task_list as task_list_impl,
    task_next as task_next_impl,
    task_status_set as task_status_set_impl,
    task_substep as task_substep_impl,
    task_tree as task_tree_impl,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context resolution
# ---------------------------------------------------------------------------


class _AttachedEnvRefs(BaseModel):
    model_config = {"extra": "forbid"}

    env_ids: list[UUID] | None = None
    env_names: list[str] | None = None


async def _resolve_ctx(
    *,
    agent_id: UUID | None,
    attached_env_ids: list[UUID] | None,
    attached_env_names: list[str] | None = None,
    settings: Settings | None = None,
) -> AgentContext:
    """Build an :class:`AgentContext` for a single tool invocation.

    Falls back to the server-default agent when ``agent_id`` is missing.
    The MCP transport does not own session_id state in v1 (each call is
    self-contained); ``attached_env_ids`` / ``attached_env_names`` are
    sourced from the per-call parameter and overlaid onto the resolved context.
    """
    settings = settings or get_settings()
    resolver = get_identity_resolver(settings)
    ctx = await resolver.resolve(
        agent_id_header=str(agent_id) if agent_id else None,
        agent_name_header=None,
        session_id_header=None,
    )
    if attached_env_ids or attached_env_names:
        refs = await _resolve_env_refs(
            _AttachedEnvRefs(env_ids=attached_env_ids, env_names=attached_env_names)
        )
        ctx.attached_env_ids = list(refs.env_ids or [])
        ctx.attached_env_names = []
    return ctx


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def _format_tool_error(exc: BaseException) -> ToolError:
    if isinstance(exc, MemoryMCPError):
        body = {"code": exc.code, "message": str(exc), "details": exc.details}
    else:
        body = {"code": "INTERNAL", "message": str(exc) or repr(exc)}
    # Embed the structured info in the message so MCP clients that don't
    # parse `data` still see actionable text.
    details_json = json.dumps(body.get("details", {}))
    return ToolError(
        f"[{body['code']}] {body['message']} :: {details_json}"
    )


def _wrap[T](
    fn: Callable[..., Awaitable[T]],
) -> Callable[..., Awaitable[T]]:
    """Decorator: error-translate + metrics-instrument an MCP tool.

    Order matters: ``instrument_tool`` is applied INSIDE the error
    translator so that :class:`MemoryMCPError` is observed BEFORE it's
    converted to :class:`ToolError`. This lets metrics distinguish
    caller-correctable errors (``outcome=mcperror``) from unexpected
    server failures (``outcome=error``).
    """
    from memory_mcp.observability import instrument_tool

    instrumented = instrument_tool(fn.__name__)(fn)

    @wraps(fn)
    async def error_translated(*args: Any, **kwargs: Any) -> T:
        try:
            return await instrumented(*args, **kwargs)
        except MemoryMCPError as exc:
            log.info("mcp tool '%s' raised %s: %s", fn.__name__, exc.code, exc)
            raise _format_tool_error(exc) from exc
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 — catch-all is the contract
            log.exception("mcp tool '%s' unexpected error", fn.__name__)
            raise _format_tool_error(exc) from exc

    return error_translated


def _dump(obj: Any) -> Any:
    """Pydantic v2 model → plain dict; pass-through for everything else."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_dump(o) for o in obj]
    return obj


def _require_env_attached(env_id: UUID, ctx: AgentContext) -> None:
    """Require explicit per-call/session attachment for single-env tools."""
    if env_id not in set(ctx.attached_env_ids):
        raise EnvNotAttachedError(
            f"ENV_NOT_ATTACHED: env {env_id} is not attached to this session",
            env_id=str(env_id),
            attached_env_ids=[str(e) for e in ctx.attached_env_ids],
        )


async def _resolve_task_env(task_id: UUID) -> UUID:
    async with session_scope() as session:
        env_id = (await session.execute(
            select(Task.env_id).where(Task.id == task_id)
        )).scalar_one_or_none()
    if env_id is None:
        raise NotFoundError(f"task {task_id} not found", task_id=str(task_id))
    return env_id


async def _resolve_memory_env(memory_id: UUID) -> UUID:
    async with session_scope() as session:
        env_id = (await session.execute(
            select(Memory.env_id).where(Memory.id == memory_id)
        )).scalar_one_or_none()
    if env_id is None:
        raise NotFoundError(f"memory {memory_id} not found", memory_id=str(memory_id))
    return env_id


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def build_mcp_server(
    settings: Settings | None = None,
) -> FastMCP:
    """Build a FastMCP server with the v1 tool surface registered.

    The returned instance is *not* started; callers either call
    ``.streamable_http_app()`` to mount it under FastAPI, or
    ``.run_streamable_http_async()`` for a standalone deployment.
    """
    settings = settings or get_settings()
    mcp = FastMCP(
        name="memory-mcp",
        instructions=(
            "Shared cross-session memory for AI agents. v1 is local-only "
            "(no auth). Use memory_write / memory_search / memory_journal "
            "to capture and retrieve facts, procedures, observations and "
            "snippets. Pass an explicit agent_id (UUID) to attribute "
            "writes to a specific identity; otherwise the server-default "
            "agent is used."
        ),
        # Mount under FastAPI at "/mcp" — the inner Starlette app's route
        # must be "/" so the final URL is /mcp (not /mcp/mcp).
        streamable_http_path="/",
        host=settings.mcp_http_host,
        port=settings.mcp_http_port,
        log_level="INFO",
        stateless_http=True,
    )

    # Override the default serverInfo.version (which falls through to the
    # MCP SDK version when FastMCP doesn't accept a `version=` kwarg) with
    # memory-mcp's own package version, so `initialize` reports it correctly.
    try:
        mcp._mcp_server.version = _pkg_version("memory-mcp")
    except PackageNotFoundError:
        # Editable / dev install where the package isn't pip-visible — leave
        # the SDK default in place rather than crashing app startup.
        pass

    # ---- Memory CRUD ------------------------------------------------------

    @mcp.tool()
    @_wrap
    async def mem_write(
        request: MemoryWriteRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new memory. Returns the canonical record + version.

        Example:
            {
              "request": {
                "kind": "fact",
                "title": "deploy summary",
                "body": "Deployment completed with no known regressions.",
                "env_name": "project-a",
                "tags": ["source:adhoc", "topic:deploy"]
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: MemoryResponse = await memory_write(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_get(
        memory_id: UUID,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch a single memory by id (canonical Postgres read).

        Example:
            {
              "memory_id": "00000000-0000-0000-0000-000000000001"
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        return _dump(await memory_get(memory_id, ctx=ctx))

    @mcp.tool()
    @_wrap
    async def mem_get_many(
        memory_ids: list[UUID],
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Bulk fetch memories by id.

        Example:
            {
              "memory_ids": [
                "00000000-0000-0000-0000-000000000001",
                "00000000-0000-0000-0000-000000000002"
              ]
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        return _dump(await memory_get_many(memory_ids, ctx=ctx))

    @mcp.tool()
    @_wrap
    async def adr_export(
        memory_id: UUID,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Export a decision memory as ADR-style markdown.

        Example:
            {
              "memory_id": "00000000-0000-0000-0000-000000000001"
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        env_id = await _resolve_memory_env(memory_id)
        _require_env_attached(env_id, ctx)
        out: AdrExportResponse = await adr_export_memory(memory_id, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_update(
        memory_id: UUID,
        patch: MemoryUpdatePatch,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Patch a memory using optimistic concurrency control.

        Example:
            {
              "memory_id": "00000000-0000-0000-0000-000000000001",
              "patch": {
                "expected_version": 1,
                "title": "updated deploy summary",
                "tags": ["source:adhoc", "topic:deploy"]
              }
            }

        ``patch.expected_version`` carries the optimistic-lock version; on
        mismatch the domain layer raises :class:`VersionConflictError`
        which surfaces as ``[VERSION_CONFLICT]`` to the MCP caller.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        return _dump(await memory_update(memory_id, patch=patch, ctx=ctx))

    @mcp.tool()
    @_wrap
    async def mem_archive(
        memory_id: UUID,
        expected_version: int,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Move a memory to the ``archived`` lifecycle state.

        Example:
            {
              "memory_id": "00000000-0000-0000-0000-000000000001",
              "expected_version": 1
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        return _dump(await memory_archive(
            memory_id, expected_version=expected_version, ctx=ctx,
        ))

    @mcp.tool()
    @_wrap
    async def mem_retire(
        memory_id: UUID,
        expected_version: int,
        reason: str,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Move a memory to the ``retired`` lifecycle state.

        Example:
            {
              "memory_id": "00000000-0000-0000-0000-000000000001",
              "expected_version": 1,
              "reason": "superseded by newer guidance"
            }

        ``reason`` is recorded in ``audit_log`` and is required.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        return _dump(await memory_retire(
            memory_id, expected_version=expected_version,
            reason=reason, ctx=ctx,
        ))

    @mcp.tool()
    @_wrap
    async def mem_hard_delete(
        memory_id: UUID,
        request: MemoryHardDeleteRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Permanently destroy a memory's canonical row, body, and projections.

        Example:
            {
              "memory_id": "00000000-0000-0000-0000-000000000001",
              "request": {
                "expected_version": 1,
                "reason": "remove sensitive test data",
                "confirm_destroy": true,
                "cascade": true,
                "dry_run": true,
                "max_cascade_depth": 5,
                "max_cascade_count": 20
              }
            }

        Unlike ``mem_retire`` (soft-delete; body remains queryable by id),
        ``mem_hard_delete`` removes the canonical row and enqueues a
        projection-eviction event so Qdrant and Neo4j drop the row too.

        Safety
        ------

        * ``request.confirm_destroy`` must be ``true`` — the call refuses
          otherwise.
        * ``request.reason`` is required; it is captured in the tombstone
          and audit_log so leak-recovery and after-the-fact review work.
        * ``request.expected_version`` is the standard optimistic-lock
          version. Mismatch ⇒ ``[VERSION_CONFLICT]``.
        * ``request.cascade=true`` opts into deleting forward-lineage
          dependents in reverse-topo order; ``max_cascade_depth`` and
          ``max_cascade_count`` bound the blast radius.
        * ``request.dry_run=true`` validates the plan and returns the
          ordered ``affected`` rows without mutating canonical storage.

        Response semantics
        ------------------

        ``cascade_root`` correlates every row touched by a cascade and
        ``affected`` lists them in execution order (leaves first, root
        last). On ``dry_run=true`` the response always reports
        ``canonical_deleted=false`` and leaves ``deleted_at``,
        ``tombstone_id``, and ``projection_eviction`` as ``null``.

        Recovery
        --------

        Hard delete is the **only** mitigation that erases the body —
        see ``memory-mcp.instructions.md §14`` for the full sensitive-
        write recovery protocol. Rotating the underlying secret is still
        the only true fix; ``mem_hard_delete`` is bookkeeping that
        unblocks audits.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        return _dump(await memory_hard_delete(memory_id, request, ctx=ctx))

    @mcp.tool()
    @_wrap
    async def mem_supersede(
        old_memory_id: UUID,
        request: MemorySupersedeRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Replace a memory with a successor; returns ``{old, new}`` pair.

        Example:
            {
              "old_memory_id": "00000000-0000-0000-0000-000000000001",
              "request": {
                "expected_version": 1,
                "new": {
                  "kind": "fact",
                  "title": "deploy summary",
                  "body": "Revised deploy summary.",
                  "tags": ["source:adhoc", "topic:deploy"]
                }
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        old, new = await memory_supersede(old_memory_id, request, ctx=ctx)
        return {"old": _dump(old), "new": _dump(new)}

    @mcp.tool()
    @_wrap
    async def mem_compose(
        request: MemComposeRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compose N≥2 source memories into a single new memory (Phase 2, v0.15.0).

        Two modes:

        * ``promote`` (default, non-destructive) — sources stay ``active``;
          new memory cites them via ``promoted_from``.
        * ``merge`` (destructive) — sources transition to ``superseded``
          with ``superseded_by`` set to the new memory; new memory cites
          them via ``supersedes``.

        Idempotent via dedupe key: same ``{mode, source_ids, target}``
        replays the same composed memory with ``idempotency_replay=true``
        and performs no mutation. Caller may override the dedupe key with
        ``request.idempotency_key``.

        Popularity caveat (v1): the composed memory starts at
        ``reference_count=0``. Citation transfer (rewriting incoming edges
        from sources to the new memory) is deferred to v1.5.

        Example:
            {
              "request": {
                "source_ids": [
                  "00000000-0000-0000-0000-000000000001",
                  "00000000-0000-0000-0000-000000000002"
                ],
                "target": {
                  "kind": "fact",
                  "title": "Combined deploy outcome",
                  "body": "Stamp A and Stamp B both reached steady state.",
                  "tags": ["topic:deploy", "release:2026.05"]
                },
                "mode": "promote"
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id,
            attached_env_ids=attached_env_ids,
            attached_env_names=attached_env_names,
        )
        out: MemComposeResponse = await memory_compose(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_decompose(
        request: MemDecomposeRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Decompose one source memory into N≥2 children (Phase 3, v0.15.0).

        Two modes:

        * ``derive`` (default, non-destructive) — source stays ``active``;
          each child cites it via ``derived_from``. Used for atomic-fact
          extraction or splitting a coarse observation into evidence
          leaves without losing the original.
        * ``split`` (destructive) — source transitions to ``retired``;
          each child cites it via ``split_from``. Used when the source
          was the wrong granularity from the start and should disappear
          from search after the decomposition.

        Idempotent via dedupe key: same ``{mode, source_id, children}``
        replays the same children with ``idempotency_replay=true`` and
        performs no mutation. Caller may override the dedupe key with
        ``request.idempotency_key``.

        Popularity caveat (v1): ``split_from`` is intentionally NOT in
        the load-bearing popularity whitelist (migration 0021) — a
        retired source should not accrue analytics from its split. New
        children start at ``reference_count=0`` regardless of mode.
        Citation transfer (rewriting incoming edges from the source to
        the children) is deferred to v1.5.

        Example:
            {
              "request": {
                "source_id": "00000000-0000-0000-0000-000000000001",
                "children": [
                  {"kind": "fact",
                   "title": "Stamp A reached steady state",
                   "body": "Stamp A finished bake at 14:02 UTC."},
                  {"kind": "fact",
                   "title": "Stamp B reached steady state",
                   "body": "Stamp B finished bake at 14:07 UTC."}
                ],
                "mode": "derive"
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id,
            attached_env_ids=attached_env_ids,
            attached_env_names=attached_env_names,
        )
        out: MemDecomposeResponse = await memory_decompose(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_inbox_open(
        request: MemInboxOpenRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Open (or fetch) an inter-agent inbox / drop-box channel (v0.17).

        Creates a ``kind=channel`` entity in the target env and returns a
        copy-pasteable ``reference`` string of the form
        ``mem-inbox://<env-name>/<slug>``. The reference is the durable
        handle the user moves between agents — it is self-describing,
        env-qualified, and stable across sessions.

        Slug behavior:

        * ``name`` provided → used as the slug. With ``idempotent=False``
          (default) a pre-existing slug returns
          ``INBOX_CHANNEL_ALREADY_EXISTS``; with ``idempotent=True`` the
          existing channel is returned and ``created=false``.
        * ``name`` omitted → server generates a pronounceable
          ``adjective-noun`` slug (e.g. ``quiet-otter``). Auto-generated
          slugs are always created fresh.

        User-orchestrated workflow examples:

        * UC1 — *recipient-initiated*: user asks Agent A to open an inbox;
          Agent A echoes the reference; user pastes it into Agent B which
          sends via :func:`mem_inbox_send`.
        * UC2 — *sender-initiated*: Agent A opens then sends in one
          session, returns the reference for the user to share.
        * UC3 — *established channel*: both agents already know the
          reference; user says "pass to <ref>" / "check <ref>".

        Example:
            {
              "request": {"env_name": "personal", "title": "RLS handoff"}
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id,
            attached_env_ids=attached_env_ids,
            attached_env_names=attached_env_names,
        )
        out: MemInboxOpenResponse = await inbox_mod.mem_inbox_open(
            request, ctx=ctx
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_inbox_send(
        request: MemInboxSendRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Drop a message into an existing inbox channel (v0.17).

        Wraps :func:`mem_write` with ``kind=message``, the channel entity
        in ``entity_links``, the fixed ``inbox`` tag, and a TTL. The
        ``to`` field accepts either the full reference
        ``mem-inbox://<env>/<slug>`` (env taken from the URL) or a bare
        slug (env required via ``env_id`` / ``env_name``). When both URL
        and arg env are supplied, they must match — mismatch raises
        ``INBOX_ENV_MISMATCH`` to prevent silent cross-env writes.

        Rejects non-existent slugs with ``INBOX_CHANNEL_NOT_FOUND`` —
        explicit :func:`mem_inbox_open` is required first. This prevents
        typo-driven channel proliferation.

        TTL: default 7 days, hard cap 90 days. Naive ``expires_at`` is
        treated as UTC. Past timestamps raise ``INBOX_TTL_IN_PAST``.

        Authorship: ``created_by_agent_id`` is always recorded server-side
        from ``ctx``. Optional ``display_from`` (free-form string) is
        stored in ``metadata`` for human-readable provenance; omit it for
        a pseudonymous message.

        Example:
            {
              "request": {
                "to": "mem-inbox://workspace/quiet-otter",
                "body": "Please refresh the build pipeline tomorrow."
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id,
            attached_env_ids=attached_env_ids,
            attached_env_names=attached_env_names,
        )
        out: MemInboxSendResponse = await inbox_mod.mem_inbox_send(
            request, ctx=ctx
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_inbox(
        request: MemInboxRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """List messages in an inbox channel (v0.17).

        Returns ``items`` (newest first by default), an opaque
        ``next_cursor`` for keyset pagination on ``(created_at, id)``,
        and ``has_more``. Pass ``cursor`` from a prior response to resume.

        Expired messages are excluded by default; pass
        ``include_expired=True`` to surface them (useful for audit /
        recovery flows). Read is pure — no acknowledgement state is
        written.

        The ``to`` reference accepts the same forms as
        :func:`mem_inbox_send`: URL (``mem-inbox://<env>/<slug>``) or
        bare slug + ``env_id`` / ``env_name``.

        Example:
            {
              "request": {"to": "mem-inbox://workspace/quiet-otter",
                          "limit": 20}
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id,
            attached_env_ids=attached_env_ids,
            attached_env_names=attached_env_names,
        )
        out: MemInboxResponse = await inbox_mod.mem_inbox(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_copy_(
        request: MemCopyRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Copy a memory from source env to destination env. See §3.2.

        Example:
            {
              "request": {
                "memory_id": "00000000-0000-0000-0000-000000000001",
                "dst_env_id": "00000000-0000-0000-0000-000000000102",
                "copy_tags": true,
                "copy_provenance": true
              }
            }

        Creates a new memory in dst with a fresh UUID identical to source's body/kind/payload. Tags, provenance, and a
        cross-env lineage edge are copied by default (toggleable). The source memory is unchanged. Embeddings replicate
        verbatim unless re_embed_if_model_mismatch=True forces re-embedding via the destination env's model.
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        out: MemCopyResponse = await mem_copy(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_move_(
        request: MemMoveRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Move a memory from source env to destination env. See §17.7.

        Example:
            {
              "request": {
                "memory_id": "00000000-0000-0000-0000-000000000001",
                "dst_env_id": "00000000-0000-0000-0000-000000000102",
                "redirect_source": true
              }
            }

        Equivalent to mem_copy followed by superseding the source memory to point at the new dst-side memory. The source
        memory's UUID remains valid as a tombstone; searches in the source env will see it as 'superseded'. The cross-env
        supersession is allowed only for mem_move (mem_supersede still blocks all cross-env supersessions).
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        out: MemMoveResponse = await mem_move(request, ctx=ctx)
        return _dump(out)

    # ---- Journal ----------------------------------------------------------

    @mcp.tool()
    @_wrap
    async def mem_journal(
        request: JournalRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Append a short-form observation; faster than memory_write.

        Example:
            {
              "request": {
                "content": "Checked deployment health; no alerts firing.",
                "env_name": "project-a",
                "tags": ["source:journal", "topic:deploy"]
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        return _dump(await memory_journal(request, ctx=ctx))

    @mcp.tool()
    @_wrap
    async def mem_digest(
        env_id: UUID | None = None,
        env_name: str | None = None,
        since_ts: dt.datetime | None = None,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize an environment into a persisted six-section digest.

        Example:
            {
              "env_name": "project-a",
              "since_ts": "2026-01-01T00:00:00Z"
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(
            DigestRequest(env_id=env_id, env_name=env_name, since_ts=since_ts)
        )
        out: DigestResponse = await digest_for_env(
            request.env_id, since_ts=request.since_ts, ctx=ctx, settings=settings,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_resume(
        env_id: UUID | None = None,
        env_name: str | None = None,
        journal_tail: int = 20,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the latest digest, recent journal entries, and counts.

        Example:
            {
              "env_name": "project-a",
              "journal_tail": 5
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(
            ResumeRequest(env_id=env_id, env_name=env_name, journal_tail=journal_tail)
        )
        out: ResumeResponse = await resume_for_env(
            request.env_id, journal_tail=request.journal_tail, ctx=ctx,
        )
        return _dump(out)

    # ---- Search -----------------------------------------------------------

    @mcp.tool()
    @_wrap
    async def mem_search(
        request: MemorySearchRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Hybrid (lex + sem + graph) search across attached envs.

        Examples:
            {
              "request": {
                "query": "outgoing deploy",
                "env_names": ["project-a"],
                "expansion": "broad",
                "limit": 5
              }
            }

            ``expansion="broad"`` is a good fit for casual recall;
            ``expansion="narrow"`` keeps only high-confidence matches.

        The top-level key MUST be "request" (not "req"). Use either env_ids
        (UUIDs) or env_names (strings) — not both. Names resolve case-insensitively.

        Modes:

        * ``hybrid`` (default) — RRF fusion of ``lex`` + ``sem`` + ``graph``.
          The graph leg degrades silently if the graph backend is unavailable
          or the query has no resolvable entities.
        * ``lex`` — Postgres FTS only.
        * ``sem`` — Qdrant only.
        * ``graph`` — graph projection only. Propagates a
          ``GRAPH_BACKEND_UNAVAILABLE`` error if the backend is down.
        * ``id`` — explicit id lookup.

        See the ``consistency`` and ``follow_superseded`` parameters for
        read-after-write and supersession semantics. ``consistency=fresh``
        waits on every projection sink the request will consult (qdrant
        and/or neo4j); if any sink lags out the response degrades to
        canonical (lex-only) and ``consistency_used`` reflects that.

        Relax / tighten knobs:

        * ``expansion`` bundles common recall presets and reports the
          exact resolved settings in ``expansion_resolved``.
        * ``min_score: float | None`` (tighten) — drop hits with fused
          score below this threshold. Empirical 50th-percentile is ~0.016,
          90th-percentile ~0.035 on the default RRF + salience boost.
        * ``fallback: bool`` (loosen) — when set and the initial search
          returns 0 hits (or all are removed by ``min_score``), the server
          re-runs the query with progressively broader scope: widen
          ``lex`` → ``hybrid``, drop optional filters, widen lifecycle,
          then boost ``limit`` 5×. Steps that fired are reported in
          ``fallback_used``. ``mode=id`` does not participate.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: MemorySearchResponse = await memory_search(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_auto_context(
        task_desc: str,
        env_id: UUID,
        top_k: int = 8,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Find memories whose trigger descriptions match the current task.

        Example:
            {
              "task_desc": "prepare outgoing deploy summary",
              "env_id": "00000000-0000-0000-0000-000000000101",
              "top_k": 5
            }
        """
        _ = ctx
        agent_ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        _require_env_attached(env_id, agent_ctx)
        rbac.require("read", env_id, agent_ctx)
        out: AutoContextResponse = await memory_auto_context(
            task_desc=task_desc,
            env_id=env_id,
            top_k=top_k,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_neighbors(
        request: MemNeighborsRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Walk the projected graph from a starting memory.

        Example:
            {
              "request": {
                "memory_id": "00000000-0000-0000-0000-000000000001",
                "env_name": "project-a",
                "hops": 1,
                "edge_types": ["derives_from"],
                "fallback": true,
                "limit": 5
              }
            }

        When ``fallback=true`` and the strict traversal is empty, the
        response reports the relaxation ladder that fired in
        ``fallback_used`` (for example ``widen_hops`` or
        ``include_retired``).
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: MemNeighborsResponse = await memory_neighbors(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_related(
        request: MemRelatedRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find memories related by shared entities or semantic similarity.

        Example:
            {
              "request": {
                "memory_id": "00000000-0000-0000-0000-000000000001",
                "env_name": "project-a",
                "relation": "semantic",
                "min_score": 0.5,
                "fallback": true,
                "limit": 5
              }
            }

        ``min_score`` is only valid for ``relation="semantic"``.
        Relaxation steps that fired are echoed back in ``fallback_used``.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: MemRelatedResponse = await memory_related(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_lineage(
        request: MemLineageRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Trace provenance lineage around a seed memory.

        Example:
            {
              "request": {
                "memory_id": "00000000-0000-0000-0000-000000000001",
                "env_name": "project-archive",
                "direction": "both",
                "max_depth": 3
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request, allow_deleted=True)
        out: MemLineageResponse = await memory_lineage(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_sources_browse(
        request: MemSourcesBrowseRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Keyset-paginated browse over memory provenance sources.

        Example:
            {
              "request": {
                "env_names": ["project-a"],
                "memory_ids": ["00000000-0000-0000-0000-000000000001"],
                "source_types": ["agent"],
                "hydrate_memories": false,
                "limit": 5
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: MemSourcesBrowseResponse = await memory_sources_browse(
            request, ctx=ctx,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_browse(
        request: MemBrowseRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Keyset-paginated listing of memories with no relevance ranking.

        Example:
            {
              "request": {
                "env_names": ["project-a"],
                "kinds": ["fact"],
                "tags": ["topic:deploy"],
                "limit": 5
              }
            }

        Filter parity with ``mem_search`` (env_ids / kinds / tags /
        statuses / time windows). Default visibility is ``[proposed,
        active]``. Order by ``updated_at`` (default) or ``created_at``;
        cursors are bound to the filter fingerprint so a mid-page filter
        change raises ``INVALID_CURSOR``.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: MemBrowseResponse = await memory_browse(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_facets(
        request: MemFacetsRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Distinct-value + count aggregation over memories.

        Example:
            {
              "request": {
                "env_names": ["project-a"],
                "facets": ["kind", "status", "tag"],
                "tag_limit": 10
              }
            }

        Default facets: ``kind``, ``status``, ``tag``. ``month`` is
        opt-in. Counts are computed under a statement timeout; on
        timeout the response carries ``approximate=True`` with whatever
        facets completed. ``max_rows`` lets the caller bound exact-mode
        cost.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: MemFacetsResponse = await memory_facets(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_top(
        request: MemTopRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the highest-ranked memories under a chosen metric.

        Example:
            {
              "request": {
                "env_names": ["project-a"],
                "by": "reference_count",
                "kinds": ["fact", "procedure"],
                "tags": ["repo:org/service-a"],
                "tag_match": "any",
                "limit": 10
              }
            }

        Metrics: ``salience`` (default), ``access_count``,
        ``reference_count`` (graph-citation sum across rel_link / lineage
        / task / playbook), ``reference_velocity`` (recent citation
        arrival rate; honors ``velocity_window_days``, default 30), or
        ``reference_authority`` (weighted citation footprint —
        ``Σ source.salience`` over inbound citations; **requires**
        ``dream_popularity_authority_weighted=True`` in settings or the
        call raises ``AUTHORITY_DISABLED``).

        Tie-breaker is stable: ``(metric DESC, created_at DESC, id DESC)``.
        Default status filter is ``[active]`` — top-of-the-board is a live
        signal. ``tag_match`` defaults to ``"any"`` (OR semantics, parity
        with ``mem_search`` / ``mem_browse``). For ``reference_velocity``
        and ``reference_authority``, zero-valued rows are excluded from
        ``items`` but counted in ``total_examined``.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: MemTopResponse = await memory_top(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_stats(
        request: MemStatsRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a read-only memory health snapshot.

        Example:
            {
              "request": {
                "env_names": ["project-a"],
                "include_substrates": false,
                "include_distributions": true
              }
            }

        Body-byte totals are memory body bytes only; disk usage is available
        through ``include_substrates=true``.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: MemStatsResponse = await compute_mem_stats(request, ctx=ctx)
        return _dump(out)

    # ---- Entities ---------------------------------------------------------

    @mcp.tool()
    @_wrap
    async def ent_upsert(
        request: EntityUpsertRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create or update an entity (canonical name + aliases + attrs).

        Example:
            {
              "request": {
                "kind": "service",
                "canonical_name": "management",
                "aliases": ["mgmt"],
                "env_name": "project-a"
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: EntityResponse = await entity_upsert(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def ent_resolve(
        request: EntityResolveRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve an entity name (canonical or alias) to id(s).

        Example:
            {
              "request": {
                "name": "management",
                "env_names": ["project-a"],
                "kinds": ["service"],
                "limit": 5
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        return _dump(await entity_resolve(request, ctx=ctx))

    @mcp.tool()
    @_wrap
    async def ent_merge(
        request: EntityMergeRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Merge duplicate entities into a canonical id.

        Example:
            {
              "request": {
                "keep_id": "00000000-0000-0000-0000-000000000601",
                "merge_ids": ["00000000-0000-0000-0000-000000000602"],
                "expected_versions": {
                  "00000000-0000-0000-0000-000000000601": 1,
                  "00000000-0000-0000-0000-000000000602": 1
                }
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        return _dump(await entity_merge(request, ctx=ctx))

    @mcp.tool()
    @_wrap
    async def ent_neighbors(
        request: EntityNeighborsRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Walk the projected entity graph from a starting entity.

        Example:
            {
              "request": {
                "entity_id": "00000000-0000-0000-0000-000000000601",
                "hops": 2,
                "edge_types": ["mentions"],
                "limit": 5
              }
            }

        Returns nodes within ``hops`` of ``entity_id`` (alias ``id``),
        filtered by edge type, terminal kind, and direction. Memories
        with hidden / archived / superseded / retired status are
        suppressed from both terminals and path transit. Pagination is
        opaque-cursor; pages may be sparse after lifecycle filtering.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: EntityNeighborsResponse = await entity_neighbors(
            request, ctx=ctx,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def ent_browse(
        request: EntityBrowseRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Keyset-paginated listing of entities, optionally prefix-filtered.

        Example:
            {
              "request": {
                "kinds": ["service"],
                "name_prefix": "man",
                "limit": 5
              }
            }

        ``name_prefix`` is normalized (NFKC + lowercase + strip
        punctuation + collapse whitespace) and matches against either
        the entity's normalized canonical name OR any normalized alias.
        Backed by ``text_pattern_ops`` indexes; LIKE ``prefix%`` only.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: EntityBrowseResponse = await entity_browse(request, ctx=ctx)
        return _dump(out)

    # ---- Relations --------------------------------------------------------

    @mcp.tool()
    @_wrap
    async def rel_link(
        request: RelationLinkRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Link two graph nodes (entity or memory) with a typed edge.

        Example:
            {
              "request": {
                "src": {"kind": "memory", "id": "00000000-0000-0000-0000-000000000001"},
                "dst": {"kind": "entity", "id": "00000000-0000-0000-0000-000000000601"},
                "type": "mentions",
                "env_name": "project-a"
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: RelationResponse = await relation_link(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def rel_browse(
        request: RelationBrowseRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Keyset-paginated listing of relation edges.

        Example:
            {
              "request": {
                "types": ["mentions"],
                "src_kind": "memory",
                "src_id": "00000000-0000-0000-0000-000000000001",
                "limit": 5
              }
            }

        Filter by env, edge ``types`` (max 20 values), endpoint kinds,
        and/or specific endpoint ids. Default order is
        ``created_at DESC``. ``src_id`` / ``dst_id`` pin endpoints to a
        specific canonical record (entity.id or memory.id, NOT
        graph_node id).
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: RelationBrowseResponse = await relation_browse(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def mem_context_pack(
        task_desc: str,
        env_id: UUID,
        token_budget: int = 4000,
        include_core: bool = True,
        include_journal: bool = True,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a token-budgeted prompt bundle for a task.

        Example:
            {
              "task_desc": "prepare release checklist",
              "env_id": "00000000-0000-0000-0000-000000000101",
              "token_budget": 4000,
              "include_journal": true
            }

        Orchestrates latest digest, F1 trigger matches, recent journal, and
        salience-ranked archival memories. ``include_core`` is reserved for
        future core-pinned support and is currently a no-op.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        out: ContextPackResponse = await context_pack(
            task_desc=task_desc,
            env_id=env_id,
            token_budget=token_budget,
            include_core=include_core,
            include_journal=include_journal,
            ctx=ctx,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def playbook_invoke(
        macro: str,
        env_id: UUID,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Invoke a playbook macro and resolve same-env memory refs.

        Example:
            {
              "macro": "release-checklist",
              "env_id": "00000000-0000-0000-0000-000000000101"
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        _require_env_attached(env_id, ctx)
        rbac.require("read", env_id, ctx)
        out: PlaybookInvokeResponse = await invoke_playbook(
            macro=macro,
            env_id=env_id,
            ctx=ctx,
        )
        return _dump(out)

    # ---- Environments -----------------------------------------------------

    @mcp.tool()
    @_wrap
    async def env_create_(
        request: EnvCreateRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new environment / namespace.

        Example:
            {
              "request": {
                "name": "workspace",
                "kind": "team"
              }
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        out: EnvResponse = await env_create(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_list_(
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List all environments (no grants enforced in v1).

        Example:
            {
              "include_deleted": false
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        return _dump(await env_list(ctx=ctx, include_deleted=include_deleted))

    @mcp.tool()
    @_wrap
    async def env_get_(
        name: str | None = None,
        env_id: UUID | None = None,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """Resolve an env by name or id.

        Example:
            {
              "name": "project-a"
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        out: EnvResponse = await env_get(
            name=name,
            env_id=env_id,
            ctx=ctx,
            include_deleted=include_deleted,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_delete_(
        request: EnvDeleteRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Soft-delete an env with cascade. See §9 + §17.4.

        Example:
            {
              "request": {
                "env_id": "00000000-0000-0000-0000-000000000101",
                "confirm_destroy": true,
                "cascade_external_refs": false
              }
            }

        All env-scoped rows are HARD-deleted in dependency order. The environment row itself
        is SOFT-deleted (status='deleted', deleted_at=now()) — its UUID remains valid forever
        to avoid breaking lineage edges in other envs. Requires ``confirm_destroy=True``.
        By default, fails fast with sample IDs if any other env has a lineage edge pointing
        into this env. Pass ``cascade_external_refs=True`` to drop those edges and proceed.
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        request = await _resolve_env_refs(request, allow_deleted=True)
        out: EnvDeleteResponse = await delete_env(request, ctx=ctx)
        return _dump(out)


    @mcp.tool()
    @_wrap
    async def env_rename_(
        request: EnvRenameRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update mutable Environment fields. See §10.

        Example:
            {
              "request": {
                "env_id": "00000000-0000-0000-0000-000000000101",
                "new_name": "project-archive"
              }
            }

        Updateable: ``name`` (unique), ``default_embedding_model_id`` (does NOT re-embed
        existing memories — only affects new ones), ``retention_policy`` (effective at next
        dream-run). The env_id is immutable (lineage anchor).
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        request = await _resolve_env_refs(request, allow_deleted=True)
        out: EnvRenameResponse = await rename_env(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_attach_(
        name: str,
        session_id: UUID,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Attach an env name to the caller's session for subsequent calls.

        Example:
            {
              "name": "project-a",
              "session_id": "00000000-0000-0000-0000-000000000301"
            }

        ``session_id`` is **required** since the MCP transport in v1 does
        not multiplex requests by session — clients track their own
        ``session_id`` and pass it explicitly. v1.5 replaces this with
        token-derived identity.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        ctx.session_id = session_id
        out: AttachedEnvsResponse = await env_attach(name=name, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_detach_(
        name: str,
        session_id: UUID,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detach an env name from the caller's session.

        Example:
            {
              "name": "project-a",
              "session_id": "00000000-0000-0000-0000-000000000301"
            }
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        ctx.session_id = session_id
        out = await env_detach(name=name, ctx=ctx)
        return _dump(out)

    # ----- Task tools -----

    @mcp.tool()
    @_wrap
    async def task_create(
        request: TaskCreateRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a task in an environment.

        Example:
            {
              "request": {
                "env_id": "00000000-0000-0000-0000-000000000101",
                "title": "prepare release checklist",
                "description": "Collect rollout prerequisites.",
                "priority": 50
              }
            }
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        request = await _resolve_env_refs(request)
        _require_env_attached(request.env_id, ctx)
        out: TaskResponse = await task_create_impl(request, ctx=ctx, settings=settings)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def task_substep(
        parent_task_id: UUID,
        title: str,
        description: str | None = None,
        priority: int = 50,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a child task under an existing task.

        Example:
            {
              "parent_task_id": "00000000-0000-0000-0000-000000000201",
              "title": "verify deployment gates",
              "priority": 40
            }
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        env_id = await _resolve_task_env(parent_task_id)
        _require_env_attached(env_id, ctx)
        out: TaskResponse = await task_substep_impl(
            parent_task_id,
            title=title,
            description=description,
            priority=priority,
            ctx=ctx,
            settings=settings,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def task_dep_link(
        src_task_id: UUID,
        dst_task_id: UUID,
        type: TaskRelationKind = TaskRelationKind.depends_on,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Link one task as depending on another.

        Example:
            {
              "src_task_id": "00000000-0000-0000-0000-000000000202",
              "dst_task_id": "00000000-0000-0000-0000-000000000201",
              "type": "depends_on"
            }
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        src_env_id = await _resolve_task_env(src_task_id)
        dst_env_id = await _resolve_task_env(dst_task_id)
        if src_env_id != dst_env_id:
            raise InvalidInputError("task dependencies must stay within one env")
        _require_env_attached(src_env_id, ctx)
        out: TaskRelationResponse = await task_dep_link_impl(
            TaskRelationRequest(src_task_id=src_task_id, dst_task_id=dst_task_id, type=type),
            ctx=ctx,
            settings=settings,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def task_status_set(
        task_id: UUID,
        status: TaskStatus,
        expected_version: int,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a task status with optimistic concurrency.

        Example:
            {
              "task_id": "00000000-0000-0000-0000-000000000201",
              "status": "in_progress",
              "expected_version": 1
            }
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        env_id = await _resolve_task_env(task_id)
        _require_env_attached(env_id, ctx)
        out: TaskResponse = await task_status_set_impl(
            task_id,
            status=status,
            expected_version=expected_version,
            ctx=ctx,
            settings=settings,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def task_list(
        request: TaskListRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """List tasks in an environment.

        Example:
            {
              "request": {
                "env_id": "00000000-0000-0000-0000-000000000101",
                "status": "pending",
                "limit": 5
              }
            }
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        request = await _resolve_env_refs(request)
        _require_env_attached(request.env_id, ctx)
        out: TaskListResponse = await task_list_impl(request, ctx=ctx, settings=settings)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def task_next(
        env_id: UUID,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return the next unblocked pending task for an environment.

        Example:
            {
              "env_id": "00000000-0000-0000-0000-000000000101"
            }
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        _require_env_attached(env_id, ctx)
        out = await task_next_impl(env_id, ctx=ctx, settings=settings)
        return _dump(out) if out is not None else None

    @mcp.tool()
    @_wrap
    async def task_tree(
        task_id: UUID,
        max_depth: int = 10,
        max_nodes: int = 200,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a bounded tree rooted at a task.

        Example:
            {
              "task_id": "00000000-0000-0000-0000-000000000201",
              "max_depth": 3,
              "max_nodes": 20
            }
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        out: TaskTreeResponse = await task_tree_impl(
            task_id,
            ctx=ctx,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def task_link_memory(
        request: TaskLinkMemoryRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Link a task to a memory.

        Example:
            {
              "request": {
                "task_id": "00000000-0000-0000-0000-000000000201",
                "memory_id": "00000000-0000-0000-0000-000000000001",
                "relation": "references"
              }
            }
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        env_id = await _resolve_task_env(request.task_id)
        _require_env_attached(env_id, ctx)
        out: TaskLinkMemoryResponse = await task_link_memory_impl(request, ctx=ctx, settings=settings)
        return _dump(out)

    # ---- Dream mode -------------------------------------------------------

    @mcp.tool()
    @_wrap
    async def dream_run_(
        request: DreamRunRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Trigger one or more dream-worker passes (decay/dedupe/promote).

        Example:
            {
              "request": {
                "env_id": "00000000-0000-0000-0000-000000000101",
                "modes": ["dedupe", "promote"],
                "wait": false
              }
            }

        With ``wait=True`` blocks until passes complete and returns
        per-(env, mode) reports. With ``wait=False`` (default) spawns
        the work as a background coordinator and returns immediately
        with the schedule. The dream-worker process and manual triggers
        share the same per-(env, mode) advisory lock, so a manual
        trigger that collides with the scheduler is reported as
        ``skipped_locked`` rather than racing.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: DreamRunResponse = await dream_run(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def dream_status_(
        request: DreamStatusRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Aggregate dream-worker state: recent runs, open proposals,
        heartbeats, summarizer kind, and a bounded LLM probe.

        Example:
            {
              "request": {
                "env_id": "00000000-0000-0000-0000-000000000101",
                "runs_per_mode": 5
              }
            }

        The LLM probe has a 2-second timeout and is best-effort; it
        will report ``status='error'`` rather than block the call.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: DreamStatusResponse = await dream_status(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def dream_proposals_list_(
        request: DreamProposalsListRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Browse ``dream_proposals`` with keyset pagination.

        Example:
            {
              "request": {
                "env_id": "00000000-0000-0000-0000-000000000101",
                "status": "open",
                "kind": "merge_candidate",
                "limit": 5
              }
            }

        Order is ``(created_at DESC, id DESC)``. Cursor is opaque and
        rejected with ``INVALID_INPUT`` if reused with different filters.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request)
        out: DreamProposalsListResponse = await dream_proposals_list(
            request, ctx=ctx,
        )
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def dream_review_(
        request: DreamReviewRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply a terminal action to an open proposal.

        Example:
            {
              "request": {
                "proposal_id": "00000000-0000-0000-0000-000000000401",
                "action": "reject",
                "notes": "Not a duplicate."
              }
            }

        ``accept`` dispatches by ``kind`` to the merge or promotion
        accept handler atomically: locks the proposal, locks involved
        memory rows in deterministic UUID order, validates same-env +
        same-kind invariants and optional ``expected_versions``, then
        creates the new memory + supersedes / lineages source rows in a
        single transaction with the proposal status update.

        ``amend`` is reserved for v1.5 and currently returns
        ``INVALID_INPUT``.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        out: DreamReviewResponse = await dream_review(request, ctx=ctx)
        return _dump(out)

    # ---- Environment Operations (v0.8) -----------------------------------

    @mcp.tool()
    @_wrap
    async def env_export_(
        request: EnvExportRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Export a full environment to a directory or .tar.gz archive.

        Example:
            {
              "request": {
                "env_id": "00000000-0000-0000-0000-000000000101",
                "format": "archive",
                "target_path": "exports/project-a-export",
                "include_embeddings": true
              }
            }

        Streams 15 env-scoped tables to JSONL plus an ``embeddings/``
        directory of named vector records, a ``manifest.json``, and
        BSD-style ``checksums.sha256``. Soft-deleted environments are
        rejected with ``ENV_DELETED``. By default ``include_grants`` and
        ``include_dream_history`` are off; ``include_embeddings`` and
        ``include_provenance`` are on. Reading uses REPEATABLE READ for
        cross-table consistency. ``format=archive`` produces a single
        ``.tar.gz`` and removes the intermediate directory; ``directory``
        leaves it in place. Decisions and playbooks are NOT split out —
        they're memory rows already inside ``memories.jsonl``.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        request = await _resolve_env_refs(request, allow_deleted=True)
        out: EnvExportResponse = await export_env(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_import_(
        request: EnvImportRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Import a v0.8 archive into a new or existing environment.

        Example:
            {
              "request": {
                "source_path": "exports/project-a-export.tar.gz",
                "target_env_name": "project-import",
                "mode": "fail",
                "dry_run": true
              }
            }

        Verifies ``checksums.sha256`` then ``manifest.json``; rejects
        future-version archives unless an explicit override is passed.
        Every UUID is remapped to a fresh value in the destination
        (source UUIDs are NEVER reused). Two-pass insert handles
        ``superseded_by`` self-references. Modes ``fail`` and ``skip``
        ship in v0.8 Phase 2; ``overwrite`` and ``merge`` arrive in
        Phase 4 (``NotImplementedError``). Bulk re-embedding (>10k
        memories on model mismatch) is blocked unless
        ``allow_bulk_reembed=True``. ``dry_run=True`` (the default)
        reports counts/conflicts without writing.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        out: EnvImportReport = await import_env(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_snapshot_(
        request: EnvSnapshotRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Take a labeled snapshot of an env. See §8.

        Example:
            {
              "request": {
                "env_id": "00000000-0000-0000-0000-000000000101",
                "label": "before-release-check",
                "include_embeddings": true
              }
            }

        Persists a tar.gz archive at ``<data_root>/snapshots/<env_id>/<snapshot_id>.memarchive.tar.gz`` and inserts a row in the ``snapshots`` table. Snapshots are not auto-pruned; the server logs a warning when the snapshot dir exceeds 10 GB.
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        request = await _resolve_env_refs(request, allow_deleted=True)
        out: EnvSnapshotResponse = await create_snapshot(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_restore_(
        request: EnvRestoreRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Restore from a snapshot. See §8 + §17.3.

        Example:
            {
              "request": {
                "snapshot_id": "00000000-0000-0000-0000-000000000501",
                "mode": "restore_to_new_env",
                "new_env_name": "project-restored"
              }
            }

        Two modes:
        - ``replace_env_in_place``: truncate the live env and reload from snapshot in a single PG transaction. Requires ``confirm_destroy=True``. The env_id and all memory UUIDs are preserved byte-for-byte (external lineage refs remain valid).
        - ``restore_to_new_env``: equivalent to env_import of the snapshot archive into a fresh env.
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        out: EnvRestoreResponse = await restore_snapshot(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_diff_(
        request: EnvDiffRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare two environments at one of four granularities.

        Example:
            {
              "request": {
                "env_a_id": "00000000-0000-0000-0000-000000000101",
                "env_b_id": "00000000-0000-0000-0000-000000000102",
                "granularity": "counts"
              }
            }

        ``counts`` returns per-table totals only. ``entity_keys`` adds
        bounded canonical entity-key set diffs. ``memory_hashes`` adds
        content-hash counts and samples. ``full`` adds bounded samples for
        tags, relations, tasks, graph nodes, and memory lineage.
        """
        ctx = await _resolve_ctx(
            agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names,
        )
        out: EnvDiffResponse = await diff_envs(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_clone_(
        request: EnvCloneRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Clone an environment into a fresh env, with optional filtered closure expansion.

        Example:
            {
              "request": {
                "src_env_id": "00000000-0000-0000-0000-000000000101",
                "new_name": "project-sandbox",
                "include_embeddings": true
              }
            }

        Source UUIDs are NEVER reused — all rows get fresh UUIDs in the destination. Closure expansion (§17.12) ensures
        FK integrity by auto-including supersession-chain targets, lineage parents (default 1 hop, max 5), referenced
        entities, and required tags. The response reports how many items were dragged in BEYOND the filter via
        ``closure_inclusions``. Embeddings are copied verbatim from the source vector store; if the destination env later
        wants different embeddings, a separate re-embed flow handles that.
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        out: EnvCloneResponse = await clone_env(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_merge_(
        request: EnvMergeRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Pairwise merge src env into dst env. See plan §6.

        Example:
            {
              "request": {
                "src_env_id": "00000000-0000-0000-0000-000000000102",
                "dst_env_id": "00000000-0000-0000-0000-000000000101",
                "dry_run": true,
                "allow_external_ref_rewrite": false
              }
            }

        Default policy: union-by-name for tags, by_canonical_key for entities (invokes ent_merge for collisions).
        Cross-env lineage edges exiting/entering src are rewritten to point at dst-side memories ONLY if
        ``allow_external_ref_rewrite=True``; otherwise aborts with ``EXTERNAL_REFS_BLOCKING``. With
        ``delete_src_after=True`` (default) the src env is soft-deleted at end — its UUID remains valid for downstream
        references. Embedding models must match unless ``allow_embedding_mismatch=True``.
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        out: EnvMergeResponse = await merge_envs(request, ctx=ctx)
        return _dump(out)

    @mcp.tool()
    @_wrap
    async def env_migrate_(
        request: EnvMigrateRequest,
        agent_id: UUID | None = None,
        attached_env_ids: list[UUID] | None = None,
        attached_env_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Bulk migration of memories from src env to dst env. See §3.2.

        Example:
            {
              "request": {
                "src_env_id": "00000000-0000-0000-0000-000000000101",
                "dst_env_id": "00000000-0000-0000-0000-000000000102",
                "mode": "copy",
                "dry_run": true
              }
            }

        Internally calls mem_copy or mem_move (per ``mode``) for each memory matching ``filter``.
        Best-effort batch: partial successes are possible; the report enumerates failures.
        Supersession chains are kept intact (the full chain migrates together) unless
        ``preserve_supersession_chain=False``. Use ``fail_fast=True`` to abort on first error.
        Embedding mismatches between src and dst envs require ``re_embed_if_model_mismatch=True``.
        """
        ctx = await _resolve_ctx(agent_id=agent_id, attached_env_ids=attached_env_ids, attached_env_names=attached_env_names)
        out: EnvMigrateResponse = await migrate_env(request, ctx=ctx)
        return _dump(out)

    log.info("mcp transport: registered %d tools", 65)
    _install_validation_hints(mcp)
    return mcp


def _install_validation_hints(mcp: FastMCP) -> None:
    """Translate FastMCP's auto-validation ``ValidationError`` into a
    structured ``[VALIDATION_FAILED]`` ToolError carrying did-you-mean hints.

    FastMCP runs Pydantic validation **inside** ``Tool.run`` (before our
    ``_wrap`` decorator gets a chance to catch anything). Its default
    behaviour is to catch the resulting :class:`pydantic.ValidationError`
    and re-raise it as a plain ``ToolError(f"Error executing tool {name}: {e}")``.
    The original exception is preserved on ``__cause__`` so we can recover
    enough context to compose actionable hints.

    We replace ``_tool_manager.call_tool`` with a wrapper that intercepts
    the framework's ``ToolError`` and (when its cause is a
    ``ValidationError``) re-raises our own ``ToolError`` whose message
    follows the canonical ``[CODE] message :: details_json`` contract.

    We can't patch the per-``Tool`` ``run`` method directly because
    ``Tool`` is a frozen Pydantic model.
    """
    from pydantic import ValidationError as _ValidationError

    from memory_mcp.errors import ValidationFailedError
    from memory_mcp.validation_hints import (
        build_hints,
        format_message,
        safe_error_payload,
    )

    tool_manager = mcp._tool_manager  # noqa: SLF001 — intentional, FastMCP exposes no hook
    original_call_tool = tool_manager.call_tool

    async def call_tool_with_hints(
        name: str,
        arguments: dict[str, Any],
        context: Any = None,
        convert_result: bool = False,
    ) -> Any:
        try:
            return await original_call_tool(
                name, arguments, context=context, convert_result=convert_result,
            )
        except ToolError as exc:
            cause = exc.__cause__
            if not isinstance(cause, _ValidationError):
                raise
            tool = tool_manager._tools.get(name)  # noqa: SLF001
            if tool is None:
                raise
            arg_model = tool.fn_metadata.arg_model
            errors = safe_error_payload(cause)
            hints = build_hints(arg_model, cause)
            msg = format_message(name, errors, hints)
            details = {"errors": errors, "hints": hints}
            rebuilt = ValidationFailedError(msg, **details)
            raise _format_tool_error(rebuilt) from cause

    tool_manager.call_tool = call_tool_with_hints  # type: ignore[method-assign]


mcp = build_mcp_server()


__all__ = ["build_mcp_server", "mcp"]
