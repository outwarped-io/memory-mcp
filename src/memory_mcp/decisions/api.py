"""Decision validation and ADR export APIs."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memory_mcp import rbac
from memory_mcp.db.models import Memory
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import MemoryKind
from memory_mcp.decisions.models import AdrExportResponse, DecisionMeta
from memory_mcp.decisions.template import render_adr
from memory_mcp.errors import InvalidInputError, NotFoundError
from memory_mcp.identity import AgentContext


async def validate_decision_meta(
    payload: dict[str, Any] | None,
    env_id: UUID,
    session: AsyncSession,
) -> DecisionMeta | None:
    """Validate optional decision metadata and same-env supersession target."""
    if payload is None:
        return None
    try:
        meta = DecisionMeta.model_validate(payload)
    except (ValidationError, ValueError) as exc:
        raise InvalidInputError(f"invalid decision_meta: {exc}") from exc

    if meta.superseded_by is not None:
        target = (
            await session.execute(
                select(Memory).where(
                    Memory.id == meta.superseded_by,
                    Memory.env_id == env_id,
                    Memory.kind == MemoryKind.decision.value,
                )
            )
        ).scalar_one_or_none()
        if target is None:
            raise InvalidInputError(
                "decision_meta.superseded_by must reference an existing kind=decision memory in the same env",
                superseded_by=str(meta.superseded_by),
                env_id=str(env_id),
            )
    return meta


async def adr_export(memory_id: UUID, ctx: AgentContext) -> AdrExportResponse:
    """Export a decision memory as ADR markdown."""
    async with session_scope() as session:
        memory = await session.get(Memory, memory_id)
        if memory is None:
            raise NotFoundError(f"memory {memory_id} not found", memory_id=str(memory_id))
        if ctx.attached_env_ids and memory.env_id not in ctx.attached_env_ids:
            raise NotFoundError(
                f"memory {memory_id} not visible in attached envs",
                memory_id=str(memory_id),
            )
        rbac.require("read", memory.env_id, ctx)
        if memory.kind != MemoryKind.decision.value:
            raise InvalidInputError("adr_export is only valid for kind=decision")
        try:
            meta = DecisionMeta.model_validate(memory.decision_meta) if memory.decision_meta is not None else None
        except ValidationError as exc:
            raise InvalidInputError(f"decision_meta is malformed: {exc}") from exc
        return AdrExportResponse(
            markdown=render_adr(memory, meta),
            status=meta.status.value if meta else None,
            memory_id=memory.id,
        )
