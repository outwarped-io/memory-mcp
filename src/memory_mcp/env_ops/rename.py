"""Environment rename/reconfiguration implementation for v0.8 env operations."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from memory_mcp_schemas.env_ops import EnvRenameRequest, EnvRenameResponse
from sqlalchemy import func, select

from memory_mcp import rbac
from memory_mcp.db.models import Environment, Outbox
from memory_mcp.db.outbox import enqueue_event
from memory_mcp.db.postgres import session_scope
from memory_mcp.db.types import OutboxAggregateType, OutboxOp
from memory_mcp.errors import InvalidInputError, MemoryMCPError, NotFoundError
from memory_mcp.identity import AgentContext

log = logging.getLogger(__name__)


NO_AUTO_REEMBED_WARNING = (
    "Changing default_embedding_model_id does not re-embed existing memories; it only affects new memories."
)


class ConflictError(MemoryMCPError):
    """Environment name is already in use by another active environment."""

    code = "ENV_NAME_TAKEN"


async def rename_env(request: EnvRenameRequest, *, ctx: AgentContext) -> EnvRenameResponse:
    """Update mutable fields on an Environment row. See §10.

    Updateable fields:
    - name: must be unique (case-insensitive) across active envs; updates display_name.
    - default_embedding_model_id: does NOT re-embed existing memories; only affects new ones.
    - retention_policy: mutates JSONB; takes effect at next dream-run.

    The env_id is IMMUTABLE (lineage anchor — see §17.4).
    """

    async with session_scope() as session:
        env = await session.get(Environment, request.env_id)
        if env is None:
            raise NotFoundError(f"environment {request.env_id} not found", env_id=str(request.env_id))
        if getattr(env, "status", "active") == "deleted":
            exc = NotFoundError(f"environment {request.env_id} is deleted", env_id=str(request.env_id))
            exc.code = "ENV_DELETED"
            raise exc

        rbac.require("write", request.env_id, ctx)

        if (
            request.new_name is None
            and request.new_default_embedding_model_id is None
            and request.new_retention_policy is None
        ):
            exc = InvalidInputError("at least one new_* field must be set")
            exc.code = "NOTHING_TO_RENAME"
            raise exc

        if request.new_name is not None:
            _validate_name(request.new_name)
        if request.new_default_embedding_model_id is not None:
            _validate_embedding_model_id(request.new_default_embedding_model_id)
        if request.new_retention_policy is not None and not isinstance(request.new_retention_policy, dict):
            raise InvalidInputError("new_retention_policy must be a JSON object")

        old_name = env.name
        old_model = env.default_embedding_model_id
        old_retention_policy = dict(env.retention_policy or {})

        if request.new_name is not None:
            await _ensure_name_available(session, request.new_name, env_id=request.env_id)

        changed_fields: list[str] = []
        if request.new_name is not None and request.new_name != env.name:
            env.name = request.new_name
            if hasattr(env, "display_name"):
                env.display_name = _slugify(request.new_name)
            changed_fields.append("name")

        if (
            request.new_default_embedding_model_id is not None
            and request.new_default_embedding_model_id != env.default_embedding_model_id
        ):
            env.default_embedding_model_id = request.new_default_embedding_model_id
            changed_fields.append("default_embedding_model_id")
            log.warning(
                "env_rename changed default_embedding_model_id for env %s; existing memories will not be re-embedded",
                request.env_id,
            )

        if request.new_retention_policy is not None and request.new_retention_policy != (env.retention_policy or {}):
            env.retention_policy = request.new_retention_policy
            changed_fields.append("retention_policy")

        if hasattr(env, "updated_at"):
            env.updated_at = func.now()

        if changed_fields:
            await _emit_env_renamed(
                session,
                env_id=request.env_id,
                old_name=old_name,
                new_name=env.name,
                changed_fields=changed_fields,
                old_model=old_model,
                new_model=env.default_embedding_model_id,
                old_retention_policy=old_retention_policy,
                new_retention_policy=dict(env.retention_policy or {}),
            )

        return EnvRenameResponse(
            env_id=env.id,
            name=env.name,
            default_embedding_model_id=env.default_embedding_model_id,
            retention_policy=dict(env.retention_policy or {}),
            changed_fields=changed_fields,
            warning=NO_AUTO_REEMBED_WARNING if "default_embedding_model_id" in changed_fields else None,
        )


async def _ensure_name_available(session: Any, new_name: str, *, env_id: UUID) -> None:
    collision = await session.scalar(
        select(Environment.id)
        .where(func.lower(Environment.name) == new_name.lower())
        .where(Environment.status == "active")
        .where(Environment.id != env_id)
        .limit(1),
    )
    if collision is not None:
        raise ConflictError(f"environment name already exists: {new_name!r}", name=new_name)


def _validate_name(name: str) -> None:
    if not isinstance(name, str):
        raise InvalidInputError("new_name must be a string")
    if name != name.strip():
        raise InvalidInputError("new_name must not have leading or trailing whitespace")
    if not 1 <= len(name) <= 255:
        raise InvalidInputError("new_name length must be between 1 and 255 characters")


def _validate_embedding_model_id(model_id: str) -> None:
    if not isinstance(model_id, str) or not model_id.strip():
        raise InvalidInputError("new_default_embedding_model_id must be a non-empty string")


def _slugify(value: str) -> str:
    slug = "-".join(value.strip().lower().split())
    return slug[:255]


async def _emit_env_renamed(
    session: Any,
    *,
    env_id: UUID,
    old_name: str,
    new_name: str,
    changed_fields: list[str],
    old_model: str,
    new_model: str,
    old_retention_policy: dict[str, Any],
    new_retention_policy: dict[str, Any],
) -> None:
    aggregate_version = await session.scalar(
        select(func.coalesce(func.max(Outbox.aggregate_version), 0) + 1).where(
            Outbox.aggregate_type == OutboxAggregateType.env.value,
            Outbox.aggregate_id == env_id,
        )
    )
    await enqueue_event(
        session,
        aggregate_type=OutboxAggregateType.env,
        aggregate_id=env_id,
        aggregate_version=int(aggregate_version or 1),
        env_id=env_id,
        op=OutboxOp.update,
        payload={
            "event": "EnvRenamed",
            "env_id": str(env_id),
            "old_name": old_name,
            "new_name": new_name,
            "changed_fields": list(changed_fields),
            "old_default_embedding_model_id": old_model,
            "new_default_embedding_model_id": new_model,
            "old_retention_policy": old_retention_policy,
            "new_retention_policy": new_retention_policy,
        },
    )
