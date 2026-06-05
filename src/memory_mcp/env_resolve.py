"""Centralized resolution of env_name/env_names friendly fields to UUIDs.

Called by every tool wrapper that takes a request shape containing env refs.
v0.9 wave 1 wires this only into ``mem_search``; wave 2 will roll it out
across the remaining schemas.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from memory_mcp.envs import get_env_by_name_ci
from memory_mcp.errors import EnvRefBothProvidedError

T = TypeVar("T", bound=BaseModel)


async def _resolve_env_refs[T: BaseModel](
    request: T,
    *,
    allow_deleted: bool = False,
) -> T:
    """Resolve env_name(s) to env_id(s) in a request model.

    Mutually exclusive: a request may provide ``env_id`` OR ``env_name`` but
    not both for the same slot (raises ``ENV_REF_BOTH_PROVIDED``). Same for
    ``env_ids``/``env_names``.

    Returns a ``model_copy`` of the request with name fields set to ``None``
    and id fields populated. Downstream code sees only UUIDs.

    ``allow_deleted`` is forwarded to ``get_env_by_name_ci``. Default False
    (writes and most reads). Set True for lineage / provenance / env_get /
    env_diff paths in wave 2.
    """
    updates: dict[str, Any] = {}

    if hasattr(request, "env_name") and hasattr(request, "env_id"):
        name = getattr(request, "env_name", None)
        env_id = getattr(request, "env_id", None)
        if name is not None and env_id is not None:
            raise EnvRefBothProvidedError(field="env")
        if name is not None:
            env = await get_env_by_name_ci(name, include_deleted=allow_deleted)
            updates["env_id"] = env.id
            updates["env_name"] = None

    if hasattr(request, "env_names") and hasattr(request, "env_ids"):
        names = getattr(request, "env_names", None)
        env_ids = getattr(request, "env_ids", None)
        if names is not None and env_ids is not None:
            raise EnvRefBothProvidedError(field="env_list")
        if names is not None:
            resolved = [(await get_env_by_name_ci(name, include_deleted=allow_deleted)).id for name in names]
            updates["env_ids"] = resolved
            updates["env_names"] = None

    # TODO wave 2: src_env_name/dst_env_name/target_env_name/new_env_name pairs.
    # Out of scope for wave 1: mem_search only has env_names.
    if not updates:
        return request
    return request.model_copy(update=updates)


__all__ = ["_resolve_env_refs"]
