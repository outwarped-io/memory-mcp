"""Decompose handler: caller-driven 1→N memory decomposition (v0.15.0 Phase 3).

This module is the entry point for the ``mem_decompose`` MCP tool. The
runtime contract is locked in by the Stage C1 design decision (see
``tasks/.../subtasks/.../plan.md`` Stage C). The transaction body lands
in C6 once C4 (decomposers skeleton + ORM model) and C5 (validation)
ship; this module is the C3 stub.

Structure mirrors :mod:`memory_mcp.composers` so the dream worker can
eventually delegate decompositions through here once the semantics line
up with whatever dream-side decompose proposal lands.
"""

from __future__ import annotations

import logging

from memory_mcp.errors import MemoryMCPError


class DecomposeNotImplementedError(MemoryMCPError):
    """C3 stub — raised until the C6 transaction body lands."""

    code = "NOT_IMPLEMENTED"


from memory_mcp.identity import AgentContext  # noqa: E402  (after error class for clarity)
from memory_mcp_schemas.decompose import (  # noqa: E402
    DecomposeLineageRow,
    DecomposeMode,
    MemDecomposeChild,
    MemDecomposeRequest,
    MemDecomposeResponse,
)

log = logging.getLogger(__name__)

__all__ = [
    "DecomposeLineageRow",
    "DecomposeMode",
    "DecomposeNotImplementedError",
    "MemDecomposeChild",
    "MemDecomposeRequest",
    "MemDecomposeResponse",
    "memory_decompose",
]


async def memory_decompose(
    request: MemDecomposeRequest,
    *,
    ctx: AgentContext,
) -> MemDecomposeResponse:
    """Decompose a source memory into N≥2 children.

    C3 stub — the surface is wired (request validation runs via
    Pydantic; a real call still raises so callers can detect the
    missing handler cleanly). The transaction body lands in v0.15.0
    Phase 3 C6.
    """
    raise DecomposeNotImplementedError(
        "mem_decompose handler not yet implemented in this build. "
        "Schema validation succeeded; transaction body lands in v0.15.0 Phase 3 C6."
    )
