"""Compose handler: caller-driven N→1 memory aggregation (v0.15.0 Phase 2).

This module is the entry point for the ``mem_compose`` MCP tool. The
runtime contract is locked in by the Stage B1 design decision (see
``tasks/.../subtasks/.../plan.md`` Stage B). The transaction body lives
in this module so the dream worker handlers (``_accept_merge`` /
``_accept_promotion``) can eventually delegate here once parity tests
prove the refactor is safe.

B2 (this commit) only stands up the tool surface and re-exports the
schemas. The atomic transaction lands in B3.
"""

from __future__ import annotations

import logging

from memory_mcp.errors import MemoryMCPError


class ComposeNotImplementedError(MemoryMCPError):
    """B2 stub — raised until the B3 transaction body lands."""

    code = "NOT_IMPLEMENTED"


from memory_mcp.identity import AgentContext
from memory_mcp_schemas.compose import (
    ComposeLineageRow,
    ComposeMode,
    ComposeTagPolicy,
    MemComposeRequest,
    MemComposeResponse,
    MemComposeTarget,
)

log = logging.getLogger(__name__)

__all__ = [
    "ComposeLineageRow",
    "ComposeMode",
    "ComposeTagPolicy",
    "MemComposeRequest",
    "MemComposeResponse",
    "MemComposeTarget",
    "memory_compose",
]


async def memory_compose(
    request: MemComposeRequest,
    *,
    ctx: AgentContext,
) -> MemComposeResponse:
    """Compose N≥2 source memories into a single new memory.

    B2 stub — the surface is wired (request validation runs via Pydantic;
    a real call still raises so callers can detect the missing handler
    cleanly). B3 lands the transaction body.
    """
    raise ComposeNotImplementedError(
        "mem_compose handler not yet implemented in this build. "
        "Schema validation succeeded; transaction body lands in v0.15.0 Phase 2 B3."
    )
