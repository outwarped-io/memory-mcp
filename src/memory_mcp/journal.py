"""Memory journal — fast append-only observation writes.

The journal is a thin convenience layer over :func:`memory_write` for
short-form events and observations a session wants to capture without
deciding their final shape.

Design choices:

* ``kind`` is always ``observation``. To promote a journal entry into a
  ``fact`` / ``decision`` / etc., use :func:`memory_promote` (Phase 2).
* ``agent_id`` is derived from :class:`AgentContext`. Callers cannot
  forge a journal entry on behalf of another agent.
* No ``title`` parameter on the public surface — journal entries are
  short-form. (The underlying ``memories.title`` column is left ``NULL``;
  ``memory_search`` lex mode falls back to body content.)
* Returns the canonical :class:`MemoryResponse` so callers get the
  ``id`` and ``version`` immediately for follow-up updates.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from memory_mcp.config import Settings
from memory_mcp.db.types import MemoryKind
from memory_mcp.identity import AgentContext
from memory_mcp.memories import (
    MemoryResponse,
    MemoryWriteRequest,
    memory_write,
)

from memory_mcp_schemas.journal import (
    JournalRequest,
)

__all__ = [
    "JournalRequest",
    "memory_journal",
]


async def memory_journal(
    request: JournalRequest,
    *,
    ctx: AgentContext,
    settings: Settings | None = None,
) -> MemoryResponse:
    """Append a short-form ``observation`` memory.

    Equivalent to ``memory_write(kind=observation, body=request.content,
    title=None, ...)`` — exists as a separate tool so the surface is
    discoverable for "log this" use cases without forcing the agent to
    spell out the kind/title.
    """
    write_req = MemoryWriteRequest(
        kind=MemoryKind.observation,
        body=request.content,
        env_id=request.env_id,
        tags=request.tags,
        metadata=request.metadata,
        salience=request.salience,
    )
    return await memory_write(write_req, ctx=ctx, settings=settings)
