"""Search pipeline.

Phase 1: ``lex`` (Postgres FTS) + ``sem`` (vector) + ``hybrid`` fusion.
Phase 2: adds ``graph`` (entity-expansion) into the hybrid fusion.
"""

from memory_mcp.search.api import (
    ConsistencyMode,
    ExpansionPreset,
    MemorySearchHit,
    MemorySearchRequest,
    MemorySearchResponse,
    ProjectionStatusEntry,
    SearchMode,
    memory_search,
)
from memory_mcp.search.auto_context import (
    AutoContextHit,
    AutoContextResponse,
    memory_auto_context,
)

__all__ = [
    "AutoContextHit",
    "AutoContextResponse",
    "ConsistencyMode",
    "ExpansionPreset",
    "MemorySearchHit",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "ProjectionStatusEntry",
    "SearchMode",
    "memory_auto_context",
    "memory_search",
]
