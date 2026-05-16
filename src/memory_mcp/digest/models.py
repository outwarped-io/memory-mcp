"""Backward-compat shim: re-export DTOs from memory_mcp_schemas.digest."""

from __future__ import annotations

from memory_mcp_schemas.digest import *  # noqa: F401,F403
from memory_mcp_schemas.digest import (
    DigestRequest,
    ResumeRequest,
    DigestSections,
    DigestResponse,
    DigestMemoryEntry,
    ResumeStats,
    ResumeResponse,
)

__all__ = ['DigestRequest', 'ResumeRequest', 'DigestSections', 'DigestResponse', 'DigestMemoryEntry', 'ResumeStats', 'ResumeResponse']
