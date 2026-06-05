"""Backward-compat shim: re-export DTOs from memory_mcp_schemas.playbooks."""

from __future__ import annotations

from memory_mcp_schemas.playbooks import *  # noqa: F401,F403
from memory_mcp_schemas.playbooks import (
    PlaybookInvokeResponse,
)

__all__ = ["PlaybookInvokeResponse"]
