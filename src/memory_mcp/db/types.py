"""Backward-compat shim: re-export domain enums from ``memory_mcp_schemas``.

This module historically owned the enum + lifecycle helper surface. It
now re-exports everything from :mod:`memory_mcp_schemas.enums` so the
server, the projection worker, the dream worker, and the Python client
SDK all share a single source of truth.

Do NOT add new symbols here. Add them to ``memory_mcp_schemas.enums``.

The Postgres schema uses ``text + CHECK`` rather than native ``ENUM``
types (see migration ``0001_v1_initial`` for rationale). Python-side we
map them to ``StrEnum`` for type safety and tool-surface validation.
"""

from __future__ import annotations

# Re-export the entire schemas.enums surface. ``*``-import is intentional:
# every public symbol over there is part of the historical contract.
from memory_mcp_schemas.enums import (  # noqa: F401
    DecisionStatus,
    DreamMode,
    DreamProposalKind,
    DreamProposalStatus,
    DreamReviewAction,
    DreamRunStatus,
    DreamRunTrigger,
    GrantRole,
    GraphNodeType,
    LineageRelation,
    MemoryKind,
    MemorySourceType,
    MemoryStatus,
    OutboxAggregateType,
    OutboxDeliveryStatus,
    OutboxOp,
    OutboxSink,
    ProjectionStatus,
    SummarizerKind,
    TaskRelationKind,
    TaskStatus,
    is_valid_task_transition,
    is_valid_transition,
)

__all__ = [
    "DecisionStatus",
    "DreamMode",
    "DreamProposalKind",
    "DreamProposalStatus",
    "DreamReviewAction",
    "DreamRunStatus",
    "DreamRunTrigger",
    "GrantRole",
    "GraphNodeType",
    "LineageRelation",
    "MemoryKind",
    "MemorySourceType",
    "MemoryStatus",
    "OutboxAggregateType",
    "OutboxDeliveryStatus",
    "OutboxOp",
    "OutboxSink",
    "ProjectionStatus",
    "SummarizerKind",
    "TaskRelationKind",
    "TaskStatus",
    "is_valid_task_transition",
    "is_valid_transition",
]
