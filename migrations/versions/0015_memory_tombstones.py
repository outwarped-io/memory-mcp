"""Add memory_tombstones table for hard-delete saga.

Revision ID: 0015_memory_tombstones
Revises: 0014_snapshots_table

The hard-delete contract (see ``memory_mcp.memories.memory_hard_delete``)
removes the canonical memory row but keeps a tombstone marker that:

* Lets ``mem_get`` return a recognisable 404 with hint ``"see tombstone
  <id>"`` instead of a bare not-found.
* Anchors lineage edges that referenced the deleted memory. The
  edge table's ``src_memory_id`` / ``dst_memory_id`` foreign keys
  ``ON DELETE CASCADE`` would otherwise drop the edges silently;
  the projection workers need at least one row to point at so the
  Neo4j eviction event has a stable subject id.
* Carries audit data (deleted_at, deleted_by_agent_id, reason) so an
  operator can correlate a leak/recovery with an actual delete call
  weeks later.

Tombstones never carry the deleted body, embedding, or tag set —
those are the values we wanted erased in the first place.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0015_memory_tombstones"
down_revision: str | None = "0014_snapshots_table"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE memory_tombstones (
            id UUID PRIMARY KEY,
            env_id UUID NOT NULL REFERENCES environments(id) ON DELETE RESTRICT,
            deleted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            deleted_by_agent_id UUID,
            reason TEXT NOT NULL,
            original_kind TEXT,
            original_status TEXT
        )
    """)
    op.execute("""
        CREATE INDEX memory_tombstones_env_deleted_at_idx
        ON memory_tombstones (env_id, deleted_at DESC)
    """)
    op.execute("""
        CREATE INDEX memory_tombstones_deleted_by_idx
        ON memory_tombstones (deleted_by_agent_id)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS memory_tombstones_deleted_by_idx")
    op.execute("DROP INDEX IF EXISTS memory_tombstones_env_deleted_at_idx")
    op.execute("DROP TABLE IF EXISTS memory_tombstones")
