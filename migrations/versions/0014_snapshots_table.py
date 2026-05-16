"""Add snapshots table.

Revision ID: 0014_snapshots_table
Revises: 0013_envops_softdelete
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0014_snapshots_table"
down_revision: str | None = "0013_envops_softdelete"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE snapshots (
            id UUID PRIMARY KEY,
            env_id UUID NOT NULL REFERENCES environments(id) ON DELETE RESTRICT,
            label TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            created_by_agent_id UUID,
            path TEXT NOT NULL,
            size_bytes BIGINT NOT NULL,
            checksum_sha256 TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            notes TEXT,
            CONSTRAINT snapshots_env_label_uniq UNIQUE (env_id, label)
        )
    """)
    op.execute("""
        CREATE INDEX snapshots_env_created_idx
        ON snapshots (env_id, created_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS snapshots_env_created_idx")
    op.execute("DROP TABLE IF EXISTS snapshots")
