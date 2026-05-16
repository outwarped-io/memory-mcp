"""Add soft-delete columns to environments.

Revision ID: 0013_envops_softdelete
Revises: 0012_dream_decision_conflicts
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0013_envops_softdelete"
down_revision: str | None = "0012_dream_decision_conflicts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE environments
        ADD COLUMN status TEXT NOT NULL DEFAULT 'active',
        ADD COLUMN deleted_at TIMESTAMP WITH TIME ZONE NULL,
        ADD CONSTRAINT environments_status_check
        CHECK (status IN ('active', 'deleted'))
    """)
    op.execute("""
        CREATE INDEX environments_active_idx
        ON environments(name)
        WHERE status = 'active'
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS environments_active_idx")
    op.execute("""
        ALTER TABLE environments
        DROP CONSTRAINT IF EXISTS environments_status_check,
        DROP COLUMN IF EXISTS deleted_at,
        DROP COLUMN IF EXISTS status
    """)
