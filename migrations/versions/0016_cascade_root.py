"""Add cascade_root to memory_tombstones.

Revision ID: 0016_cascade_root
Revises: 0015_memory_tombstones
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0016_cascade_root"
down_revision: str | None = "0015_memory_tombstones"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE memory_tombstones ADD COLUMN cascade_root UUID NULL")
    op.execute("CREATE INDEX ix_memory_tombstones_cascade_root ON memory_tombstones (cascade_root)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memory_tombstones_cascade_root")
    op.execute("ALTER TABLE memory_tombstones DROP COLUMN IF EXISTS cascade_root")
