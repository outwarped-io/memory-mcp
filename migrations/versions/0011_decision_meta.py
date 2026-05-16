"""Add structured decision metadata.

Revision ID: 0011_decision_meta
Revises: 0010_tasks_table
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_decision_meta"
down_revision: str | None = "0010_tasks_table"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column("decision_meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        "ix_memories_decision_meta_gin",
        "memories",
        ["decision_meta"],
        postgresql_using="gin",
    )
    op.execute(
        "CREATE INDEX ix_memories_decision_status "
        "ON memories (env_id, ((decision_meta->>'status'))) "
        "WHERE decision_meta IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memories_decision_status")
    op.drop_index("ix_memories_decision_meta_gin", table_name="memories")
    op.drop_column("memories", "decision_meta")
