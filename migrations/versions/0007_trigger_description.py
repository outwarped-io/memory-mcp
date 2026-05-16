"""Add trigger-conditioned auto-context field.

Revision ID: 0007_trigger_description
Revises: 0006_import_source_type
Create Date: 2026-05-12

Qdrant projection decision: v0.6 stores trigger embeddings as a second named
vector (``trigger``) on the existing per-env memory point, alongside the body
vector (``body``). This keeps body semantic search isolated from auto-context
matching without introducing a second collection or duplicating payload state.
The canonical migration only adds the nullable Postgres text column; existing
Qdrant collections are rebuildable projections.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0007_trigger_description"
down_revision: str | None = "0006_import_source_type"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("trigger_description", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("memories", "trigger_description")
