"""Allow playbook memory kind with steps and macro.

Revision ID: 0009_playbook_kind_and_macro
Revises: 0008_session_digest
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0009_playbook_kind_and_macro"
down_revision: str | None = "0008_session_digest"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KINDS_WITH_PLAYBOOK = (
    "('fact','procedure','event','decision','preference','observation',"
    "'journal_entry','playbook','session_digest','snippet')"
)
_KINDS_WITHOUT_PLAYBOOK = (
    "('fact','procedure','event','decision','preference','observation',"
    "'journal_entry','session_digest','snippet')"
)


def upgrade() -> None:
    op.execute("ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_kind_check")
    op.execute(
        "ALTER TABLE memories ADD CONSTRAINT memories_kind_check "
        f"CHECK (kind IN {_KINDS_WITH_PLAYBOOK})"
    )
    op.add_column(
        "memories",
        sa.Column("steps", postgresql.ARRAY(sa.Text()), nullable=True),
    )
    op.add_column("memories", sa.Column("macro", sa.Text(), nullable=True))
    op.execute(
        "CREATE UNIQUE INDEX ix_memories_macro_per_env "
        "ON memories (env_id, lower(macro)) WHERE macro IS NOT NULL"
    )


def downgrade() -> None:
    op.execute(
        "DO $$ BEGIN RAISE WARNING "
        "'Downgrading 0009 maps playbook memories to procedure and drops steps/macro.'; END $$;"
    )
    op.execute("UPDATE memories SET kind = 'procedure' WHERE kind = 'playbook'")
    op.drop_index("ix_memories_macro_per_env", table_name="memories")
    op.drop_column("memories", "macro")
    op.drop_column("memories", "steps")
    op.execute("ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_kind_check")
    op.execute(
        "ALTER TABLE memories ADD CONSTRAINT memories_kind_check "
        f"CHECK (kind IN {_KINDS_WITHOUT_PLAYBOOK})"
    )
