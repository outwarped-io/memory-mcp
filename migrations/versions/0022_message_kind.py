"""Allow ``message`` memory kind for inter-agent inbox (v0.17).

Adds ``message`` to the ``memories.kind`` CHECK constraint. No new columns;
no new tables. Inbox messages are addressed via the existing
``entity_links`` column to ``channel``-kind entities (entity ``kind`` is
free-form text and needs no migration).

Revision ID: 0022_message_kind
Revises: 0021_decompose_operations
Create Date: 2026-06-03
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0022_message_kind"
down_revision: str | None = "0021_decompose_operations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KINDS_WITH_MESSAGE = (
    "('fact','procedure','event','decision','preference','observation',"
    "'journal_entry','playbook','session_digest','snippet','message')"
)
_KINDS_WITHOUT_MESSAGE = (
    "('fact','procedure','event','decision','preference','observation',"
    "'journal_entry','playbook','session_digest','snippet')"
)


def upgrade() -> None:
    op.execute("ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_kind_check")
    op.execute(f"ALTER TABLE memories ADD CONSTRAINT memories_kind_check CHECK (kind IN {_KINDS_WITH_MESSAGE})")


def downgrade() -> None:
    op.execute(
        "DO $$ BEGIN RAISE WARNING "
        "'Downgrading 0022 maps message memories to observation '"
        "'(closest existing kind).'; END $$;"
    )
    op.execute("UPDATE memories SET kind = 'observation' WHERE kind = 'message'")
    op.execute("ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_kind_check")
    op.execute(f"ALTER TABLE memories ADD CONSTRAINT memories_kind_check CHECK (kind IN {_KINDS_WITHOUT_MESSAGE})")
