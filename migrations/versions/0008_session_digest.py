"""Allow session digest memory kind and provenance.

Revision ID: 0008_session_digest
Revises: 0007_trigger_description
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0008_session_digest"
down_revision: str | None = "0007_trigger_description"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KINDS_WITH_DIGEST = (
    "('fact','procedure','event','decision','preference','observation','journal_entry','session_digest','snippet')"
)
_KINDS_WITHOUT_DIGEST = "('fact','procedure','event','decision','preference','observation','snippet')"
_SOURCE_TYPES_WITH_DIGEST = (
    "('session','file','import','url','llm','dream','digest','digest-template','user','agent','other')"
)
_SOURCE_TYPES_WITHOUT_DIGEST = "('session','file','import','url','llm','dream','user','agent','other')"


def upgrade() -> None:
    op.execute("ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_kind_check")
    op.execute(f"ALTER TABLE memories ADD CONSTRAINT memories_kind_check CHECK (kind IN {_KINDS_WITH_DIGEST})")
    op.execute("ALTER TABLE memory_sources DROP CONSTRAINT IF EXISTS memory_sources_source_type_check")
    op.execute(
        "ALTER TABLE memory_sources ADD CONSTRAINT memory_sources_source_type_check "
        f"CHECK (source_type IN {_SOURCE_TYPES_WITH_DIGEST})"
    )


def downgrade() -> None:
    op.execute(
        "DO $$ BEGIN RAISE WARNING "
        "'Downgrading 0008 maps session_digest memories to observation and "
        "digest source types to other; non-0008 values are left unchanged.'; END $$;"
    )
    op.execute("UPDATE memories SET kind = 'observation' WHERE kind = 'session_digest'")
    op.execute("ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_kind_check")
    op.execute(f"ALTER TABLE memories ADD CONSTRAINT memories_kind_check CHECK (kind IN {_KINDS_WITHOUT_DIGEST})")
    op.execute("UPDATE memory_sources SET source_type = 'other' WHERE source_type IN ('digest','digest-template')")
    op.execute("ALTER TABLE memory_sources DROP CONSTRAINT IF EXISTS memory_sources_source_type_check")
    op.execute(
        "ALTER TABLE memory_sources ADD CONSTRAINT memory_sources_source_type_check "
        f"CHECK (source_type IN {_SOURCE_TYPES_WITHOUT_DIGEST})"
    )
