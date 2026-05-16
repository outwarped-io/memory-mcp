"""Allow import provenance source type.

Revision ID: 0006_import_source_type
Revises: 0005_explore_api_sprint_b
Create Date: 2026-05-11
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006_import_source_type"
down_revision: str | None = "0005_explore_api_sprint_b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SOURCE_TYPES_WITH_IMPORT = "('session','file','import','url','llm','dream','user','agent','other')"
_SOURCE_TYPES_WITHOUT_IMPORT = "('session','file','url','llm','dream','user','agent','other')"


def upgrade() -> None:
    op.execute("ALTER TABLE memory_sources DROP CONSTRAINT IF EXISTS memory_sources_source_type_check")
    op.execute(
        "ALTER TABLE memory_sources ADD CONSTRAINT memory_sources_source_type_check "
        f"CHECK (source_type IN {_SOURCE_TYPES_WITH_IMPORT})"
    )


def downgrade() -> None:
    op.execute("UPDATE memory_sources SET source_type = 'other' WHERE source_type = 'import'")
    op.execute("ALTER TABLE memory_sources DROP CONSTRAINT IF EXISTS memory_sources_source_type_check")
    op.execute(
        "ALTER TABLE memory_sources ADD CONSTRAINT memory_sources_source_type_check "
        f"CHECK (source_type IN {_SOURCE_TYPES_WITHOUT_IMPORT})"
    )
