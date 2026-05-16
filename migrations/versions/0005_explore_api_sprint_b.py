"""Explore-API Sprint B — graph + provenance indexes.

Adds the indexes needed by Sprint B's graph + provenance tools. The
``memory_lineage`` primary key already covers ``(parent_memory_id,
child_memory_id, relation)`` for descendants walks rooted at parent; this
migration adds the inverse shape for ancestors walks rooted at child. The
tail-column ``parent_memory_id`` covers the recursive CTE join without a heap
fetch.

``memory_sources`` had no scope index today. This migration adds three
plain-btree scope + keyset indexes for ``memory_id``, ``source_type``, and
``agent_id``. The ``agent_id`` variant is partial because the column is
nullable, keeping the index small.

Plain btree (no ``DESC`` qualifiers) is intentional per the Sprint A
rubber-duck lesson: Postgres btrees scan bidirectionally, so the same indexes
serve both ascending and descending keyset traversals.

Idempotency: every CREATE INDEX uses ``IF NOT EXISTS`` so re-running the
migration on a partially-applied database is safe.

Revision ID: 0005_explore_api_sprint_b
Revises: 0004_explore_api_sprint_a
Create Date: 2026-05-10
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_explore_api_sprint_b"
down_revision: str | None = "0004_explore_api_sprint_a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    version = conn.execute(sa.text("SHOW server_version_num")).scalar_one()
    if int(version) < 140000:
        raise RuntimeError(
            "memory-mcp Sprint B requires PostgreSQL 14+ (CYCLE clause). "
            f"Detected server_version_num={version}."
        )

    # Plain btree (no DESC) so Postgres can do forward OR backward scans.
    op.execute(
        "CREATE INDEX IF NOT EXISTS memory_lineage_child_relation_idx "
        "ON memory_lineage (child_memory_id, relation, parent_memory_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS memory_sources_memory_id_created_at_id_idx "
        "ON memory_sources (memory_id, created_at, id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS memory_sources_source_type_created_at_id_idx "
        "ON memory_sources (source_type, created_at, id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS memory_sources_agent_id_created_at_id_idx "
        "ON memory_sources (agent_id, created_at, id) "
        "WHERE agent_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS memory_sources_agent_id_created_at_id_idx")
    op.execute("DROP INDEX IF EXISTS memory_sources_source_type_created_at_id_idx")
    op.execute("DROP INDEX IF EXISTS memory_sources_memory_id_created_at_id_idx")
    op.execute("DROP INDEX IF EXISTS memory_lineage_child_relation_idx")
