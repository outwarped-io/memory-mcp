"""Explore-API Sprint A — keyset-paging indexes for browse/facet tools.

Adds the indexes needed by the Sprint A browse tools (mem_browse,
mem_facets, ent_browse, rel_browse). All four tools rely on keyset
pagination over ``(order_value, id)`` pairs, plus prefix lookups against
``entities.normalized_name`` / ``entity_aliases.normalized_alias``. The
existing v1 schema already carries the wider single-column indexes; this
migration adds the composite + ``text_pattern_ops`` indexes the new
tools need to plan efficiently.

Idempotency: every CREATE INDEX uses ``IF NOT EXISTS`` so re-running the
migration on a partially-applied database is safe. Indexes are created
with ``CONCURRENTLY`` where the DDL is supported outside a transaction.

Why not ``relations_env_type_idx``: the v1 migration already created
``relations_env_type_idx (env_id, type)``. Sprint A's ``rel_browse``
keyset adds ``id`` for the tiebreak, but the planner can reuse the
existing index for the (env_id, type) range scan and apply an in-memory
tie-break on ``id`` — adding a redundant composite index is not worth
the write cost. Same for ``memory_tags_tag_idx``; the existing single-
column index suffices for the facet groupby.

What we DO add (7 indexes):

* ``memories_env_status_updated_at_id_idx`` — covers ``mem_browse``'s
  default ``order_by="updated_at"`` keyset traversal. Filter columns
  first (``env_id, status``), order columns next (``updated_at, id``).
  Postgres btree is bidirectional, so this index serves both ASC and
  ``ORDER BY ... DESC, id DESC`` (reverse scan).

* ``memories_env_status_created_at_id_idx`` — same pattern for the
  ``order_by="created_at"`` option.

* ``entities_env_kind_canonical_name_id_idx`` — covers ``ent_browse``'s
  default ``order_by="canonical_name"`` keyset traversal when no
  ``name_prefix`` is supplied.

* ``entities_env_kind_norm_name_pattern_idx`` — ``text_pattern_ops``
  btree on ``normalized_name`` so ``LIKE 'prefix%'`` queries from
  ``ent_browse(name_prefix=...)`` use the index rather than seq-scan.

* ``entity_aliases_normalized_alias_pattern_idx`` — same
  ``text_pattern_ops`` treatment for alias prefix matches.

* ``relations_env_type_created_at_id_idx`` — covers ``rel_browse``'s
  ``ORDER BY created_at, id`` when ``types[]`` filter is present.

* ``relations_env_created_at_id_idx`` — same for the un-filtered case.

Revision ID: 0004_explore_api_sprint_a
Revises: 0003_v1_dream_heartbeat
Create Date: 2026-05-10
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004_explore_api_sprint_a"
down_revision: str | None = "0003_v1_dream_heartbeat"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Plain btree (no DESC) so Postgres can do forward OR backward scans;
    # ``mem_browse``'s ``ORDER BY ... DESC, id DESC`` is served by a
    # reverse-direction index scan. Including ``id`` provides the keyset
    # tiebreak without a sort.
    op.execute(
        "CREATE INDEX IF NOT EXISTS memories_env_status_updated_at_id_idx "
        "ON memories (env_id, status, updated_at, id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS memories_env_status_created_at_id_idx "
        "ON memories (env_id, status, created_at, id)"
    )
    # ``entities_env_kind_canonical_name_id_idx`` supports ``ent_browse``'s
    # default ``order_by="canonical_name"`` traversal (no prefix filter).
    op.execute(
        "CREATE INDEX IF NOT EXISTS entities_env_kind_canonical_name_id_idx "
        "ON entities (env_id, kind, canonical_name, id)"
    )
    # ``text_pattern_ops`` variant for ``LIKE 'prefix%'`` against the
    # normalized form. Both indexes are useful; the planner picks based on
    # whether ``name_prefix`` is set.
    op.execute(
        "CREATE INDEX IF NOT EXISTS entities_env_kind_norm_name_pattern_idx "
        "ON entities (env_id, kind, normalized_name text_pattern_ops, id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS entity_aliases_normalized_alias_pattern_idx "
        "ON entity_aliases (normalized_alias text_pattern_ops, entity_id)"
    )
    # ``rel_browse`` orders by ``(created_at, id)`` regardless of type
    # filter. Two indexes cover the common cases:
    #   - (env_id, type, created_at, id): when types[] is set
    #   - (env_id, created_at, id):       when types[] is not set
    op.execute(
        "CREATE INDEX IF NOT EXISTS relations_env_type_created_at_id_idx "
        "ON relations (env_id, type, created_at, id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS relations_env_created_at_id_idx "
        "ON relations (env_id, created_at, id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS relations_env_created_at_id_idx")
    op.execute("DROP INDEX IF EXISTS relations_env_type_created_at_id_idx")
    op.execute("DROP INDEX IF EXISTS entity_aliases_normalized_alias_pattern_idx")
    op.execute("DROP INDEX IF EXISTS entities_env_kind_norm_name_pattern_idx")
    op.execute("DROP INDEX IF EXISTS entities_env_kind_canonical_name_id_idx")
    op.execute("DROP INDEX IF EXISTS memories_env_status_created_at_id_idx")
    op.execute("DROP INDEX IF EXISTS memories_env_status_updated_at_id_idx")
