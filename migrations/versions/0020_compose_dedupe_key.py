"""Add ``compose_dedupe_key`` column + partial unique index (Phase 2 B3b).

Revision ID: 0020_compose_dedupe_key
Revises: 0019_salience_formula_version

Background
----------
Phase 2 ``mem_compose`` (N→1 caller-driven aggregation, both ``promote`` and
``merge`` modes) needs durable idempotency: a retry of an identical compose
call must return the same memory rather than create a second one. The
strategy is to compute a deterministic 32-hex dedupe key over the request's
content and constraints (``schema_version``, ``operation``, ``env_id``,
``mode``, sorted ``source_ids``, and the target's ``kind``/``title``/``body``
/sorted ``tags``/``metadata``/``decision_meta``/``confidence``/``salience``
/``pinned``) and store it on the resulting memory row.

A partial unique index ``(env_id, compose_dedupe_key) WHERE
compose_dedupe_key IS NOT NULL`` makes the "did we already create this
memory?" check a single index lookup and lets concurrent identical writes
race onto the same row (the loser catches ``UniqueViolation`` and re-fetches
the winning row).

Mechanism
---------
* ``memories.compose_dedupe_key TEXT NULL`` — defaults to ``NULL`` for every
  existing row. Only ``mem_compose``-created memories carry a value; the
  field is invisible to all other write paths.
* Partial unique index ``ix_memories_compose_dedupe`` on
  ``(env_id, compose_dedupe_key)`` with the predicate
  ``WHERE compose_dedupe_key IS NOT NULL``. The partial predicate keeps the
  index small (NULL-heavy column) and avoids burdening writes that don't
  use compose.

Why a partial index (not a column-level UNIQUE)
-----------------------------------------------
A column-level ``UNIQUE`` constraint with NULLs is fine in PostgreSQL (NULL
is treated as distinct), but a partial index is more honest about intent:
"only ``mem_compose``-produced rows participate in dedupe". A future
``mem_decompose`` track (Phase 3) might add its own dedupe column; both can
live as partial indexes side-by-side without confusion.

Backfill
--------
None — all existing rows stay ``NULL``. Pre-existing memories were not
created via ``mem_compose`` and cannot retroactively claim a dedupe key.

Out of scope
------------
* Indexing the dedupe key globally (without ``env_id``) — compose is
  env-local; cross-env compose is explicitly out of scope for v0.15.0.
* Backfilling dedupe keys for memories that look like compose products
  (e.g. those with multiple ``MemoryLineage`` parents) — they predate the
  contract; there is no canonical key to assign.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0020_compose_dedupe_key"
down_revision: str | None = "0019_salience_formula_version"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE memories
            ADD COLUMN compose_dedupe_key TEXT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX ix_memories_compose_dedupe
            ON memories (env_id, compose_dedupe_key)
            WHERE compose_dedupe_key IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memories_compose_dedupe")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS compose_dedupe_key")
