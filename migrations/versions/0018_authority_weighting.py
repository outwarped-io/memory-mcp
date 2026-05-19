"""Add authority-weighted citation columns to ``memories`` (Phase 1e).

Revision ID: 0018_authority_weighting
Revises: 0017_popularity_counters

This migration extends the popularity machinery introduced in 0017 with a
parallel **authority** axis: instead of counting incoming citations as
``Σ 1``, the recount dream pass (extended in a separate slice) will
populate ``Σ source.salience`` per kind. The columns are added at zero;
no trigger maintenance — the recount pass is the **only writer** because
authority is fractional and source-salience drifts continuously, making
trigger-time ``+= delta`` infeasible.

Columns
-------
* ``ref_authority_rel_link``  — Σ salience(source) for incoming rel_link
  edges (src is NOT a task node).
* ``ref_authority_lineage``   — Σ salience(source) for incoming whitelisted
  ``memory_lineage`` rows (same whitelist as 0017).
* ``ref_authority_task``      — Σ salience(source) for incoming rel_link
  edges where src IS a task node.
* ``ref_authority_playbook``  — Σ salience(source) for active playbooks
  embedding this memory via ``{{memory:<uuid>}}`` macro.

All four are ``NUMERIC(18,6)`` — the per-citation-occurrence worst case
(not per-memory) requires generous headroom: a pathological store with
millions of incoming citations summed across full-salience citers stays
well below ``999_999_999_999.999999``.

The stored sum ``reference_authority`` is ``NUMERIC(19,6)`` (≥ 4 ×
``NUMERIC(18,6)``) and ``GENERATED ALWAYS AS STORED`` so
``mem_top by=reference_authority`` can use a btree index with a stable
tie-breaker (``ORDER BY reference_authority DESC, created_at DESC, id DESC``).

Staleness marker
----------------
``authority_last_recount_at TIMESTAMPTZ NULL`` is stamped by the recount
pass each cycle it touches a memory. Surfaces on ``MemoryResponse`` so
callers can distinguish "recently recomputed and still zero" from "never
recomputed" (the knob has never been on for this env).

Backfill
--------
None. Columns ship at 0 (and ``authority_last_recount_at`` ships at NULL).
First recount cycle after ``DREAM_POPULARITY_AUTHORITY_WEIGHTED=true``
populates them. Operators can force an immediate pass via
``dream_run_now(mode='recount')``.

Out of scope
------------
* Triggers — authority is recount-only because ``Σ source.salience`` is
  fractional and the source's salience drifts under read-axis events
  (``access_count`` ↑, decay ↓). A trigger-time ``+= delta`` model cannot
  react to a citer's salience changing; the recount pass walks canonical
  edge state each cycle.
* Per-kind authority multipliers — v0.14.1 ships **uniform** authority
  weighting (``Σ salience`` un-multiplied). Per-kind weights deferred to
  v0.14.2+ if telemetry justifies (see Phase 1e plan §A8 / R-B5).
* Damping factor knob ``authority_damping`` (defaults to ``1.0`` = off in
  config; see Phase 1e plan §A3 / R-B3) — config-only, no schema bit.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0018_authority_weighting"
down_revision: str | None = "0017_popularity_counters"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Per-kind authority columns (parallel to 0017's reference_count_*)
    # ------------------------------------------------------------------
    op.execute(
        """
        ALTER TABLE memories
            ADD COLUMN ref_authority_rel_link NUMERIC(18, 6) NOT NULL DEFAULT 0,
            ADD COLUMN ref_authority_lineage  NUMERIC(18, 6) NOT NULL DEFAULT 0,
            ADD COLUMN ref_authority_task     NUMERIC(18, 6) NOT NULL DEFAULT 0,
            ADD COLUMN ref_authority_playbook NUMERIC(18, 6) NOT NULL DEFAULT 0
        """
    )

    # Computed total — NUMERIC(19,6) accommodates 4 maxed NUMERIC(18,6).
    op.execute(
        """
        ALTER TABLE memories
            ADD COLUMN reference_authority NUMERIC(19, 6)
            GENERATED ALWAYS AS (
                ref_authority_rel_link
                + ref_authority_lineage
                + ref_authority_task
                + ref_authority_playbook
            ) STORED
        """
    )

    # Staleness marker — stamped by recount pass each cycle a memory is
    # touched. NULL means "never recomputed" (knob has never been on).
    op.execute(
        """
        ALTER TABLE memories
            ADD COLUMN authority_last_recount_at TIMESTAMPTZ
        """
    )

    # ------------------------------------------------------------------
    # mem_top by=reference_authority ordering index
    # ------------------------------------------------------------------
    # Partial index — only active rows are sortable by this axis, and the
    # mem_top tool filters status='active' by default. Matches 0017's
    # `memories_reference_count_idx` shape but limits to active to keep
    # the index narrow.
    op.execute(
        """
        CREATE INDEX memories_reference_authority_idx
            ON memories (env_id, reference_authority DESC, created_at DESC, id DESC)
            WHERE status = 'active'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS memories_reference_authority_idx")

    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS authority_last_recount_at")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS reference_authority")
    op.execute(
        """
        ALTER TABLE memories
            DROP COLUMN IF EXISTS ref_authority_playbook,
            DROP COLUMN IF EXISTS ref_authority_task,
            DROP COLUMN IF EXISTS ref_authority_lineage,
            DROP COLUMN IF EXISTS ref_authority_rel_link
        """
    )
