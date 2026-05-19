"""Add ``salience_formula_version`` stamp column to ``memories`` (Phase 1e-d).

Revision ID: 0019_salience_formula_version
Revises: 0018_authority_weighting

This migration ships the formula-version mechanism that closes the D2
forward issue from slice 1e-c: when ``compute_salience`` math changes
(e.g. the authority term wired in 1e-d), existing rows' stored salience
values become stale. Without an explicit "what formula version
produced this salience?" stamp, the recount pass cannot tell which
rows need re-stamping vs which are already current.

Mechanism
---------
* ``memories.salience_formula_version INTEGER NOT NULL DEFAULT 0`` —
  ``0`` is the "unstamped" baseline (pre-1e-d, or a row created before
  any recount touched it under the current formula).
* ``Settings.dream_salience_formula_version`` (default ``1`` once 1e-d
  ships) declares the current formula version.
* The recount pass compares ``Memory.salience_formula_version`` against
  the settings value and adds any "behind" row to its salience-recompute
  set (chunked at ``dream_recount_salience_recompute_cap`` per cycle to
  bound the first-pass deploy spike).
* On every successful ``_recompute_salience_for`` write, the recount
  stamps the row with the current formula version atomically with the
  new salience value — both flow through ``MemoryUpdatePatch`` so the
  audit trail + outbox + Qdrant payload sync stay consistent.

Future-formula-change invariant
-------------------------------
**ANY change to ``compute_salience`` math MUST bump
``Settings.dream_salience_formula_version``.** Otherwise existing rows
keep their pre-change salience indefinitely (until they happen to drift
on counters / authority and pick up a recompute incidentally). The
``salience.py`` module docstring carries the same callout for the
developer touching the formula.

Backfill
--------
None at migration time — column ships at ``0`` for every existing row.
The first recount cycle after the deploy that bumps
``dream_salience_formula_version`` will start stamping rows in batches
of up to ``dream_recount_salience_recompute_cap`` (default 500). Large
envs require multiple cycles to fully restamp; that's intentional —
bounded per-cycle work over a longer wall-clock window beats one giant
spike.

Operators who want faster backfill can raise the cap temporarily, or
trigger additional ``dream_run_now(mode='recount')`` invocations.

Out of scope
------------
* Index on ``salience_formula_version`` — recount's mismatched-lookup
  is bounded by the per-cycle cap and runs at most daily, so a
  sequential scan over a filtered active subset is acceptable. Revisit
  if telemetry shows the scan dominating recount pass time.
* Hash-derived formula version (instead of a manual integer) — deferred
  to v0.14.2+ if proven necessary.
* API surfacing on ``MemoryResponse`` — defer to 1e-e/1e-f.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0019_salience_formula_version"
down_revision: str | None = "0018_authority_weighting"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE memories
            ADD COLUMN salience_formula_version INTEGER NOT NULL DEFAULT 0
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS salience_formula_version")
