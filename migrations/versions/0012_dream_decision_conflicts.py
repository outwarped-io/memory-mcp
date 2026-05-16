"""Add dream decision-conflict mode and proposal kind.

Revision ID: 0012_dream_decision_conflicts
Revises: 0011_decision_meta
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0012_dream_decision_conflicts"
down_revision: str | None = "0011_decision_meta"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE dream_runs DROP CONSTRAINT IF EXISTS dream_runs_mode_check")
    op.execute("""
        ALTER TABLE dream_runs
        ADD CONSTRAINT dream_runs_mode_check
        CHECK (mode IN ('decay','dedupe','promote','retention','decision_conflicts'))
    """)

    op.execute("ALTER TABLE dream_proposals DROP CONSTRAINT IF EXISTS dream_proposals_kind_check")
    op.execute("""
        ALTER TABLE dream_proposals
        ADD CONSTRAINT dream_proposals_kind_check
        CHECK (kind IN (
            'merge_candidate',
            'promotion_candidate',
            'decay_candidate',
            'decision_conflict_candidate'
        ))
    """)


def downgrade() -> None:
    op.execute("DELETE FROM dream_proposals WHERE kind = 'decision_conflict_candidate'")
    op.execute("DELETE FROM dream_runs WHERE mode = 'decision_conflicts'")

    op.execute("ALTER TABLE dream_proposals DROP CONSTRAINT IF EXISTS dream_proposals_kind_check")
    op.execute("""
        ALTER TABLE dream_proposals
        ADD CONSTRAINT dream_proposals_kind_check
        CHECK (kind IN ('merge_candidate','promotion_candidate','decay_candidate'))
    """)

    op.execute("ALTER TABLE dream_runs DROP CONSTRAINT IF EXISTS dream_runs_mode_check")
    op.execute("""
        ALTER TABLE dream_runs
        ADD CONSTRAINT dream_runs_mode_check
        CHECK (mode IN ('decay','dedupe','promote','retention'))
    """)
