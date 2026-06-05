"""Phase 2.2 follow-up: loosen ``projection_state.sink`` constraint for dream-worker heartbeats.

The dream-worker process writes per-mode liveness rows to
``projection_state`` using sink names of the form ``dream_worker:<mode>``
(e.g. ``dream_worker:decay``, ``dream_worker:dedupe``,
``dream_worker:promote``). The original ``0001_v1_initial`` migration
constrained ``sink`` to ``('qdrant', 'neo4j', 'pgvector')`` for the
projection-worker sinks; the dream-worker heartbeat would error on a
real DB.

This migration drops the ``sink`` CHECK constraint and replaces it with
a wider one that accepts the projection-worker sinks plus any
``dream_worker:%`` value. The same change is applied to
``outbox_delivery.sink`` for consistency, even though dream-worker
events do not flow through the outbox today (forward-compat — keeps the
two sink columns symmetric).

This migration is **safe to deploy on any environment** — it only adds
to the allowed value set; existing rows remain valid.

Revision ID: 0003_v1_dream_heartbeat
Revises: 0002_v1_dream
Create Date: 2026-05-10
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003_v1_dream_heartbeat"
down_revision: str | None = "0002_v1_dream"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_OLD_SINK_CHECK = "sink IN ('qdrant','neo4j','pgvector')"
_NEW_SINK_CHECK = "sink IN ('qdrant','neo4j','pgvector') OR sink LIKE 'dream_worker:%'"


def upgrade() -> None:
    # ``projection_state.sink``
    op.execute("""
        ALTER TABLE projection_state
            DROP CONSTRAINT IF EXISTS projection_state_sink_check
    """)
    op.execute(f"""
        ALTER TABLE projection_state
            ADD CONSTRAINT projection_state_sink_check CHECK ({_NEW_SINK_CHECK})
    """)

    # ``outbox_delivery.sink`` — keep symmetric with projection_state.
    op.execute("""
        ALTER TABLE outbox_delivery
            DROP CONSTRAINT IF EXISTS outbox_delivery_sink_check
    """)
    op.execute(f"""
        ALTER TABLE outbox_delivery
            ADD CONSTRAINT outbox_delivery_sink_check CHECK ({_NEW_SINK_CHECK})
    """)


def downgrade() -> None:
    # Revert to the original projection-only constraint. Any rows with
    # ``dream_worker:%`` sinks will fail this validation; the operator
    # must clear them first.
    op.execute("""
        ALTER TABLE projection_state
            DROP CONSTRAINT IF EXISTS projection_state_sink_check
    """)
    op.execute(f"""
        ALTER TABLE projection_state
            ADD CONSTRAINT projection_state_sink_check CHECK ({_OLD_SINK_CHECK})
    """)
    op.execute("""
        ALTER TABLE outbox_delivery
            DROP CONSTRAINT IF EXISTS outbox_delivery_sink_check
    """)
    op.execute(f"""
        ALTER TABLE outbox_delivery
            ADD CONSTRAINT outbox_delivery_sink_check CHECK ({_OLD_SINK_CHECK})
    """)
