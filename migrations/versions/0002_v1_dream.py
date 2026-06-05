"""Phase 2.2: dream_runs + dream_proposals

Adds the two tables that drive the dream-worker (decay, dedupe, promote)
and the agent-facing review surface.

Design:

* ``dream_runs`` records every pass executed by the dream-worker (or
  triggered manually via ``dream_run``). Tracks mode, env, lifecycle
  status, and a JSON ``summary`` with counts of items inspected /
  transitioned / proposals emitted / errors per run.
* ``dream_proposals`` is the Postgres-canonical store of merge / promotion
  candidates produced by passes. Status walks
  ``open → accepted | rejected | amended | deferred | expired``.
  ``payload`` carries everything the reviewer needs to decide
  (candidate_ids, cosine_scores, suggested merged content, etc.) — and is
  fully describable per-kind in design.md / journal.md.
* **No outbox events**: proposals are local Postgres state, not
  projected to Qdrant or Neo4j. Acceptance is the moment canonical
  state mutates, and that goes through the existing
  ``entity_merge`` / ``memory_supersede`` / ``memory_promote`` paths
  which already emit outbox events.

ENUMs use ``text + CHECK`` for forward-compatibility with new modes /
proposal kinds — same convention as ``0001_v1_initial``.

Revision ID: 0002_v1_dream
Revises: 0001_v1_initial
Create Date: 2026-05-09
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_v1_dream"
down_revision: str | None = "0001_v1_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---- dream_runs --------------------------------------------------------
    # One row per pass execution. ``status`` walks running → done|failed|cancelled.
    # ``summary`` is opaque JSON shaped per-mode by the pass implementation.
    op.execute("""
        CREATE TABLE dream_runs (
            id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            env_id        uuid NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
            mode          text NOT NULL CHECK (mode IN ('decay','dedupe','promote','retention')),
            status        text NOT NULL DEFAULT 'running'
                          CHECK (status IN ('running','done','failed','cancelled')),
            started_at    timestamptz NOT NULL DEFAULT now(),
            ended_at      timestamptz,
            triggered_by  text NOT NULL DEFAULT 'scheduler'
                          CHECK (triggered_by IN ('scheduler','tool','test')),
            summarizer_kind text CHECK (summarizer_kind IN ('llm','template')),
            summary       jsonb NOT NULL DEFAULT '{}'::jsonb,
            last_error    text
        )
    """)
    op.execute("CREATE INDEX dream_runs_env_started_idx ON dream_runs(env_id, started_at DESC)")
    op.execute("CREATE INDEX dream_runs_mode_started_idx ON dream_runs(mode, started_at DESC)")
    op.execute("CREATE INDEX dream_runs_running_idx ON dream_runs(env_id, mode) WHERE status = 'running'")

    # ---- dream_proposals ---------------------------------------------------
    # ``kind`` distinguishes payload shape (see ``payload`` JSON contract in
    # design.md). ``status`` walks open → terminal. ``reviewed_by_agent_id``
    # is set when an agent calls dream_review.
    op.execute("""
        CREATE TABLE dream_proposals (
            id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            env_id                uuid NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
            kind                  text NOT NULL
                                  CHECK (kind IN ('merge_candidate','promotion_candidate','decay_candidate')),
            status                text NOT NULL DEFAULT 'open'
                                  CHECK (status IN ('open','accepted','rejected','amended','deferred','expired')),
            payload               jsonb NOT NULL DEFAULT '{}'::jsonb,
            summarizer_kind       text CHECK (summarizer_kind IN ('llm','template')),
            llm_failed            boolean NOT NULL DEFAULT false,
            dedupe_key            text,
            dream_run_id          uuid REFERENCES dream_runs(id) ON DELETE SET NULL,
            created_at            timestamptz NOT NULL DEFAULT now(),
            updated_at            timestamptz NOT NULL DEFAULT now(),
            reviewed_at           timestamptz,
            reviewed_by_agent_id  uuid REFERENCES agents(id) ON DELETE SET NULL,
            review_action         text CHECK (review_action IN ('accept','reject','amend','defer')),
            review_notes          text,
            CONSTRAINT dream_proposals_review_fields_chk CHECK (
                (status = 'open') = (reviewed_at IS NULL)
                AND
                (status = 'open') = (review_action IS NULL)
            )
        )
    """)

    # Most common reads are: list-open by env, or by (env, kind, status).
    op.execute("CREATE INDEX dream_proposals_env_status_idx ON dream_proposals(env_id, status)")
    op.execute("CREATE INDEX dream_proposals_env_kind_status_idx ON dream_proposals(env_id, kind, status)")
    op.execute("CREATE INDEX dream_proposals_run_idx ON dream_proposals(dream_run_id) WHERE dream_run_id IS NOT NULL")
    op.execute("CREATE INDEX dream_proposals_created_idx ON dream_proposals(env_id, created_at DESC)")

    # Idempotency: a pass over an unchanged input must not produce duplicate
    # open proposals for the same logical cluster. ``dedupe_key`` is set by
    # the pass to a stable hash of the cluster contents (e.g. sorted member
    # ids); a partial unique index over open proposals enforces this.
    op.execute(
        "CREATE UNIQUE INDEX dream_proposals_open_dedupe_key_uniq "
        "ON dream_proposals(env_id, kind, dedupe_key) "
        "WHERE status = 'open' AND dedupe_key IS NOT NULL"
    )

    # Keep ``updated_at`` fresh on every row update without forcing callers
    # to thread it through SET-clauses. dream_proposals has no ``version``
    # column so the shared monotonic-version trigger doesn't apply.
    op.execute("""
        CREATE OR REPLACE FUNCTION dream_proposals_touch_updated_at() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END
        $$
    """)
    op.execute(
        "CREATE TRIGGER dream_proposals_updated_at_trg "
        "BEFORE UPDATE ON dream_proposals FOR EACH ROW "
        "EXECUTE FUNCTION dream_proposals_touch_updated_at()"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS dream_proposals CASCADE")
    op.execute("DROP TABLE IF EXISTS dream_runs      CASCADE")
    op.execute("DROP FUNCTION IF EXISTS dream_proposals_touch_updated_at()")
