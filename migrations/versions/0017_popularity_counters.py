"""Add popularity / graph-citation counters to ``memories``.

Revision ID: 0017_popularity_counters
Revises: 0016_cascade_root

This migration adds four per-citation-kind integer counters plus one stored
``GENERATED ALWAYS AS`` sum, and wires three trigger functions that keep
those counters in sync with structural graph edges.

Counter columns
---------------
* ``reference_count_rel_link``  — incoming ``relations`` whose ``src`` graph
  node is *not* a task node and whose ``type`` is not the reserved Phase 4
  predicate ``related_to_popular``.
* ``reference_count_lineage``   — incoming ``memory_lineage`` rows whose
  ``relation`` is in the *load-bearing* whitelist (excludes ``supersedes``;
  ``split_from`` / ``derived_from`` are listed even though no rows carry
  those values yet in Phase 1 — Phase 3 introduces them, and forward-listing
  them here avoids a churn migration later).
* ``reference_count_task``      — incoming ``relations`` whose ``src`` graph
  node is a task node.
* ``reference_count_playbook``  — count of ``{{memory:<uuid>}}`` macro
  embeds in active playbook ``steps[]``. **Not** trigger-maintained — the
  dream ``recount`` pass (separate file) is the canonical writer for this
  column; trigger-side it stays at zero until the first recount run.

The stored sum ``reference_count`` is ``GENERATED ALWAYS AS STORED`` so
``mem_top by=reference_count`` can use a btree index with a stable
tie-breaker (``ORDER BY reference_count DESC, created_at DESC, id DESC``).

Trigger functions
-----------------
1. ``memories_bump_on_relation_change`` — ``AFTER INSERT OR DELETE ON
   relations``. Resolves ``dst_node_id → graph_nodes.memory_id``, branches
   on ``src.node_type`` (task vs other), skips ``type = 'related_to_popular'``
   so Phase 4's auto-wire neighbors do not feed back into popularity.
2. ``memories_bump_on_lineage_change`` — ``AFTER INSERT OR DELETE ON
   memory_lineage``. Filters to the load-bearing whitelist; bumps the
   ``parent_memory_id``'s lineage counter.
3. ``memories_status_flip_decrement`` — ``AFTER UPDATE OF status ON
   memories WHEN OLD.status IS DISTINCT FROM NEW.status``. On
   ``active → retired|superseded`` walks outgoing edges from the flipping
   memory and decrements targets; on ``retired|superseded → active``
   re-increments. Symmetric.

Backfill
--------
A bounded fast-path runs inside ``upgrade()``: when
``relations + memory_lineage`` rows together are below the envelope
(``100_000``) the migration performs a single-pass aggregate and writes
counter values. Above that threshold the migration leaves the columns at
zero and emits a NOTICE — the dream ``recount`` pass is the canonical
source of truth and will fix drift on its next run regardless of which
path the migration took.

Out of scope
------------
* Playbook macro scan (text-search of ``memories.steps``) — recount only.
* Ancestry exclusion across supersede chains — recount only; triggers
  may transiently over-count and recount converges.
* Authority weighting (``Σ source.salience``) — deferred (was Phase 1e;
  blocker B3 in the v2 plan).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0017_popularity_counters"
down_revision: str | None = "0016_cascade_root"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_LINEAGE_WHITELIST = (
    "summarized_from",
    "promoted_from",
    "derives_from",
    "split_from",
    "derived_from",
)


def _whitelist_sql_array() -> str:
    """SQL literal: ``ARRAY['summarized_from', 'promoted_from', ...]::text[]``."""
    quoted = ", ".join(f"'{v}'" for v in _LINEAGE_WHITELIST)
    return f"ARRAY[{quoted}]::text[]"


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------
    op.execute(
        """
        ALTER TABLE memories
            ADD COLUMN reference_count_rel_link INTEGER NOT NULL DEFAULT 0,
            ADD COLUMN reference_count_lineage  INTEGER NOT NULL DEFAULT 0,
            ADD COLUMN reference_count_task     INTEGER NOT NULL DEFAULT 0,
            ADD COLUMN reference_count_playbook INTEGER NOT NULL DEFAULT 0
        """
    )

    op.execute(
        """
        ALTER TABLE memories
            ADD COLUMN reference_count INTEGER
            GENERATED ALWAYS AS (
                reference_count_rel_link
                + reference_count_lineage
                + reference_count_task
                + reference_count_playbook
            ) STORED
        """
    )

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------
    # mem_top by=reference_count: env + status filter + ordered desc with
    # stable tie-breaker.
    op.execute(
        """
        CREATE INDEX memories_reference_count_idx
            ON memories (env_id, status, reference_count DESC, created_at DESC, id DESC)
        """
    )

    # Velocity CTE: count relations to a dst within a window.
    op.execute(
        """
        CREATE INDEX relations_velocity_idx
            ON relations (env_id, created_at DESC, dst_node_id)
        """
    )

    # Velocity CTE: count lineage parents within a window.
    op.execute(
        """
        CREATE INDEX memory_lineage_velocity_idx
            ON memory_lineage (created_at DESC, parent_memory_id)
        """
    )

    # ------------------------------------------------------------------
    # Trigger 1: relations change → rel_link / task counter
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION memories_bump_on_relation_change()
            RETURNS TRIGGER
            LANGUAGE plpgsql
        AS $fn$
        DECLARE
            v_dst_memory_id UUID;
            v_src_type      TEXT;
            v_rel_type      TEXT;
        BEGIN
            IF TG_OP = 'INSERT' THEN
                v_rel_type := NEW.type;
                -- Phase 4 auto-wire predicate must not feed back into popularity.
                IF v_rel_type = 'related_to_popular' THEN
                    RETURN NEW;
                END IF;

                SELECT gn.memory_id
                INTO   v_dst_memory_id
                FROM   graph_nodes gn
                WHERE  gn.id = NEW.dst_node_id;

                IF v_dst_memory_id IS NULL THEN
                    RETURN NEW;
                END IF;

                SELECT gn.node_type
                INTO   v_src_type
                FROM   graph_nodes gn
                WHERE  gn.id = NEW.src_node_id;

                IF v_src_type = 'task' THEN
                    UPDATE memories
                       SET reference_count_task = reference_count_task + 1
                     WHERE id = v_dst_memory_id;
                ELSE
                    UPDATE memories
                       SET reference_count_rel_link = reference_count_rel_link + 1
                     WHERE id = v_dst_memory_id;
                END IF;

                RETURN NEW;

            ELSIF TG_OP = 'DELETE' THEN
                v_rel_type := OLD.type;
                IF v_rel_type = 'related_to_popular' THEN
                    RETURN OLD;
                END IF;

                SELECT gn.memory_id
                INTO   v_dst_memory_id
                FROM   graph_nodes gn
                WHERE  gn.id = OLD.dst_node_id;

                IF v_dst_memory_id IS NULL THEN
                    RETURN OLD;
                END IF;

                SELECT gn.node_type
                INTO   v_src_type
                FROM   graph_nodes gn
                WHERE  gn.id = OLD.src_node_id;

                IF v_src_type = 'task' THEN
                    UPDATE memories
                       SET reference_count_task = GREATEST(reference_count_task - 1, 0)
                     WHERE id = v_dst_memory_id;
                ELSE
                    UPDATE memories
                       SET reference_count_rel_link = GREATEST(reference_count_rel_link - 1, 0)
                     WHERE id = v_dst_memory_id;
                END IF;

                RETURN OLD;
            END IF;

            RETURN NULL;
        END;
        $fn$
        """
    )

    op.execute(
        """
        CREATE TRIGGER memories_bump_on_relation_change_t
            AFTER INSERT OR DELETE ON relations
            FOR EACH ROW
            EXECUTE FUNCTION memories_bump_on_relation_change()
        """
    )

    # ------------------------------------------------------------------
    # Trigger 2: memory_lineage change → lineage counter
    # ------------------------------------------------------------------
    whitelist = _whitelist_sql_array()
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION memories_bump_on_lineage_change()
            RETURNS TRIGGER
            LANGUAGE plpgsql
        AS $fn$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.relation = ANY({whitelist}) THEN
                    UPDATE memories
                       SET reference_count_lineage = reference_count_lineage + 1
                     WHERE id = NEW.parent_memory_id;
                END IF;
                RETURN NEW;

            ELSIF TG_OP = 'DELETE' THEN
                IF OLD.relation = ANY({whitelist}) THEN
                    UPDATE memories
                       SET reference_count_lineage = GREATEST(reference_count_lineage - 1, 0)
                     WHERE id = OLD.parent_memory_id;
                END IF;
                RETURN OLD;
            END IF;

            RETURN NULL;
        END;
        $fn$
        """
    )

    op.execute(
        """
        CREATE TRIGGER memories_bump_on_lineage_change_t
            AFTER INSERT OR DELETE ON memory_lineage
            FOR EACH ROW
            EXECUTE FUNCTION memories_bump_on_lineage_change()
        """
    )

    # ------------------------------------------------------------------
    # Trigger 3: status flip → propagate to citation targets
    # ------------------------------------------------------------------
    #
    # When a memory M flips ``active → retired|superseded`` its outgoing
    # citations (rel_link / task / lineage where M is the child) should
    # stop counting. We don't delete edges — we decrement targets and
    # symmetrically re-increment on ``retired|superseded → active``.
    #
    # ``AFTER UPDATE OF status`` ensures we only fire on status column
    # changes — counter-only UPDATEs do not re-enter the trigger.
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION memories_status_flip_decrement()
            RETURNS TRIGGER
            LANGUAGE plpgsql
        AS $fn$
        DECLARE
            v_going_inactive BOOLEAN;
            v_going_active   BOOLEAN;
            v_delta          INTEGER;
        BEGIN
            v_going_inactive := (OLD.status = 'active'
                                 AND NEW.status IN ('retired', 'superseded'));
            v_going_active   := (NEW.status = 'active'
                                 AND OLD.status IN ('retired', 'superseded'));

            IF NOT (v_going_inactive OR v_going_active) THEN
                RETURN NEW;
            END IF;

            IF v_going_inactive THEN
                v_delta := -1;
            ELSE
                v_delta := 1;
            END IF;

            -- Outgoing rel_link / task edges where this memory is the src.
            -- Match the regular trigger's rules:
            --   * skip type = 'related_to_popular'
            --   * resolve src node type to choose rel_link vs task counter
            --   * skip rows whose dst is not a memory
            --
            -- IMPORTANT: aggregate counts per target before updating.
            -- ``UPDATE ... FROM`` would otherwise update each target memory
            -- exactly once even if multiple edges fan in, silently dropping
            -- N-1 of N expected adjustments.
            UPDATE memories m
               SET reference_count_task = GREATEST(m.reference_count_task + (v_delta * agg.n), 0)
              FROM (
                SELECT gn_dst.memory_id AS dst_memory_id, count(*) AS n
                  FROM relations r
                  JOIN graph_nodes gn_src ON gn_src.id = r.src_node_id
                  JOIN graph_nodes gn_dst ON gn_dst.id = r.dst_node_id
                 WHERE gn_src.memory_id = NEW.id
                   AND gn_src.node_type = 'task'      -- defensive only
                   AND gn_dst.memory_id IS NOT NULL
                   AND r.type <> 'related_to_popular'
                 GROUP BY gn_dst.memory_id
              ) AS agg
             WHERE m.id = agg.dst_memory_id;
            -- NB: src=task path above never fires today — when a memory
            -- flips status its src graph node is node_type='memory'.
            -- Path kept defensive for future entity-as-src semantics.

            UPDATE memories m
               SET reference_count_rel_link = GREATEST(m.reference_count_rel_link + (v_delta * agg.n), 0)
              FROM (
                SELECT gn_dst.memory_id AS dst_memory_id, count(*) AS n
                  FROM relations r
                  JOIN graph_nodes gn_src ON gn_src.id = r.src_node_id
                  JOIN graph_nodes gn_dst ON gn_dst.id = r.dst_node_id
                 WHERE gn_src.memory_id = NEW.id
                   AND gn_src.node_type <> 'task'
                   AND gn_dst.memory_id IS NOT NULL
                   AND r.type <> 'related_to_popular'
                 GROUP BY gn_dst.memory_id
              ) AS agg
             WHERE m.id = agg.dst_memory_id;

            -- Outgoing lineage rows where this memory is the child.
            UPDATE memories m
               SET reference_count_lineage = GREATEST(m.reference_count_lineage + (v_delta * agg.n), 0)
              FROM (
                SELECT ml.parent_memory_id AS parent_id, count(*) AS n
                  FROM memory_lineage ml
                 WHERE ml.child_memory_id = NEW.id
                   AND ml.relation = ANY({whitelist})
                 GROUP BY ml.parent_memory_id
              ) AS agg
             WHERE m.id = agg.parent_id;

            RETURN NEW;
        END;
        $fn$
        """
    )

    op.execute(
        """
        CREATE TRIGGER memories_status_flip_decrement_t
            AFTER UPDATE OF status ON memories
            FOR EACH ROW
            WHEN (OLD.status IS DISTINCT FROM NEW.status)
            EXECUTE FUNCTION memories_status_flip_decrement()
        """
    )

    # ------------------------------------------------------------------
    # Bounded backfill (fast-path).
    # ------------------------------------------------------------------
    # Recount-pass is canonical; this is just a courtesy for small envs.
    op.execute(
        f"""
        DO $$
        DECLARE
            v_edge_count BIGINT;
        BEGIN
            SELECT
                (SELECT count(*) FROM relations)
                + (SELECT count(*) FROM memory_lineage)
            INTO v_edge_count;

            IF v_edge_count > 100000 THEN
                RAISE NOTICE
                    'memory-mcp 0017: skipping fast-path backfill '
                    '(edges=% > 100000); dream recount pass will populate counters.',
                    v_edge_count;
                RETURN;
            END IF;

            -- rel_link / task from relations.
            --
            -- Mirrors what the live triggers would have produced if they
            -- had fired on every existing edge: edges whose src is a
            -- retired/superseded memory are *excluded* (matching the
            -- status-flip trigger's after-the-fact decrement), while
            -- edges from non-memory sources (task / entity graph_nodes)
            -- are always counted.
            WITH per_dst AS (
                SELECT
                    gn_dst.memory_id   AS dst_memory_id,
                    sum((gn_src.node_type =  'task')::int) AS task_n,
                    sum((gn_src.node_type <> 'task')::int) AS rl_n
                FROM relations r
                JOIN graph_nodes gn_src ON gn_src.id = r.src_node_id
                JOIN graph_nodes gn_dst ON gn_dst.id = r.dst_node_id
                LEFT JOIN memories sm    ON sm.id = gn_src.memory_id
                WHERE gn_dst.memory_id IS NOT NULL
                  AND r.type <> 'related_to_popular'
                  AND (
                      gn_src.node_type <> 'memory'
                      OR sm.status = 'active'
                  )
                GROUP BY gn_dst.memory_id
            )
            UPDATE memories m
               SET reference_count_rel_link = COALESCE(p.rl_n, 0)::int,
                   reference_count_task     = COALESCE(p.task_n, 0)::int
              FROM per_dst p
             WHERE m.id = p.dst_memory_id;

            -- lineage from memory_lineage (whitelist; child must be active)
            WITH per_parent AS (
                SELECT
                    ml.parent_memory_id AS parent_id,
                    count(*)            AS n
                FROM memory_lineage ml
                JOIN memories child ON child.id = ml.child_memory_id
                WHERE ml.relation = ANY({whitelist})
                  AND child.status = 'active'
                GROUP BY ml.parent_memory_id
            )
            UPDATE memories m
               SET reference_count_lineage = COALESCE(p.n, 0)::int
              FROM per_parent p
             WHERE m.id = p.parent_id;
        END
        $$
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS memories_status_flip_decrement_t ON memories")
    op.execute("DROP TRIGGER IF EXISTS memories_bump_on_lineage_change_t ON memory_lineage")
    op.execute("DROP TRIGGER IF EXISTS memories_bump_on_relation_change_t ON relations")
    op.execute("DROP FUNCTION IF EXISTS memories_status_flip_decrement()")
    op.execute("DROP FUNCTION IF EXISTS memories_bump_on_lineage_change()")
    op.execute("DROP FUNCTION IF EXISTS memories_bump_on_relation_change()")

    op.execute("DROP INDEX IF EXISTS memory_lineage_velocity_idx")
    op.execute("DROP INDEX IF EXISTS relations_velocity_idx")
    op.execute("DROP INDEX IF EXISTS memories_reference_count_idx")

    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS reference_count")
    op.execute(
        """
        ALTER TABLE memories
            DROP COLUMN IF EXISTS reference_count_playbook,
            DROP COLUMN IF EXISTS reference_count_task,
            DROP COLUMN IF EXISTS reference_count_lineage,
            DROP COLUMN IF EXISTS reference_count_rel_link
        """
    )
