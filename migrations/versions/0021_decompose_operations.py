"""Phase 3 substrate: widen lineage CHECK, add decompose_operations, retune triggers.

Revision ID: 0021_decompose_operations
Revises: 0020_compose_dedupe_key

This migration prepares the ``memories`` substrate for Phase 3 ``mem_decompose``
(``split`` and ``derive`` modes) **without** introducing the tool surface itself
(that lands in C3/C4). Three substrate changes ship together so the rest of the
phase has a clean foundation:

1. Widen the ``memory_lineage.relation`` CHECK constraint to admit two new
   Phase 3 relation values: ``split_from`` (M→0 split, source retired) and
   ``derived_from`` (M→N derive, source active).
2. Create the ``decompose_operations`` idempotency table. This is the
   substrate-level race arbiter for the new tool — its
   ``(env_id, dedupe_key)`` UNIQUE index lets concurrent identical decomposes
   collapse onto one operation row.
3. Re-issue the two popularity trigger functions (``memories_bump_on_lineage_
   change`` and ``memories_status_flip_decrement``) with the load-bearing
   lineage whitelist updated: ``split_from`` is **removed** (rows of that
   relation must not bump ``reference_count_lineage``), ``derived_from``
   stays so derive children contribute to the source's popularity.

Why CHECK widening, not enum ALTER
----------------------------------
``memory_lineage.relation`` is a ``text`` column with an inline CHECK
constraint (defined in migration 0001 line 369). There is **no** Postgres
ENUM type named ``lineage_relation`` — the C1.5 plan briefly assumed there
was one. Verified at design time:

  * ``\\d memory_lineage`` shows ``relation text``.
  * ``\\dT lineage_relation`` returns no rows.
  * ``db/models.py:607`` declares ``relation: Mapped[str] = mapped_column(Text, ...)``.

The 0001 inline CHECK was given an auto-generated name by Postgres
(typically ``memory_lineage_check`` plus a digit suffix). We locate it via a
``pg_constraint`` lookup, drop it, and add back a named constraint
(``memory_lineage_relation_check``) with the widened value list. The 0017
``dream_runs_mode_check`` widening is the template for this pattern.

Why ``split_from`` is removed from the popularity whitelist
-----------------------------------------------------------
Migration 0017 forward-listed ``split_from`` + ``derived_from`` in the
load-bearing whitelist so a future Phase 3 migration would not require a
trigger rewrite. The intent was to save churn. The C1.5 rubber-duck
consultation surfaced that ``split_from`` was the wrong call:

  * ``split_from`` connects N child memories to a **retired** parent — the
    parent's popularity does not matter once it's no longer active, and
    bumping a retired parent's counter pollutes any later un-retire path.
  * ``derived_from`` connects N child memories to an **active** parent
    (derive mode is non-destructive) — that bump is real signal.

We re-issue the trigger function bodies with the corrected whitelist. The
trigger names themselves are unchanged so the existing
``AFTER INSERT OR DELETE ON memory_lineage`` and
``AFTER UPDATE OF status ON memories`` bindings continue to fire.

Companion code changes (same commit)
------------------------------------
Three runtime whitelist literals must stay in sync with the trigger SQL.
They are edited in the same commit as this migration:

  * ``src/memory_mcp/dream/passes/recount.py`` ``_LINEAGE_WHITELIST``
    (frozenset) — recount-pass canonical sum.
  * ``src/memory_mcp/top.py`` ``_LINEAGE_VELOCITY_WHITELIST`` (tuple) —
    ``mem_top`` velocity CTE.
  * ``schemas/src/memory_mcp_schemas/enums.py`` ``LineageRelation`` (StrEnum)
    — gains ``split_from`` + ``derived_from`` members.

``derives_from`` continues to be forward-listed in all three whitelist sites
(originally added by 0017). Phase 3 has no mandate to touch it; removing
it would be unrelated cleanup.

Backfill
--------
None. No existing rows carry ``split_from`` or ``derived_from`` relations
(the prior CHECK would have rejected them at insert time). The widened
CHECK admits future rows; the corrected whitelist applies to those future
rows only. The recount pass's canonical sum is unchanged because the
old + new whitelists differ only on ``split_from`` and no such rows exist.

Out of scope
------------
* The ``DecomposeOperation`` SQLAlchemy ORM model — lands with C4 in the
  same change-set as ``decomposers.py``.
* The ``mem_decompose`` tool registration — lands in C3.
* Validation logic, transaction body, smoke + matrix tests — C5/C6/C7/C9.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0021_decompose_operations"
down_revision: str | None = "0020_compose_dedupe_key"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pre-0021 memory_lineage.relation CHECK values (verbatim from migration
# 0001 line 370).
_OLD_LINEAGE_CHECK_VALUES: tuple[str, ...] = (
    "promoted_from",
    "summarized_from",
    "copied_from",
    "moved_from",
    "supersedes",
)

# 0021 widens the CHECK to admit Phase 3 relation values.
_NEW_LINEAGE_CHECK_VALUES: tuple[str, ...] = (
    *_OLD_LINEAGE_CHECK_VALUES,
    "split_from",
    "derived_from",
)

# Pre-0021 popularity-trigger whitelist (verbatim from migration 0017
# line 77-82). ``split_from`` and ``derived_from`` were forward-listed
# speculatively; ``derives_from`` was forward-listed for an unrelated
# future operation.
_OLD_POPULARITY_WHITELIST: tuple[str, ...] = (
    "summarized_from",
    "promoted_from",
    "derives_from",
    "split_from",
    "derived_from",
)

# 0021 removes ``split_from`` (C1.5 redirect E.11). ``derives_from`` stays
# forward-listed (no Phase 3 mandate to touch it). ``derived_from`` stays
# because derive children legitimately bump their (active) source.
_NEW_POPULARITY_WHITELIST: tuple[str, ...] = (
    "summarized_from",
    "promoted_from",
    "derives_from",
    "derived_from",
)


# ---------------------------------------------------------------------------
# SQL emitters (parameterised on whitelist for 0017/0021 lock-step)
# ---------------------------------------------------------------------------


def _whitelist_array(values: tuple[str, ...]) -> str:
    """Render ``ARRAY['a','b',...]::text[]`` for a Postgres SQL literal."""

    quoted = ", ".join(f"'{v}'" for v in values)
    return f"ARRAY[{quoted}]::text[]"


def _check_in_values(values: tuple[str, ...]) -> str:
    """Render ``'a','b',...`` for a Postgres ``IN (...)`` predicate."""

    return ", ".join(f"'{v}'" for v in values)


def _lineage_bump_function_sql(whitelist: tuple[str, ...]) -> str:
    """Return ``CREATE OR REPLACE FUNCTION memories_bump_on_lineage_change``.

    Body is verbatim from 0017 (lines 264-291) with the whitelist literal
    parameterised so 0017 and 0021 cannot drift.
    """

    whitelist_sql = _whitelist_array(whitelist)
    return f"""
        CREATE OR REPLACE FUNCTION memories_bump_on_lineage_change()
            RETURNS TRIGGER
            LANGUAGE plpgsql
        AS $fn$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.relation = ANY({whitelist_sql}) THEN
                    UPDATE memories
                       SET reference_count_lineage = reference_count_lineage + 1
                     WHERE id = NEW.parent_memory_id;
                END IF;
                RETURN NEW;

            ELSIF TG_OP = 'DELETE' THEN
                IF OLD.relation = ANY({whitelist_sql}) THEN
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


def _status_flip_function_sql(whitelist: tuple[str, ...]) -> str:
    """Return ``CREATE OR REPLACE FUNCTION memories_status_flip_decrement``.

    Body is verbatim from 0017 (lines 313-397) with the whitelist literal
    parameterised. The ``relations`` legs (rel_link / task) are unchanged
    by 0021 — only the lineage leg's whitelist changes.
    """

    whitelist_sql = _whitelist_array(whitelist)
    return f"""
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
            -- (Body unchanged from 0017; reproduced here so CREATE OR REPLACE
            -- emits the full function in one statement.)
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
            -- 0021: whitelist now excludes 'split_from'.
            UPDATE memories m
               SET reference_count_lineage = GREATEST(m.reference_count_lineage + (v_delta * agg.n), 0)
              FROM (
                SELECT ml.parent_memory_id AS parent_id, count(*) AS n
                  FROM memory_lineage ml
                 WHERE ml.child_memory_id = NEW.id
                   AND ml.relation = ANY({whitelist_sql})
                 GROUP BY ml.parent_memory_id
              ) AS agg
             WHERE m.id = agg.parent_id;

            RETURN NEW;
        END;
        $fn$
    """


def _drop_existing_relation_check_sql() -> str:
    """Locate and drop the auto-named CHECK constraint on memory_lineage.relation.

    The 0001 migration defined the CHECK inline, so Postgres assigned it a
    name like ``memory_lineage_check`` or ``memory_lineage_relation_check``
    depending on backend version. Postgres also normalises ``IN (...)`` to
    ``= ANY (ARRAY[...])`` in ``pg_get_constraintdef``, so we match by
    column-name token rather than the ``IN`` keyword. The lookup excludes
    ``memory_lineage_not_self_chk`` because its definition references
    ``parent_memory_id`` / ``child_memory_id`` rather than ``relation``.
    """

    return """
        DO $$
        DECLARE
            v_name TEXT;
        BEGIN
            FOR v_name IN
                SELECT conname
                  FROM pg_constraint
                 WHERE conrelid = 'memory_lineage'::regclass
                   AND contype  = 'c'
                   AND pg_get_constraintdef(oid) ~ '\\mrelation\\M'
            LOOP
                EXECUTE format('ALTER TABLE memory_lineage DROP CONSTRAINT %I', v_name);
            END LOOP;
        END $$;
    """


# ---------------------------------------------------------------------------
# upgrade / downgrade
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # 1. Widen memory_lineage CHECK constraint.
    op.execute(_drop_existing_relation_check_sql())
    op.execute(
        f"""
        ALTER TABLE memory_lineage
            ADD CONSTRAINT memory_lineage_relation_check
            CHECK (relation IN ({_check_in_values(_NEW_LINEAGE_CHECK_VALUES)}))
        """
    )

    # 2. decompose_operations table — substrate-level idempotency.
    op.create_table(
        "decompose_operations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "env_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("environments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("memories.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("dedupe_key", sa.Text(), nullable=False),
        sa.Column("request_fingerprint", sa.Text(), nullable=False),
        sa.Column(
            "child_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_by_agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id"),
            nullable=True,
        ),
        sa.CheckConstraint(
            "mode IN ('split', 'derive')",
            name="decompose_operations_mode_check",
        ),
    )
    op.create_index(
        "ix_decompose_operations_dedupe",
        "decompose_operations",
        ["env_id", "dedupe_key"],
        unique=True,
    )
    op.create_index(
        "ix_decompose_operations_source",
        "decompose_operations",
        ["source_id"],
    )

    # 3. Re-issue popularity trigger functions with the corrected whitelist.
    #    CREATE OR REPLACE atomically swaps the function body; existing
    #    AFTER INSERT OR DELETE ON memory_lineage and AFTER UPDATE OF status
    #    ON memories triggers (named ``..._t``) continue to point at the
    #    same function names without disruption.
    op.execute(_lineage_bump_function_sql(_NEW_POPULARITY_WHITELIST))
    op.execute(_status_flip_function_sql(_NEW_POPULARITY_WHITELIST))


def downgrade() -> None:
    # 3. Revert trigger function bodies to the 0017 whitelist (re-admit
    #    ``split_from`` to the load-bearing set).
    op.execute(_status_flip_function_sql(_OLD_POPULARITY_WHITELIST))
    op.execute(_lineage_bump_function_sql(_OLD_POPULARITY_WHITELIST))

    # 2. Drop decompose_operations.
    op.drop_index("ix_decompose_operations_source", table_name="decompose_operations")
    op.drop_index("ix_decompose_operations_dedupe", table_name="decompose_operations")
    op.drop_table("decompose_operations")

    # 1. Narrow memory_lineage CHECK back to the pre-0021 values. Safe iff
    #    no rows of ``split_from`` / ``derived_from`` exist — downgrade is
    #    best-effort by convention; tests arrange for this.
    op.execute("ALTER TABLE memory_lineage DROP CONSTRAINT IF EXISTS memory_lineage_relation_check")
    op.execute(
        f"""
        ALTER TABLE memory_lineage
            ADD CONSTRAINT memory_lineage_relation_check
            CHECK (relation IN ({_check_in_values(_OLD_LINEAGE_CHECK_VALUES)}))
        """
    )
