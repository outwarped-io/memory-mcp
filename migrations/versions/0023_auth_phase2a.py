"""Phase 2a auth substrate: env_acls table + nullable principal_id audit columns.

Revision ID: 0023_auth_phase2a
Revises: 0022_message_kind
Create Date: 2026-06-06

Ships the substrate changes from `docs/adr/0001-auth-phase-2a.md` §7. All
additions are additive and nullable so v0.17.x callers continue inserting
without code change:

1. ``env_acls`` — per-env principal/role grants. Bootstrapped by the first
   authenticated ``env_create_`` call (subtask 04). ``role`` is constrained
   to ``admin`` / ``writer`` / ``reader``; the hierarchy is enforced at the
   dispatcher, not at the table.
2. ``agents.principal_id`` — nullable OIDC/Entra subject claim. A partial
   unique index (``WHERE principal_id IS NOT NULL``) prevents two agent
   rows from claiming the same subject while leaving room for the existing
   synthetic ``agent-<uuid>`` rows that have no principal.
3. ``memories.created_by_principal_id`` / ``relations.created_by_principal_id``
   / ``memory_tombstones.created_by_principal_id`` — nullable audit columns.
   Legacy rows stay NULL by design (we do not back-fill an identity that
   wasn't asserted at write time; audit honesty wins over completeness —
   ADR §4 rationale).

Migration-number note
---------------------
ADR 0001 written 2026-06-06 reserved the number ``0022_auth_phase2a`` but
``0022_message_kind`` had already shipped on ``main`` in v0.17.2. Renumbered
to ``0023`` here; the ADR text is corrected in the same PR series.

Downgrade
---------
Drops the table + columns + index. All Phase 2a auth state is lost. Per ADR
§7 this is acceptable for a v0.x service.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0023_auth_phase2a"
down_revision: str | None = "0022_message_kind"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. env_acls -----------------------------------------------------------
    op.execute(
        """
        CREATE TABLE env_acls (
            env_id        uuid        NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
            principal_id  text        NOT NULL,
            role          text        NOT NULL,
            granted_at    timestamptz NOT NULL DEFAULT now(),
            granted_by    text        NOT NULL,
            PRIMARY KEY (env_id, principal_id),
            CONSTRAINT env_acls_role_check CHECK (role IN ('admin','writer','reader'))
        )
        """
    )
    op.execute("CREATE INDEX env_acls_principal_idx ON env_acls (principal_id)")

    # 2. agents.principal_id -----------------------------------------------
    op.execute("ALTER TABLE agents ADD COLUMN principal_id text")
    op.execute("CREATE UNIQUE INDEX agents_principal_id_uniq ON agents (principal_id) WHERE principal_id IS NOT NULL")

    # 3. created_by_principal_id audit columns -----------------------------
    op.execute("ALTER TABLE memories           ADD COLUMN created_by_principal_id text")
    op.execute("ALTER TABLE relations          ADD COLUMN created_by_principal_id text")
    op.execute("ALTER TABLE memory_tombstones  ADD COLUMN created_by_principal_id text")


def downgrade() -> None:
    op.execute("ALTER TABLE memory_tombstones  DROP COLUMN IF EXISTS created_by_principal_id")
    op.execute("ALTER TABLE relations          DROP COLUMN IF EXISTS created_by_principal_id")
    op.execute("ALTER TABLE memories           DROP COLUMN IF EXISTS created_by_principal_id")
    op.execute("DROP INDEX IF EXISTS agents_principal_id_uniq")
    op.execute("ALTER TABLE agents             DROP COLUMN IF EXISTS principal_id")
    op.execute("DROP TABLE IF EXISTS env_acls CASCADE")
