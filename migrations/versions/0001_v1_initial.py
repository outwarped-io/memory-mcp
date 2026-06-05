"""v1 initial schema

Creates the 18 v1 tables documented in design.md → "Schema (Postgres) — Final v1":

  environments, agents, sessions, tokens, env_grants,
  memories (with generated body_tsv + FTS GIN index),
  entities, entity_aliases, graph_nodes, relations,
  tags, memory_tags,
  audit_log, memory_sources, memory_lineage,
  outbox, outbox_delivery, projection_state.

Out of scope (deferred):
  * memory_assertions       — v1.5 (conflict detection)
  * dream_runs, proposals   — Phase 2 (dream worker)

ENUM-shaped columns use ``text + CHECK`` rather than Postgres ``CREATE TYPE``
so we can extend them without a follow-up migration.

This revision incorporates the rubber-duck checkpoint #1 fixes:
  * Cross-env integrity is enforced via composite ``(id, env_id)`` FKs on
    entity_aliases / memory_tags / relations (BLOCKER).
  * graph_nodes uses explicit nullable ``memory_id`` / ``entity_id`` FKs
    instead of polymorphic ``record_id`` (BLOCKER).
  * Outbox uniqueness collapses to ``(aggregate_type, aggregate_id,
    aggregate_version)`` so a single version emits a single event (BLOCKER).
  * Outbox carries ``env_id`` so workers, replay, and projection_state can
    operate per-env without parsing payload (SHOULD-FIX).
  * ``superseded_by`` uses ON DELETE RESTRICT and is enforced via CHECK to
    match ``status='superseded'`` exactly (SHOULD-FIX).
  * Optimistic-locking has ``version > 0`` CHECKs and a per-table trigger
    rejecting non-monotonic decrements; access-tracking updates that don't
    change the version are still allowed (SHOULD-FIX).
  * Salience / confidence / counters are bounded with CHECKs (SHOULD-FIX).
  * Audit log carries ``env_id``, ``subject_hash``, ``redacted_at``,
    ``redaction_reason`` (SHOULD-FIX, GDPR).
  * Outbox/delivery indexes are reshaped for the projection-worker's hot
    path (SHOULD-FIX).
  * memory_lineage rejects self-cycles (NICE-TO-HAVE).

Revision ID: 0001_v1_initial
Revises:
Create Date: 2026-05-06
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001_v1_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # Required extensions.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")  # gen_random_uuid()
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")

    # ---- shared trigger function: optimistic-lock guard --------------------
    # Allows updates that don't touch ``version`` (e.g. access tracking) and
    # rejects any update that decreases ``version``. Strict increment on real
    # edits is enforced by the application via
    # ``WHERE id = $1 AND version = $expected ... SET version = version + 1``.
    op.execute("""
        CREATE OR REPLACE FUNCTION require_version_monotonic() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.version < OLD.version THEN
                RAISE EXCEPTION 'version must be monotonic non-decreasing: % -> %',
                    OLD.version, NEW.version
                    USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END
        $$
    """)

    # ---- environments / agents / sessions / tokens / env_grants ------------

    op.execute("""
        CREATE TABLE environments (
            id                          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name                        text NOT NULL UNIQUE,
            kind                        text,
            retention_policy            jsonb NOT NULL DEFAULT '{}'::jsonb,
            default_embedding_model_id  text NOT NULL,
            created_at                  timestamptz NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE agents (
            id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name          text NOT NULL,
            created_at    timestamptz NOT NULL DEFAULT now(),
            last_seen_at  timestamptz
        )
    """)

    op.execute("""
        CREATE TABLE sessions (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id    uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            started_at  timestamptz NOT NULL DEFAULT now(),
            ended_at    timestamptz
        )
    """)
    op.execute("CREATE INDEX sessions_agent_idx ON sessions(agent_id, started_at DESC)")

    op.execute("""
        CREATE TABLE tokens (
            token_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id        uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            hashed_secret   bytea NOT NULL,
            scopes          text[] NOT NULL DEFAULT ARRAY[]::text[],
            created_at      timestamptz NOT NULL DEFAULT now(),
            expires_at      timestamptz,
            revoked_at      timestamptz
        )
    """)
    op.execute("CREATE INDEX tokens_agent_idx ON tokens(agent_id) WHERE revoked_at IS NULL")

    op.execute("""
        CREATE TABLE env_grants (
            env_id    uuid NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
            agent_id  uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            role      text NOT NULL CHECK (role IN ('read','write','admin')),
            granted_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (env_id, agent_id)
        )
    """)
    op.execute("CREATE INDEX env_grants_agent_idx ON env_grants(agent_id)")

    # ---- memories ----------------------------------------------------------

    op.execute("""
        CREATE TABLE memories (
            id                        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            env_id                    uuid NOT NULL REFERENCES environments(id) ON DELETE RESTRICT,
            kind                      text NOT NULL CHECK (kind IN
                ('fact','procedure','event','decision','preference','observation','snippet')),
            status                    text NOT NULL CHECK (status IN
                ('proposed','active','stale','archived','superseded','retired'))
                DEFAULT 'active',
            title                     text,
            body                      text NOT NULL,
            body_tsv                  tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
                setweight(to_tsvector('english', coalesce(body,'')), 'B')
            ) STORED,
            salience                  real NOT NULL DEFAULT 0.5
                CHECK (salience BETWEEN 0 AND 1),
            confidence                real NOT NULL DEFAULT 0.5
                CHECK (confidence BETWEEN 0 AND 1),
            access_count              bigint NOT NULL DEFAULT 0
                CHECK (access_count >= 0),
            last_accessed_at          timestamptz,
            pinned                    boolean NOT NULL DEFAULT false,
            negative_feedback_count   integer NOT NULL DEFAULT 0
                CHECK (negative_feedback_count >= 0),
            verified_at               timestamptz,
            created_at                timestamptz NOT NULL DEFAULT now(),
            updated_at                timestamptz NOT NULL DEFAULT now(),
            expires_at                timestamptz,
            superseded_by             uuid REFERENCES memories(id) ON DELETE RESTRICT,
            metadata                  jsonb NOT NULL DEFAULT '{}'::jsonb,
            version                   bigint NOT NULL DEFAULT 1 CHECK (version > 0),
            CONSTRAINT memories_superseded_status_chk CHECK (
                (status = 'superseded' AND superseded_by IS NOT NULL)
             OR (status <> 'superseded' AND superseded_by IS NULL)
            ),
            CONSTRAINT memories_not_self_superseded_chk CHECK (
                superseded_by IS NULL OR superseded_by <> id
            )
        )
    """)
    op.execute("ALTER TABLE memories ADD CONSTRAINT memories_id_env_uniq UNIQUE (id, env_id)")
    op.execute("CREATE INDEX memories_env_status_idx ON memories(env_id, status)")
    op.execute("CREATE INDEX memories_env_kind_idx   ON memories(env_id, kind)")
    op.execute("CREATE INDEX memories_updated_idx    ON memories(updated_at DESC)")
    op.execute("CREATE INDEX memories_pinned_idx     ON memories(env_id) WHERE pinned")
    op.execute("CREATE INDEX memories_expires_idx    ON memories(expires_at) WHERE expires_at IS NOT NULL")
    op.execute("CREATE INDEX memories_superseded_idx ON memories(superseded_by) WHERE superseded_by IS NOT NULL")
    op.execute("CREATE INDEX memories_metadata_idx   ON memories USING GIN (metadata jsonb_path_ops)")
    # Global FTS GIN — used for canonical/admin scans across all statuses.
    op.execute("CREATE INDEX memories_body_tsv_idx   ON memories USING GIN (body_tsv)")
    # Hot-path FTS GIN — env+status filtered, only the statuses that ``memory_search`` exposes.
    op.execute(
        "CREATE INDEX memories_search_active_gin_idx "
        "ON memories USING GIN (env_id, status, body_tsv) "
        "WHERE status IN ('active','stale')"
    )
    # Optimistic-lock guard.
    op.execute(
        "CREATE TRIGGER memories_version_monotonic "
        "BEFORE UPDATE ON memories FOR EACH ROW "
        "EXECUTE FUNCTION require_version_monotonic()"
    )

    # ---- entities / entity_aliases ----------------------------------------

    op.execute("""
        CREATE TABLE entities (
            id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            env_id          uuid NOT NULL REFERENCES environments(id) ON DELETE RESTRICT,
            kind            text NOT NULL,
            canonical_name  text NOT NULL,
            normalized_name text NOT NULL,
            metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at      timestamptz NOT NULL DEFAULT now(),
            updated_at      timestamptz NOT NULL DEFAULT now(),
            version         bigint NOT NULL DEFAULT 1 CHECK (version > 0),
            UNIQUE(env_id, normalized_name)
        )
    """)
    op.execute("ALTER TABLE entities ADD CONSTRAINT entities_id_env_uniq UNIQUE (id, env_id)")
    op.execute("CREATE INDEX entities_env_kind_idx ON entities(env_id, kind)")
    op.execute(
        "CREATE TRIGGER entities_version_monotonic "
        "BEFORE UPDATE ON entities FOR EACH ROW "
        "EXECUTE FUNCTION require_version_monotonic()"
    )

    op.execute("""
        CREATE TABLE entity_aliases (
            entity_id        uuid NOT NULL,
            env_id           uuid NOT NULL,
            alias            text NOT NULL,
            normalized_alias text NOT NULL,
            created_at       timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (entity_id, normalized_alias),
            UNIQUE (env_id, normalized_alias),
            CONSTRAINT entity_aliases_entity_env_fk
                FOREIGN KEY (entity_id, env_id)
                REFERENCES entities(id, env_id) ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX entity_aliases_entity_idx ON entity_aliases(entity_id)")

    # ---- graph_nodes / relations ------------------------------------------
    # Type-safe registry: exactly one of memory_id / entity_id is set, matched
    # by node_type. Composite UNIQUE(id, env_id) lets relations enforce same-env
    # via composite FK.

    op.execute("""
        CREATE TABLE graph_nodes (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            env_id      uuid NOT NULL REFERENCES environments(id) ON DELETE RESTRICT,
            node_type   text NOT NULL CHECK (node_type IN ('entity','memory')),
            memory_id   uuid REFERENCES memories(id) ON DELETE CASCADE,
            entity_id   uuid REFERENCES entities(id) ON DELETE CASCADE,
            created_at  timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT graph_nodes_exactly_one_target_chk CHECK (
                (node_type = 'memory' AND memory_id IS NOT NULL AND entity_id IS NULL)
             OR (node_type = 'entity' AND entity_id IS NOT NULL AND memory_id IS NULL)
            )
        )
    """)
    op.execute("ALTER TABLE graph_nodes ADD CONSTRAINT graph_nodes_id_env_uniq UNIQUE (id, env_id)")
    op.execute("CREATE UNIQUE INDEX graph_nodes_memory_uniq ON graph_nodes(memory_id) WHERE memory_id IS NOT NULL")
    op.execute("CREATE UNIQUE INDEX graph_nodes_entity_uniq ON graph_nodes(entity_id) WHERE entity_id IS NOT NULL")
    op.execute("CREATE INDEX graph_nodes_env_idx ON graph_nodes(env_id)")

    op.execute("""
        CREATE TABLE relations (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            env_id      uuid NOT NULL REFERENCES environments(id) ON DELETE RESTRICT,
            src_node_id uuid NOT NULL,
            dst_node_id uuid NOT NULL,
            type        text NOT NULL,
            properties  jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at  timestamptz NOT NULL DEFAULT now(),
            updated_at  timestamptz NOT NULL DEFAULT now(),
            version     bigint NOT NULL DEFAULT 1 CHECK (version > 0),
            UNIQUE(src_node_id, dst_node_id, type),
            CONSTRAINT relations_src_env_fk
                FOREIGN KEY (src_node_id, env_id)
                REFERENCES graph_nodes(id, env_id) ON DELETE CASCADE,
            CONSTRAINT relations_dst_env_fk
                FOREIGN KEY (dst_node_id, env_id)
                REFERENCES graph_nodes(id, env_id) ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX relations_env_type_idx ON relations(env_id, type)")
    op.execute("CREATE INDEX relations_src_idx ON relations(src_node_id, type)")
    op.execute("CREATE INDEX relations_dst_idx ON relations(dst_node_id, type)")
    op.execute(
        "CREATE TRIGGER relations_version_monotonic "
        "BEFORE UPDATE ON relations FOR EACH ROW "
        "EXECUTE FUNCTION require_version_monotonic()"
    )

    # ---- tags / memory_tags ------------------------------------------------

    op.execute("""
        CREATE TABLE tags (
            id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            env_id  uuid NOT NULL REFERENCES environments(id) ON DELETE RESTRICT,
            name    text NOT NULL,
            UNIQUE(env_id, name)
        )
    """)
    op.execute("ALTER TABLE tags ADD CONSTRAINT tags_id_env_uniq UNIQUE (id, env_id)")

    op.execute("""
        CREATE TABLE memory_tags (
            memory_id  uuid NOT NULL,
            tag_id     uuid NOT NULL,
            env_id     uuid NOT NULL,
            PRIMARY KEY (memory_id, tag_id),
            CONSTRAINT memory_tags_memory_env_fk
                FOREIGN KEY (memory_id, env_id) REFERENCES memories(id, env_id) ON DELETE CASCADE,
            CONSTRAINT memory_tags_tag_env_fk
                FOREIGN KEY (tag_id, env_id)    REFERENCES tags(id, env_id)    ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX memory_tags_tag_idx ON memory_tags(tag_id)")
    op.execute("CREATE INDEX memory_tags_env_idx ON memory_tags(env_id)")

    # ---- audit_log / memory_sources / memory_lineage ----------------------

    op.execute("""
        CREATE TABLE audit_log (
            id                bigserial PRIMARY KEY,
            record_type       text NOT NULL,
            record_id         uuid,
            env_id            uuid REFERENCES environments(id) ON DELETE SET NULL,
            op                text NOT NULL,
            at                timestamptz NOT NULL DEFAULT now(),
            by_agent_id       uuid REFERENCES agents(id) ON DELETE SET NULL,
            before            jsonb,
            after             jsonb,
            subject_hash      text,
            redacted_at       timestamptz,
            redaction_reason  text
        )
    """)
    op.execute(
        "CREATE INDEX audit_record_idx       ON audit_log(record_type, record_id, at DESC) WHERE record_id IS NOT NULL"
    )
    op.execute("CREATE INDEX audit_env_at_idx       ON audit_log(env_id, at DESC) WHERE env_id IS NOT NULL")
    op.execute("CREATE INDEX audit_agent_at_idx     ON audit_log(by_agent_id, at DESC) WHERE by_agent_id IS NOT NULL")
    op.execute("CREATE INDEX audit_subject_hash_idx ON audit_log(subject_hash) WHERE subject_hash IS NOT NULL")
    op.execute("CREATE INDEX audit_at_idx           ON audit_log(at DESC)")

    op.execute("""
        CREATE TABLE memory_sources (
            id            bigserial PRIMARY KEY,
            memory_id     uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            source_type   text NOT NULL CHECK (source_type IN
                ('session','file','url','llm','dream','user','agent','other')),
            source_ref    text,
            agent_id      uuid REFERENCES agents(id) ON DELETE SET NULL,
            created_at    timestamptz NOT NULL DEFAULT now(),
            evidence_span text
        )
    """)
    op.execute("CREATE INDEX memory_sources_memory_idx ON memory_sources(memory_id)")

    op.execute("""
        CREATE TABLE memory_lineage (
            parent_memory_id uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            child_memory_id  uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            relation         text NOT NULL CHECK (relation IN
                ('promoted_from','summarized_from','copied_from','moved_from','supersedes')),
            created_at       timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (parent_memory_id, child_memory_id, relation),
            CONSTRAINT memory_lineage_not_self_chk CHECK (parent_memory_id <> child_memory_id)
        )
    """)
    op.execute("CREATE INDEX memory_lineage_child_idx ON memory_lineage(child_memory_id)")

    # ---- outbox / outbox_delivery / projection_state ----------------------
    # ``env_id`` lives on the outbox row (not just the payload) so workers,
    # replay, and projection_state can operate per-env without parsing JSON.

    op.execute("""
        CREATE TABLE outbox (
            event_id          bigserial PRIMARY KEY,
            aggregate_type    text NOT NULL CHECK (aggregate_type IN
                ('memory','entity','relation','env')),
            aggregate_id      uuid NOT NULL,
            aggregate_version bigint NOT NULL CHECK (aggregate_version > 0),
            env_id            uuid NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
            op                text NOT NULL CHECK (op IN ('upsert','tombstone','update')),
            payload           jsonb NOT NULL,
            created_at        timestamptz NOT NULL DEFAULT now(),
            available_at      timestamptz NOT NULL DEFAULT now(),
            UNIQUE(aggregate_type, aggregate_id, aggregate_version)
        )
    """)
    op.execute("CREATE INDEX outbox_aggregate_idx     ON outbox(aggregate_id, aggregate_version)")
    op.execute("CREATE INDEX outbox_available_idx     ON outbox(available_at, event_id)")
    op.execute("CREATE INDEX outbox_env_event_idx     ON outbox(env_id, event_id)")
    op.execute("CREATE INDEX outbox_env_available_idx ON outbox(env_id, available_at, event_id)")

    op.execute("""
        CREATE TABLE outbox_delivery (
            event_id      bigint NOT NULL REFERENCES outbox(event_id) ON DELETE CASCADE,
            sink          text NOT NULL CHECK (sink IN ('qdrant','neo4j','pgvector')),
            status        text NOT NULL CHECK (status IN ('pending','in_flight','done','dead'))
                          DEFAULT 'pending',
            attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            locked_by     text,
            locked_until  timestamptz,
            last_error    text,
            done_at       timestamptz,
            PRIMARY KEY (event_id, sink),
            CONSTRAINT outbox_delivery_state_chk CHECK (
                (status = 'pending'   AND done_at IS NULL)
             OR (status = 'in_flight' AND locked_by IS NOT NULL
                                       AND locked_until IS NOT NULL
                                       AND done_at IS NULL)
             OR (status = 'done'      AND done_at IS NOT NULL)
             OR (status = 'dead'      AND last_error IS NOT NULL)
            )
        )
    """)
    # Hot path: pick up next pending events for a sink in event_id order.
    op.execute("CREATE INDEX outbox_delivery_pending_idx ON outbox_delivery(sink, event_id) WHERE status = 'pending'")
    # Lease reaper: find expired leases per sink ordered by lease expiry.
    op.execute(
        "CREATE INDEX outbox_delivery_expired_lease_idx "
        "ON outbox_delivery(sink, locked_until, event_id) WHERE status = 'in_flight'"
    )
    # Admin: list dead deliveries per sink in event order.
    op.execute("CREATE INDEX outbox_delivery_dead_idx ON outbox_delivery(sink, event_id) WHERE status = 'dead'")

    op.execute("""
        CREATE TABLE projection_state (
            sink            text NOT NULL CHECK (sink IN ('qdrant','neo4j','pgvector')),
            env_id          uuid NOT NULL REFERENCES environments(id) ON DELETE CASCADE,
            last_event_id   bigint,
            last_success_at timestamptz,
            lag_seconds     numeric,
            status          text CHECK (status IS NULL OR status IN ('healthy','degraded','down','rebuilding')),
            last_error      text,
            PRIMARY KEY (sink, env_id)
        )
    """)


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS projection_state CASCADE")
    op.execute("DROP TABLE IF EXISTS outbox_delivery  CASCADE")
    op.execute("DROP TABLE IF EXISTS outbox           CASCADE")
    op.execute("DROP TABLE IF EXISTS memory_lineage   CASCADE")
    op.execute("DROP TABLE IF EXISTS memory_sources   CASCADE")
    op.execute("DROP TABLE IF EXISTS audit_log        CASCADE")
    op.execute("DROP TABLE IF EXISTS memory_tags      CASCADE")
    op.execute("DROP TABLE IF EXISTS tags             CASCADE")
    op.execute("DROP TABLE IF EXISTS relations        CASCADE")
    op.execute("DROP TABLE IF EXISTS graph_nodes      CASCADE")
    op.execute("DROP TABLE IF EXISTS entity_aliases   CASCADE")
    op.execute("DROP TABLE IF EXISTS entities         CASCADE")
    op.execute("DROP TABLE IF EXISTS memories         CASCADE")
    op.execute("DROP TABLE IF EXISTS env_grants       CASCADE")
    op.execute("DROP TABLE IF EXISTS tokens           CASCADE")
    op.execute("DROP TABLE IF EXISTS sessions         CASCADE")
    op.execute("DROP TABLE IF EXISTS agents           CASCADE")
    op.execute("DROP TABLE IF EXISTS environments     CASCADE")
    op.execute("DROP FUNCTION IF EXISTS require_version_monotonic()")
