"""Schema assertions for migration 0018 — authority weighting columns (Phase 1e).

These tests verify the four ``ref_authority_*`` columns and the generated
``reference_authority`` total exist with the correct types, defaults, and
nullability — the migration itself runs as part of the testcontainer setup
(see ``conftest.postgres_session_factories``), so reaching the assertions
implies the upgrade path is healthy. A round-trip downgrade smoke is
covered separately in ``test_authority_migration_roundtrip``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from .conftest import SessionPairFactory

pytestmark = pytest.mark.integration


async def _mk_env(session) -> UUID:
    env_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO environments (id, name, kind, default_embedding_model_id) "
            "VALUES (:id, :name, 'test', 'test-embedding')"
        ),
        {"id": env_id, "name": f"authcols-{env_id}"},
    )
    return env_id


async def _mk_mem(session, env_id: UUID) -> UUID:
    mem_id = uuid4()
    await session.execute(
        text("INSERT INTO memories (id, env_id, kind, status, body) VALUES (:id, :env_id, 'fact', 'active', 'b')"),
        {"id": mem_id, "env_id": env_id},
    )
    return mem_id


async def test_authority_columns_default_to_zero(postgres_session_factories: SessionPairFactory):
    """New memories start with all four authority columns at 0 and the total at 0."""
    pair = postgres_session_factories()
    async with pair[0]() as session:
        env_id = await _mk_env(session)
        mem_id = await _mk_mem(session, env_id)
        await session.commit()

        row = (
            await session.execute(
                text(
                    "SELECT ref_authority_rel_link, ref_authority_lineage, "
                    "       ref_authority_task, ref_authority_playbook, "
                    "       reference_authority, authority_last_recount_at "
                    "FROM memories WHERE id = :id"
                ),
                {"id": mem_id},
            )
        ).one()
        assert row.ref_authority_rel_link == 0
        assert row.ref_authority_lineage == 0
        assert row.ref_authority_task == 0
        assert row.ref_authority_playbook == 0
        assert row.reference_authority == 0
        # staleness marker — NULL until first recount writes to this memory
        assert row.authority_last_recount_at is None


async def test_generated_reference_authority_sums_per_kind(postgres_session_factories: SessionPairFactory):
    """``reference_authority`` is GENERATED ALWAYS — writes to per-kind cols re-sum."""
    pair = postgres_session_factories()
    async with pair[0]() as session:
        env_id = await _mk_env(session)
        mem_id = await _mk_mem(session, env_id)
        await session.commit()

        await session.execute(
            text(
                "UPDATE memories SET "
                "  ref_authority_rel_link = 1.250000, "
                "  ref_authority_lineage  = 2.500000, "
                "  ref_authority_task     = 0.750000, "
                "  ref_authority_playbook = 0.500000  "
                "WHERE id = :id"
            ),
            {"id": mem_id},
        )
        await session.commit()

        total = (
            await session.execute(
                text("SELECT reference_authority FROM memories WHERE id = :id"),
                {"id": mem_id},
            )
        ).scalar_one()
        # 1.25 + 2.5 + 0.75 + 0.5 = 5.0
        assert float(total) == 5.0


async def test_authority_columns_reject_direct_write_to_generated_column(
    postgres_session_factories: SessionPairFactory,
):
    """``reference_authority`` is GENERATED — direct writes must fail."""
    pair = postgres_session_factories()
    async with pair[0]() as session:
        env_id = await _mk_env(session)
        mem_id = await _mk_mem(session, env_id)
        await session.commit()

        with pytest.raises(Exception):
            await session.execute(
                text("UPDATE memories SET reference_authority = 9.99 WHERE id = :id"),
                {"id": mem_id},
            )
        await session.rollback()


async def test_authority_index_exists_and_is_partial(postgres_session_factories: SessionPairFactory):
    """``memories_reference_authority_idx`` exists, is partial on status='active'."""
    pair = postgres_session_factories()
    async with pair[0]() as session:
        row = (
            await session.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE schemaname='public' AND indexname='memories_reference_authority_idx'"
                )
            )
        ).one_or_none()
        assert row is not None, "memories_reference_authority_idx not created"
        defn = row.indexdef.lower()
        assert "reference_authority desc" in defn
        assert "created_at desc" in defn
        assert "id desc" in defn
        assert "where (status = 'active'::text)" in defn or "where status = 'active'" in defn


async def test_0019_salience_formula_version_column(
    postgres_session_factories: SessionPairFactory,
):
    """Migration 0019 adds ``memories.salience_formula_version`` —
    NOT NULL, default 0, integer. New rows insert with version=0 by
    default. Verifies the upgrade path landed; downgrade is covered
    indirectly by the testcontainer round-trip in CI.
    """
    pair = postgres_session_factories()
    async with pair[0]() as session:
        env_id = await _mk_env(session)
        mem_id = await _mk_mem(session, env_id)
        await session.commit()

        # 1. Column exists, default applied, integer-typed.
        row = (
            await session.execute(
                text("SELECT salience_formula_version FROM memories WHERE id = :id"),
                {"id": mem_id},
            )
        ).one()
        assert row.salience_formula_version == 0

        # 2. Schema metadata: NOT NULL + integer + default 0.
        meta = (
            await session.execute(
                text(
                    "SELECT data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_name = 'memories' "
                    "  AND column_name = 'salience_formula_version'"
                )
            )
        ).one()
        assert meta.data_type == "integer"
        assert meta.is_nullable == "NO"
        # Postgres normalizes the literal default; just confirm '0' is in there.
        assert meta.column_default is not None
        assert "0" in meta.column_default

        # 3. Bump-then-read sanity: column accepts arbitrary positive ints.
        await session.execute(
            text("UPDATE memories SET salience_formula_version = 42 WHERE id = :id"),
            {"id": mem_id},
        )
        await session.commit()
        bumped = (
            await session.execute(
                text("SELECT salience_formula_version FROM memories WHERE id = :id"),
                {"id": mem_id},
            )
        ).scalar_one()
        assert bumped == 42


async def test_0020_compose_dedupe_key_column_and_partial_unique_index(
    postgres_session_factories: SessionPairFactory,
):
    """Migration 0020 adds ``memories.compose_dedupe_key`` (TEXT NULL) and a
    partial unique index on ``(env_id, compose_dedupe_key)`` scoped to non-NULL
    keys.

    The partial-uniqueness contract is the load-bearing piece of Phase 2's
    idempotency story: two concurrent identical ``mem_compose`` calls must
    race onto the same winning row. The race-recovery path (savepoint +
    re-fetch on ``UniqueViolation``) lives in ``composers.py``; this test
    asserts the index exists, is unique, is partial, and lets multiple NULLs
    coexist.
    """
    pair = postgres_session_factories()
    async with pair[0]() as session:
        env_id = await _mk_env(session)

        # 1. Column exists, defaults to NULL, no NOT NULL constraint.
        meta = (
            await session.execute(
                text(
                    "SELECT data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_name = 'memories' "
                    "  AND column_name = 'compose_dedupe_key'"
                )
            )
        ).one()
        assert meta.data_type == "text"
        assert meta.is_nullable == "YES"
        assert meta.column_default is None

        # 2. Partial unique index exists with the expected predicate.
        row = (
            await session.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE schemaname='public' "
                    "  AND indexname='ix_memories_compose_dedupe'"
                )
            )
        ).one_or_none()
        assert row is not None, "ix_memories_compose_dedupe not created"
        defn = row.indexdef.lower()
        assert "unique index" in defn
        assert "env_id" in defn
        assert "compose_dedupe_key" in defn
        assert "where (compose_dedupe_key is not null)" in defn

        # 3. Multiple NULL keys are fine in the same env (partial predicate).
        mem_a = await _mk_mem(session, env_id)
        mem_b = await _mk_mem(session, env_id)
        await session.commit()

        # 4. Same dedupe-key + same env_id is rejected by the partial index.
        await session.execute(
            text("UPDATE memories SET compose_dedupe_key = 'duplicate-key' WHERE id = :id"),
            {"id": mem_a},
        )
        await session.commit()

        with pytest.raises(Exception):
            await session.execute(
                text("UPDATE memories SET compose_dedupe_key = 'duplicate-key' WHERE id = :id"),
                {"id": mem_b},
            )
            await session.commit()
        await session.rollback()

        # 5. Same dedupe-key in a DIFFERENT env is allowed (index is per env).
        other_env = await _mk_env(session)
        mem_c = await _mk_mem(session, other_env)
        await session.execute(
            text("UPDATE memories SET compose_dedupe_key = 'duplicate-key' WHERE id = :id"),
            {"id": mem_c},
        )
        await session.commit()

        # Final read: both rows hold the same key in their respective envs.
        keys = (
            await session.execute(
                text(
                    "SELECT id, env_id, compose_dedupe_key FROM memories "
                    "WHERE compose_dedupe_key = 'duplicate-key' "
                    "ORDER BY env_id"
                )
            )
        ).all()
        assert len(keys) == 2
        assert {k.env_id for k in keys} == {env_id, other_env}
