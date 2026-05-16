"""Unit tests for ``memory_mcp.db.outbox`` — pure-Python coverage.

Covers ``resolve_sinks_for`` (pure function over ``Settings``) plus the
``EnqueuedEvent`` no-op short-circuit.  DB-side behavior
(``ON CONFLICT DO NOTHING`` watermark seed, idempotency on duplicate
versions, atomic insert) is exercised by the integration smoke test
against a real Postgres.
"""

from __future__ import annotations

import uuid
from typing import cast
from unittest.mock import AsyncMock

import pytest

from memory_mcp.config import Settings
from memory_mcp.db.outbox import (
    EnqueuedEvent,
    enqueue_event,
    resolve_sinks_for,
)
from memory_mcp.db.types import OutboxAggregateType, OutboxOp, OutboxSink


def _settings(**overrides: object) -> Settings:
    """Build a ``Settings`` instance with arbitrary overrides for tests."""
    return Settings(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolve_sinks_for
# ---------------------------------------------------------------------------


class TestResolveSinksFor:
    def test_memory_with_qdrant_backend(self) -> None:
        s = _settings(vector_backend="qdrant")
        assert resolve_sinks_for(OutboxAggregateType.memory, s) == (OutboxSink.qdrant,)

    def test_memory_with_pgvector_backend(self) -> None:
        s = _settings(vector_backend="pgvector")
        assert resolve_sinks_for(OutboxAggregateType.memory, s) == (OutboxSink.pgvector,)

    def test_entity_with_neo4j_backend(self) -> None:
        s = _settings(graph_backend="neo4j")
        assert resolve_sinks_for(OutboxAggregateType.entity, s) == (OutboxSink.neo4j,)

    def test_relation_with_neo4j_backend(self) -> None:
        s = _settings(graph_backend="neo4j")
        assert resolve_sinks_for(OutboxAggregateType.relation, s) == (OutboxSink.neo4j,)

    def test_task_with_neo4j_backend(self) -> None:
        s = _settings(graph_backend="neo4j")
        assert resolve_sinks_for(OutboxAggregateType.task, s) == (OutboxSink.neo4j,)

    def test_entity_with_postgres_backend_yields_no_sinks(self) -> None:
        s = _settings(graph_backend="postgres")
        assert resolve_sinks_for(OutboxAggregateType.entity, s) == ()

    def test_relation_with_postgres_backend_yields_no_sinks(self) -> None:
        s = _settings(graph_backend="postgres")
        assert resolve_sinks_for(OutboxAggregateType.relation, s) == ()

    def test_task_with_postgres_backend_yields_no_sinks(self) -> None:
        s = _settings(graph_backend="postgres")
        assert resolve_sinks_for(OutboxAggregateType.task, s) == ()

    def test_env_aggregate_yields_no_sinks(self) -> None:
        # env aggregate type carries metadata only — never projects in v1.
        for vb in ("qdrant", "pgvector"):
            for gb in ("neo4j", "postgres"):
                s = _settings(vector_backend=vb, graph_backend=gb)
                assert resolve_sinks_for(OutboxAggregateType.env, s) == ()

    def test_default_settings_route_memory_to_qdrant_and_entity_to_neo4j(self) -> None:
        # Sanity check the default-config routing the rest of the codebase relies on.
        s = _settings()
        assert resolve_sinks_for(OutboxAggregateType.memory, s) == (OutboxSink.qdrant,)
        assert resolve_sinks_for(OutboxAggregateType.entity, s) == (OutboxSink.neo4j,)
        assert resolve_sinks_for(OutboxAggregateType.relation, s) == (OutboxSink.neo4j,)
        assert resolve_sinks_for(OutboxAggregateType.task, s) == (OutboxSink.neo4j,)
        assert resolve_sinks_for(OutboxAggregateType.env, s) == ()


# ---------------------------------------------------------------------------
# enqueue_event no-op short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEnqueueEventNoSinks:
    async def test_returns_no_op_when_aggregate_has_no_sinks(self) -> None:
        """``env`` aggregate type → no sinks → no DB calls.

        Critical: when ``resolve_sinks_for`` returns ``()`` the helper must
        not touch the session at all. Otherwise canonical writers paying
        the cost of a no-op outbox row when no projection consumes it.
        """
        session = AsyncMock()
        s = _settings()

        result = await enqueue_event(
            session,
            aggregate_type=OutboxAggregateType.env,
            aggregate_id=uuid.uuid4(),
            aggregate_version=1,
            env_id=uuid.uuid4(),
            op=OutboxOp.upsert,
            payload={"name": "work"},
            settings=s,
        )

        assert result == EnqueuedEvent(event_id=None, sinks=())
        session.execute.assert_not_called()

    async def test_returns_no_op_when_postgres_graph_backend(self) -> None:
        """``entity`` aggregate type with ``graph_backend=postgres`` is also a no-op."""
        session = AsyncMock()
        s = _settings(graph_backend="postgres")

        result = await enqueue_event(
            session,
            aggregate_type=OutboxAggregateType.entity,
            aggregate_id=uuid.uuid4(),
            aggregate_version=1,
            env_id=uuid.uuid4(),
            op=OutboxOp.upsert,
            payload={"canonical_name": "foo"},
            settings=s,
        )

        assert result == EnqueuedEvent(event_id=None, sinks=())
        session.execute.assert_not_called()

    async def test_explicit_empty_sinks_short_circuits(self) -> None:
        """Caller can force a no-op via ``sinks=()`` — useful for admin tools."""
        session = AsyncMock()
        s = _settings()

        result = await enqueue_event(
            session,
            aggregate_type=OutboxAggregateType.memory,
            aggregate_id=uuid.uuid4(),
            aggregate_version=1,
            env_id=uuid.uuid4(),
            op=OutboxOp.upsert,
            payload={"title": "x"},
            sinks=(),
            settings=s,
        )

        assert result == EnqueuedEvent(event_id=None, sinks=())
        session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# enqueue_event: explicit sinks override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEnqueueEventExplicitSinks:
    async def test_explicit_sinks_override_resolution(self) -> None:
        """Passing explicit sinks bypasses ``resolve_sinks_for``.

        The first ``execute`` call inserts the outbox row and returns
        ``event_id=42``; subsequent calls are the delivery batch and
        the projection_state upsert. We verify count and the resulting
        ``EnqueuedEvent`` payload — full SQL composition is covered by
        the integration smoke.
        """
        session = AsyncMock()

        # The first call (insert outbox) returns event_id; later calls are bulk-insert
        # deliveries + upsert projection state — their result is unused.
        outbox_result = AsyncMock()
        outbox_result.scalar_one = lambda: 42  # type: ignore[method-assign]
        session.execute = AsyncMock(side_effect=[outbox_result, AsyncMock(), AsyncMock()])

        s = _settings(graph_backend="postgres")  # would resolve to () for entity

        result = await enqueue_event(
            session,
            aggregate_type=OutboxAggregateType.entity,
            aggregate_id=uuid.uuid4(),
            aggregate_version=1,
            env_id=uuid.uuid4(),
            op=OutboxOp.upsert,
            payload={"canonical_name": "foo"},
            sinks=(OutboxSink.neo4j,),  # explicit override
            settings=s,
        )

        assert result == EnqueuedEvent(event_id=42, sinks=(OutboxSink.neo4j,))
        # 3 calls: outbox insert, delivery bulk insert, projection_state upsert.
        assert cast(AsyncMock, session.execute).await_count == 3
