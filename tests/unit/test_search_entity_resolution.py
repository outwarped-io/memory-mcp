"""Unit tests for ``memory_mcp.search.entity_resolution.resolve_query_entities``.

Strategy: mock the ``AsyncSession.execute`` to return controlled
canonical-name and alias rows; assert per-env bucketing, dedupe, and
hard-cap behaviour.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from memory_mcp.config import Settings
from memory_mcp.search.entity_resolution import resolve_query_entities


def _settings(**overrides) -> Settings:
    base = {
        "graph_search_max_resolved_entities_per_env": 8,
        "graph_search_max_resolved_entities_total": 16,
    }
    base.update(overrides)
    s = Settings()
    for k, v in base.items():
        object.__setattr__(s, k, v)
    return s


def _result_with_rows(rows: list[tuple[Any, ...]]):
    """Build a SQLAlchemy-result-like object whose ``.all()`` yields rows."""
    res = MagicMock()
    res.all.return_value = rows
    return res


def _make_session(canonical_rows: list[tuple], alias_rows: list[tuple]):
    """Mock a session that returns canonical rows on the first execute call
    and alias rows on the second."""
    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=[
            _result_with_rows(canonical_rows),
            _result_with_rows(alias_rows),
        ]
    )
    return session


# ---------------------------------------------------------------------------
# Empty-input short-circuits
# ---------------------------------------------------------------------------


def test_empty_mentions_returns_empty():
    session = MagicMock()
    out = asyncio.run(resolve_query_entities(
        session, mentions=[], env_ids=[uuid4()], settings=_settings(),
    ))
    assert out == {}
    session.execute.assert_not_called() if hasattr(session.execute, "assert_not_called") else None


def test_empty_env_ids_returns_empty():
    session = MagicMock()
    out = asyncio.run(resolve_query_entities(
        session, mentions=["x"], env_ids=[], settings=_settings(),
    ))
    assert out == {}


def test_only_blank_mentions_returns_empty():
    session = MagicMock()
    out = asyncio.run(resolve_query_entities(
        session, mentions=["", ""], env_ids=[uuid4()], settings=_settings(),
    ))
    assert out == {}


# ---------------------------------------------------------------------------
# Canonical + alias resolution
# ---------------------------------------------------------------------------


def test_canonical_match_resolves():
    env = uuid4()
    e1 = uuid4()
    session = _make_session(
        canonical_rows=[(e1, env, "service a")],
        alias_rows=[],
    )
    out = asyncio.run(resolve_query_entities(
        session, mentions=["service a"], env_ids=[env], settings=_settings(),
    ))
    assert out == {env: [e1]}


def test_alias_match_resolves():
    env = uuid4()
    e1 = uuid4()
    session = _make_session(
        canonical_rows=[],
        alias_rows=[(e1, env, "svc a")],
    )
    out = asyncio.run(resolve_query_entities(
        session, mentions=["svc a"], env_ids=[env], settings=_settings(),
    ))
    assert out == {env: [e1]}


def test_canonical_and_alias_dedupe_to_single_entity():
    """If an entity matches via both canonical name and alias for the same
    mention, it should appear ONCE per env."""
    env = uuid4()
    e1 = uuid4()
    session = _make_session(
        canonical_rows=[(e1, env, "service a")],
        alias_rows=[(e1, env, "service a")],
    )
    out = asyncio.run(resolve_query_entities(
        session, mentions=["service a"], env_ids=[env], settings=_settings(),
    ))
    assert out == {env: [e1]}


def test_per_env_bucketing():
    """Same mention may resolve to different entities in different envs."""
    env_a, env_b = uuid4(), uuid4()
    e_a, e_b = uuid4(), uuid4()
    session = _make_session(
        canonical_rows=[
            (e_a, env_a, "service a"),
            (e_b, env_b, "service a"),
        ],
        alias_rows=[],
    )
    out = asyncio.run(resolve_query_entities(
        session, mentions=["service a"],
        env_ids=[env_a, env_b],
        settings=_settings(),
    ))
    assert out == {env_a: [e_a], env_b: [e_b]}


def test_multiple_mentions_preserve_order_per_env():
    env = uuid4()
    e1, e2 = uuid4(), uuid4()
    session = _make_session(
        canonical_rows=[
            (e1, env, "first"),
            (e2, env, "second"),
        ],
        alias_rows=[],
    )
    out = asyncio.run(resolve_query_entities(
        session, mentions=["first", "second"], env_ids=[env], settings=_settings(),
    ))
    assert out[env] == [e1, e2]


# ---------------------------------------------------------------------------
# Hard caps
# ---------------------------------------------------------------------------


def test_per_env_cap_truncates():
    env = uuid4()
    ids = [uuid4() for _ in range(10)]
    canon = [(eid, env, f"m{i}") for i, eid in enumerate(ids)]
    session = _make_session(canonical_rows=canon, alias_rows=[])
    out = asyncio.run(resolve_query_entities(
        session,
        mentions=[f"m{i}" for i in range(10)],
        env_ids=[env],
        settings=_settings(graph_search_max_resolved_entities_per_env=3),
    ))
    assert len(out[env]) == 3


def test_total_cap_truncates_across_envs():
    env_a, env_b = uuid4(), uuid4()
    ids_a = [uuid4() for _ in range(5)]
    ids_b = [uuid4() for _ in range(5)]
    canon = (
        [(eid, env_a, f"m{i}") for i, eid in enumerate(ids_a)]
        + [(eid, env_b, f"m{i}") for i, eid in enumerate(ids_b)]
    )
    session = _make_session(canonical_rows=canon, alias_rows=[])
    out = asyncio.run(resolve_query_entities(
        session,
        mentions=[f"m{i}" for i in range(5)],
        env_ids=[env_a, env_b],
        settings=_settings(
            graph_search_max_resolved_entities_per_env=10,
            graph_search_max_resolved_entities_total=4,
        ),
    ))
    total = sum(len(v) for v in out.values())
    assert total == 4
    # env_a is processed first (input order) so its entities exhaust first.
    assert len(out[env_a]) == 4
    assert env_b not in out


# ---------------------------------------------------------------------------
# Dedupe across mentions within same env
# ---------------------------------------------------------------------------


def test_same_entity_via_two_mentions_appears_once():
    """If e1 has both canonical 'foo' and alias 'bar', and the query mentions
    both, the entity must appear once per env."""
    env = uuid4()
    e1 = uuid4()
    session = _make_session(
        canonical_rows=[(e1, env, "foo")],
        alias_rows=[(e1, env, "bar")],
    )
    out = asyncio.run(resolve_query_entities(
        session, mentions=["foo", "bar"], env_ids=[env], settings=_settings(),
    ))
    assert out[env] == [e1]
