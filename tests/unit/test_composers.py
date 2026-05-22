"""Unit tests for ``memory_mcp.composers`` helpers (Phase 2 B3c+).

These tests exercise pure functions in ``composers.py`` — at B3c that
means just the deterministic dedupe-key helper.  Transaction-body tests
land at B3d as integration tests (testcontainers Postgres + outbox +
audit) in ``tests/integration/test_compose_transaction.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from memory_mcp.composers import _compute_compose_dedupe_key
from memory_mcp_schemas.compose import MemComposeRequest, MemComposeTarget
from memory_mcp_schemas.enums import MemoryKind


def _env_id() -> UUID:
    return uuid4()


def _two_sources() -> list[UUID]:
    return [uuid4(), uuid4()]


def _target(
    *,
    kind: MemoryKind = MemoryKind.fact,
    body: str = "combined body",
    title: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    salience: float | None = None,
    confidence: float | None = None,
    pinned: bool = False,
    decision_meta: dict | None = None,
) -> MemComposeTarget:
    return MemComposeTarget(
        kind=kind,
        title=title,
        body=body,
        tags=tags,
        metadata=metadata or {},
        salience=salience,
        confidence=confidence,
        pinned=pinned,
        decision_meta=decision_meta,
    )


def _req(
    *,
    sources: list[UUID] | None = None,
    target: MemComposeTarget | None = None,
    mode: str = "promote",
    idempotency_key: str | None = None,
) -> MemComposeRequest:
    return MemComposeRequest(
        source_ids=sources or _two_sources(),
        target=target or _target(),
        mode=mode,  # type: ignore[arg-type]
        idempotency_key=idempotency_key,
    )


# ---------------------------------------------------------------------------
# Determinism + shape
# ---------------------------------------------------------------------------


def test_dedupe_key_is_deterministic_same_request_same_key() -> None:
    sources = _two_sources()
    target = _target(body="x")
    env = _env_id()
    r1 = _req(sources=sources, target=target)
    r2 = _req(sources=sources, target=target)
    assert _compute_compose_dedupe_key(r1, env_id=env) == _compute_compose_dedupe_key(
        r2, env_id=env
    )


def test_dedupe_key_is_32_lowercase_hex_chars() -> None:
    key = _compute_compose_dedupe_key(_req(), env_id=_env_id())
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# Field sensitivity — these MUST differ
# ---------------------------------------------------------------------------


def test_dedupe_key_changes_with_mode() -> None:
    sources = _two_sources()
    target = _target()
    env = _env_id()
    promote = _compute_compose_dedupe_key(
        _req(sources=sources, target=target, mode="promote"), env_id=env
    )
    merge = _compute_compose_dedupe_key(
        _req(sources=sources, target=target, mode="merge"), env_id=env
    )
    assert promote != merge


def test_dedupe_key_changes_with_kind() -> None:
    sources = _two_sources()
    env = _env_id()
    k1 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(kind=MemoryKind.fact)), env_id=env
    )
    k2 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(kind=MemoryKind.observation)), env_id=env
    )
    assert k1 != k2


def test_dedupe_key_changes_with_title() -> None:
    sources = _two_sources()
    env = _env_id()
    k1 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(title=None)), env_id=env
    )
    k2 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(title="t")), env_id=env
    )
    assert k1 != k2


def test_dedupe_key_changes_with_body() -> None:
    sources = _two_sources()
    env = _env_id()
    k1 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(body="a")), env_id=env
    )
    k2 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(body="b")), env_id=env
    )
    assert k1 != k2


def test_dedupe_key_changes_with_tags_content() -> None:
    sources = _two_sources()
    env = _env_id()
    k1 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(tags=["a"])), env_id=env
    )
    k2 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(tags=["b"])), env_id=env
    )
    assert k1 != k2


def test_dedupe_key_changes_with_env_id() -> None:
    sources = _two_sources()
    target = _target()
    r = _req(sources=sources, target=target)
    e1 = _env_id()
    e2 = _env_id()
    assert _compute_compose_dedupe_key(r, env_id=e1) != _compute_compose_dedupe_key(
        r, env_id=e2
    )


def test_dedupe_key_changes_with_source_ids() -> None:
    target = _target()
    env = _env_id()
    sources_a = _two_sources()
    sources_b = _two_sources()
    assert _compute_compose_dedupe_key(
        _req(sources=sources_a, target=target), env_id=env
    ) != _compute_compose_dedupe_key(_req(sources=sources_b, target=target), env_id=env)


def test_dedupe_key_changes_with_pinned_flag() -> None:
    sources = _two_sources()
    env = _env_id()
    k1 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(pinned=False)), env_id=env
    )
    k2 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(pinned=True)), env_id=env
    )
    assert k1 != k2


# ---------------------------------------------------------------------------
# Canonicalisation — these MUST be equal
# ---------------------------------------------------------------------------


def test_dedupe_key_invariant_under_source_order() -> None:
    a, b = uuid4(), uuid4()
    target = _target()
    env = _env_id()
    k1 = _compute_compose_dedupe_key(_req(sources=[a, b], target=target), env_id=env)
    k2 = _compute_compose_dedupe_key(_req(sources=[b, a], target=target), env_id=env)
    assert k1 == k2


def test_dedupe_key_invariant_under_tag_order() -> None:
    sources = _two_sources()
    env = _env_id()
    k1 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(tags=["a", "b"])), env_id=env
    )
    k2 = _compute_compose_dedupe_key(
        _req(sources=sources, target=_target(tags=["b", "a"])), env_id=env
    )
    assert k1 == k2


# ---------------------------------------------------------------------------
# Override path
# ---------------------------------------------------------------------------


def test_dedupe_key_caller_override_returned_verbatim() -> None:
    sources = _two_sources()
    target = _target()
    env = _env_id()
    custom = "client-supplied-idempotency-key-xyz"
    assert (
        _compute_compose_dedupe_key(
            _req(sources=sources, target=target, idempotency_key=custom),
            env_id=env,
        )
        == custom
    )


def test_dedupe_key_caller_override_ignores_payload() -> None:
    """When caller passes idempotency_key, payload changes don't matter."""
    env = _env_id()
    custom = "client-id-001"
    k1 = _compute_compose_dedupe_key(
        _req(
            sources=_two_sources(),
            target=_target(body="aaa"),
            idempotency_key=custom,
        ),
        env_id=env,
    )
    k2 = _compute_compose_dedupe_key(
        _req(
            sources=_two_sources(),
            target=_target(body="bbb"),
            idempotency_key=custom,
        ),
        env_id=_env_id(),
    )
    assert k1 == k2 == custom


# ---------------------------------------------------------------------------
# Fields deliberately EXCLUDED from the key (per B1 rubber-duck)
# ---------------------------------------------------------------------------


def test_dedupe_key_unchanged_by_expected_versions() -> None:
    """expected_versions is a precondition, not identity."""
    sources = _two_sources()
    target = _target()
    env = _env_id()
    r1 = MemComposeRequest(source_ids=sources, target=target, mode="merge")
    r2 = MemComposeRequest(
        source_ids=sources,
        target=target,
        mode="merge",
        expected_versions={sources[0]: 1, sources[1]: 1},
    )
    assert _compute_compose_dedupe_key(r1, env_id=env) == _compute_compose_dedupe_key(
        r2, env_id=env
    )


def test_dedupe_key_unchanged_by_trigger_description() -> None:
    """trigger_description is descriptive only, not identity."""
    sources = _two_sources()
    env = _env_id()
    t1 = _target()
    t2 = MemComposeTarget(
        kind=t1.kind,
        body=t1.body,
        trigger_description="auto-fired by foo",
        metadata=t1.metadata,
    )
    r1 = MemComposeRequest(source_ids=sources, target=t1)
    r2 = MemComposeRequest(source_ids=sources, target=t2)
    assert _compute_compose_dedupe_key(r1, env_id=env) == _compute_compose_dedupe_key(
        r2, env_id=env
    )


def test_dedupe_key_unchanged_by_tag_policy() -> None:
    """tag_policy resolves *into* target.tags server-side, so the policy
    itself doesn't enter the key (would double-count once B3d lands)."""
    sources = _two_sources()
    target = _target(tags=["a", "b"])
    env = _env_id()
    r1 = MemComposeRequest(source_ids=sources, target=target, mode="merge")
    r2 = MemComposeRequest(
        source_ids=sources, target=target, mode="merge", tag_policy="union"
    )
    assert _compute_compose_dedupe_key(r1, env_id=env) == _compute_compose_dedupe_key(
        r2, env_id=env
    )
