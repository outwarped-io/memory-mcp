"""Unit tests for ``memory_mcp.decomposers`` helpers (Phase 3 C4).

Pure-function tests for the three idempotency primitives that the
``mem_decompose`` transaction body (C6) will consume:

* ``_compute_decompose_dedupe_key`` — caller-override passthrough or
  sha256(canonical_json) → 32 hex; children sorted by per-child canonical
  hash for re-ordering invariance.
* ``_compute_request_fingerprint`` — always-canonical envelope hash that
  IS sensitive to ``expected_version`` / ``trigger_description`` /
  ``expires_at`` and IGNORES ``idempotency_key``.
* ``_is_decompose_dedupe_error`` — IntegrityError classifier on
  ``ix_decompose_operations_dedupe``.

Transaction-body tests (race resolution, replay, RBAC) land at C7 as
integration tests in ``tests/integration/test_decompose_transaction.py``.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError

from memory_mcp.decomposers import (
    _canonical_child_payload,
    _compute_decompose_dedupe_key,
    _compute_request_fingerprint,
    _is_decompose_dedupe_error,
)
from memory_mcp_schemas.decompose import MemDecomposeChild, MemDecomposeRequest
from memory_mcp_schemas.enums import MemoryKind


def _env_id() -> UUID:
    return uuid4()


def _source_id() -> UUID:
    return uuid4()


def _child(
    *,
    kind: MemoryKind = MemoryKind.fact,
    body: str = "atomic fact",
    title: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    salience: float | None = None,
    confidence: float | None = None,
    pinned: bool = False,
    decision_meta: dict | None = None,
    trigger_description: str | None = None,
    expires_at: dt.datetime | None = None,
) -> MemDecomposeChild:
    return MemDecomposeChild(
        kind=kind,
        title=title,
        body=body,
        tags=tags,
        metadata=metadata or {},
        salience=salience,
        confidence=confidence,
        pinned=pinned,
        decision_meta=decision_meta,
        trigger_description=trigger_description,
        expires_at=expires_at,
    )


def _req(
    *,
    source_id: UUID | None = None,
    children: list[MemDecomposeChild] | None = None,
    mode: str = "derive",
    idempotency_key: str | None = None,
    expected_version: int | None = None,
) -> MemDecomposeRequest:
    if children is None:
        children = [_child(body="a"), _child(body="b")]
    return MemDecomposeRequest(
        source_id=source_id or _source_id(),
        children=children,
        mode=mode,  # type: ignore[arg-type]
        idempotency_key=idempotency_key,
        expected_version=expected_version,
    )


# ---------------------------------------------------------------------------
# Dedupe-key — determinism, shape, ordering invariance
# ---------------------------------------------------------------------------


def test_decompose_key_deterministic_same_input() -> None:
    src = _source_id()
    env = _env_id()
    children = [_child(body="a"), _child(body="b")]
    r1 = _req(source_id=src, children=list(children))
    r2 = _req(source_id=src, children=list(children))
    assert _compute_decompose_dedupe_key(r1, env_id=env) == _compute_decompose_dedupe_key(
        r2, env_id=env
    )


def test_decompose_key_length_32_hex() -> None:
    key = _compute_decompose_dedupe_key(_req(), env_id=_env_id())
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)


def test_decompose_key_child_order_invariant() -> None:
    """Swapping the two children produces the same key (canonical sort)."""
    src = _source_id()
    env = _env_id()
    c1 = _child(body="a", tags=["t-a"])
    c2 = _child(body="b", tags=["t-b"])
    r_ab = _req(source_id=src, children=[c1, c2])
    r_ba = _req(source_id=src, children=[c2, c1])
    assert _compute_decompose_dedupe_key(r_ab, env_id=env) == _compute_decompose_dedupe_key(
        r_ba, env_id=env
    )


# ---------------------------------------------------------------------------
# Dedupe-key — sensitivity (these MUST differ)
# ---------------------------------------------------------------------------


def test_decompose_key_changes_with_mode() -> None:
    src = _source_id()
    env = _env_id()
    derive = _compute_decompose_dedupe_key(
        _req(source_id=src, mode="derive"), env_id=env
    )
    split = _compute_decompose_dedupe_key(
        _req(source_id=src, mode="split"), env_id=env
    )
    assert derive != split


def test_decompose_key_changes_with_source_id() -> None:
    env = _env_id()
    children = [_child(body="a"), _child(body="b")]
    k1 = _compute_decompose_dedupe_key(
        _req(source_id=_source_id(), children=list(children)), env_id=env
    )
    k2 = _compute_decompose_dedupe_key(
        _req(source_id=_source_id(), children=list(children)), env_id=env
    )
    assert k1 != k2


def test_decompose_key_changes_with_env_id() -> None:
    src = _source_id()
    children = [_child(body="a"), _child(body="b")]
    r = _req(source_id=src, children=children)
    k1 = _compute_decompose_dedupe_key(r, env_id=_env_id())
    k2 = _compute_decompose_dedupe_key(r, env_id=_env_id())
    assert k1 != k2


def test_decompose_key_changes_with_child_content() -> None:
    src = _source_id()
    env = _env_id()
    k1 = _compute_decompose_dedupe_key(
        _req(source_id=src, children=[_child(body="a"), _child(body="b")]),
        env_id=env,
    )
    k2 = _compute_decompose_dedupe_key(
        _req(source_id=src, children=[_child(body="a"), _child(body="b-modified")]),
        env_id=env,
    )
    assert k1 != k2


def test_decompose_key_changes_with_tag_set() -> None:
    src = _source_id()
    env = _env_id()
    k_no_tags = _compute_decompose_dedupe_key(
        _req(source_id=src, children=[_child(body="a", tags=None), _child(body="b")]),
        env_id=env,
    )
    k_with_tag = _compute_decompose_dedupe_key(
        _req(source_id=src, children=[_child(body="a", tags=["x"]), _child(body="b")]),
        env_id=env,
    )
    assert k_no_tags != k_with_tag


# ---------------------------------------------------------------------------
# Dedupe-key — caller override
# ---------------------------------------------------------------------------


def test_decompose_key_idempotency_override_passthrough() -> None:
    """``idempotency_key`` is returned verbatim, bypassing the sha256 path."""
    r = _req(idempotency_key="my-caller-key-v1")
    key = _compute_decompose_dedupe_key(r, env_id=_env_id())
    assert key == "my-caller-key-v1"


def test_decompose_key_override_ignores_payload_differences() -> None:
    """Two requests with same override key + different payloads → same key."""
    env = _env_id()
    r1 = _req(
        source_id=_source_id(),
        children=[_child(body="a"), _child(body="b")],
        mode="derive",
        idempotency_key="shared-key",
    )
    r2 = _req(
        source_id=_source_id(),
        children=[_child(body="z"), _child(body="y"), _child(body="w")],
        mode="split",
        idempotency_key="shared-key",
    )
    assert _compute_decompose_dedupe_key(r1, env_id=env) == _compute_decompose_dedupe_key(
        r2, env_id=env
    ) == "shared-key"


# ---------------------------------------------------------------------------
# Dedupe-key — documented exclusions (these MUST NOT change the key)
# ---------------------------------------------------------------------------


def test_decompose_key_unchanged_by_expected_version() -> None:
    src = _source_id()
    env = _env_id()
    children = [_child(body="a"), _child(body="b")]
    k_none = _compute_decompose_dedupe_key(
        _req(source_id=src, children=list(children), expected_version=None),
        env_id=env,
    )
    k_set = _compute_decompose_dedupe_key(
        _req(source_id=src, children=list(children), expected_version=5),
        env_id=env,
    )
    assert k_none == k_set


def test_decompose_key_unchanged_by_per_child_trigger_description() -> None:
    src = _source_id()
    env = _env_id()
    k_plain = _compute_decompose_dedupe_key(
        _req(source_id=src, children=[_child(body="a"), _child(body="b")]),
        env_id=env,
    )
    k_trig = _compute_decompose_dedupe_key(
        _req(
            source_id=src,
            children=[
                _child(body="a", trigger_description="when X"),
                _child(body="b", trigger_description="when Y"),
            ],
        ),
        env_id=env,
    )
    assert k_plain == k_trig


def test_decompose_key_unchanged_by_per_child_expires_at() -> None:
    src = _source_id()
    env = _env_id()
    future = dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc)
    k_plain = _compute_decompose_dedupe_key(
        _req(source_id=src, children=[_child(body="a"), _child(body="b")]),
        env_id=env,
    )
    k_exp = _compute_decompose_dedupe_key(
        _req(
            source_id=src,
            children=[
                _child(body="a", expires_at=future),
                _child(body="b", expires_at=future),
            ],
        ),
        env_id=env,
    )
    assert k_plain == k_exp


# ---------------------------------------------------------------------------
# Request fingerprint — determinism + sensitivity
# ---------------------------------------------------------------------------


def test_decompose_fp_deterministic_same_input() -> None:
    src = _source_id()
    env = _env_id()
    children = [_child(body="a"), _child(body="b")]
    r1 = _req(source_id=src, children=list(children))
    r2 = _req(source_id=src, children=list(children))
    assert _compute_request_fingerprint(r1, env_id=env) == _compute_request_fingerprint(
        r2, env_id=env
    )


def test_decompose_fp_length_32_hex() -> None:
    fp = _compute_request_fingerprint(_req(), env_id=_env_id())
    assert len(fp) == 32
    assert all(c in "0123456789abcdef" for c in fp)


def test_decompose_fp_changes_with_mode() -> None:
    src = _source_id()
    env = _env_id()
    fp_d = _compute_request_fingerprint(_req(source_id=src, mode="derive"), env_id=env)
    fp_s = _compute_request_fingerprint(_req(source_id=src, mode="split"), env_id=env)
    assert fp_d != fp_s


def test_decompose_fp_changes_with_source_id() -> None:
    env = _env_id()
    children = [_child(body="a"), _child(body="b")]
    fp1 = _compute_request_fingerprint(
        _req(source_id=_source_id(), children=list(children)), env_id=env
    )
    fp2 = _compute_request_fingerprint(
        _req(source_id=_source_id(), children=list(children)), env_id=env
    )
    assert fp1 != fp2


def test_decompose_fp_changes_with_expected_version() -> None:
    """The KEY distinction from the dedupe-key: fingerprint IS sensitive."""
    src = _source_id()
    env = _env_id()
    children = [_child(body="a"), _child(body="b")]
    fp_none = _compute_request_fingerprint(
        _req(source_id=src, children=list(children), expected_version=None),
        env_id=env,
    )
    fp_set = _compute_request_fingerprint(
        _req(source_id=src, children=list(children), expected_version=5),
        env_id=env,
    )
    assert fp_none != fp_set


def test_decompose_fp_changes_with_trigger_description() -> None:
    src = _source_id()
    env = _env_id()
    fp_plain = _compute_request_fingerprint(
        _req(source_id=src, children=[_child(body="a"), _child(body="b")]),
        env_id=env,
    )
    fp_trig = _compute_request_fingerprint(
        _req(
            source_id=src,
            children=[_child(body="a", trigger_description="when X"), _child(body="b")],
        ),
        env_id=env,
    )
    assert fp_plain != fp_trig


def test_decompose_fp_changes_with_expires_at() -> None:
    src = _source_id()
    env = _env_id()
    future = dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc)
    fp_plain = _compute_request_fingerprint(
        _req(source_id=src, children=[_child(body="a"), _child(body="b")]),
        env_id=env,
    )
    fp_exp = _compute_request_fingerprint(
        _req(
            source_id=src,
            children=[_child(body="a", expires_at=future), _child(body="b")],
        ),
        env_id=env,
    )
    assert fp_plain != fp_exp


def test_decompose_fp_ignores_idempotency_key() -> None:
    """Fingerprint reflects request scope; ``idempotency_key`` is out-of-scope."""
    src = _source_id()
    env = _env_id()
    children = [_child(body="a"), _child(body="b")]
    fp_a = _compute_request_fingerprint(
        _req(source_id=src, children=list(children), idempotency_key="key-A"),
        env_id=env,
    )
    fp_b = _compute_request_fingerprint(
        _req(source_id=src, children=list(children), idempotency_key="key-B"),
        env_id=env,
    )
    fp_none = _compute_request_fingerprint(
        _req(source_id=src, children=list(children), idempotency_key=None),
        env_id=env,
    )
    assert fp_a == fp_b == fp_none


# ---------------------------------------------------------------------------
# Cross-checks between dedupe-key and fingerprint
# ---------------------------------------------------------------------------


def test_decompose_key_and_fp_differ_for_same_request() -> None:
    """Domain separator (``mem_decompose`` vs ``mem_decompose_fp``)
    guarantees that the same canonical envelope produces different
    output strings for the two helpers."""
    r = _req()
    env = _env_id()
    assert _compute_decompose_dedupe_key(r, env_id=env) != _compute_request_fingerprint(
        r, env_id=env
    )


def test_decompose_key_caller_override_vs_fp_canonical() -> None:
    """When ``idempotency_key`` is set, the dedupe-key takes the override
    path while the fingerprint stays canonical sha256."""
    r = _req(idempotency_key="my-key")
    env = _env_id()
    key = _compute_decompose_dedupe_key(r, env_id=env)
    fp = _compute_request_fingerprint(r, env_id=env)
    assert key == "my-key"
    assert len(fp) == 32
    assert all(c in "0123456789abcdef" for c in fp)
    assert key != fp


# ---------------------------------------------------------------------------
# IntegrityError classifier
# ---------------------------------------------------------------------------


def test_is_decompose_dedupe_error_matches_named_constraint() -> None:
    """``orig.constraint_name`` carries the unique-index name → True."""
    orig = SimpleNamespace(
        constraint_name="ix_decompose_operations_dedupe",
        diag=SimpleNamespace(constraint_name=None),
    )
    exc = IntegrityError("INSERT INTO decompose_operations ...", {}, orig)
    assert _is_decompose_dedupe_error(exc) is True


def test_is_decompose_dedupe_error_via_diag_constraint_name() -> None:
    """Older psycopg shapes carry the constraint name on ``orig.diag`` only."""
    orig = SimpleNamespace(
        constraint_name=None,
        diag=SimpleNamespace(constraint_name="ix_decompose_operations_dedupe"),
    )
    exc = IntegrityError("INSERT INTO decompose_operations ...", {}, orig)
    assert _is_decompose_dedupe_error(exc) is True


def test_is_decompose_dedupe_error_substring_fallback() -> None:
    """When ``orig`` is missing both attributes, fall back to the rendered string."""
    exc = IntegrityError(
        'duplicate key value violates unique constraint "ix_decompose_operations_dedupe"',
        {},
        Exception("synthetic"),
    )
    assert _is_decompose_dedupe_error(exc) is True


def test_is_decompose_dedupe_error_rejects_other_constraint() -> None:
    orig = SimpleNamespace(
        constraint_name="ix_memories_compose_dedupe",
        diag=SimpleNamespace(constraint_name="ix_memories_compose_dedupe"),
    )
    exc = IntegrityError("INSERT INTO memories ...", {}, orig)
    assert _is_decompose_dedupe_error(exc) is False


# ---------------------------------------------------------------------------
# Canonical-child helper sanity (exercised indirectly above; one direct test
# to lock the documented field set)
# ---------------------------------------------------------------------------


def test_canonical_child_payload_field_set() -> None:
    """The dedupe-key canonical child shape must NOT include
    ``trigger_description`` or ``expires_at`` (those are fingerprint-only)."""
    payload = _canonical_child_payload(
        _child(
            body="x",
            trigger_description="when Y",
            expires_at=dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc),
        )
    )
    assert "trigger_description" not in payload
    assert "expires_at" not in payload
    assert payload["body"] == "x"
    assert payload["kind"] == "fact"
