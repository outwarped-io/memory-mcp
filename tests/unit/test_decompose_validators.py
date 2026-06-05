"""Unit tests for ``memory_mcp.decomposers`` validators (Phase 3 C5).

Pure-function tests for the two validators that the transaction body
(C6) will gate on:

* ``_validate_children`` — pre-lock envelope: duplicate-child detection
  and ``decision_meta`` shape against ``kind``.
* ``_validate_source`` — post-lock: env visibility (always),
  ``kind != playbook`` (always), status ∈ {active, stale} (first-write
  only), ``expected_version`` match (first-write only).

The Pydantic schema already covers cardinality (2..20), per-field
constraints (body min_length, title max_length, salience/confidence
range), and the ``kind != playbook`` per-child rule. Those are
re-verified at the schema layer in ``test_decompose_schemas.py``.

Integration-level tests against a real source row land in C7/C9 as
testcontainer-backed integration tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.decompose import MemDecomposeChild, MemDecomposeRequest

from memory_mcp.db.types import MemoryKind, MemoryStatus
from memory_mcp.decomposers import (
    _validate_children,
    _validate_source,
)
from memory_mcp.errors import (
    InvalidInputError,
    InvalidTransitionError,
    NotFoundError,
    VersionConflictError,
)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _child(
    *,
    kind: MemoryKind = MemoryKind.fact,
    body: str = "atomic fact",
    title: str | None = None,
    tags: list[str] | None = None,
    decision_meta: dict | None = None,
) -> MemDecomposeChild:
    return MemDecomposeChild(
        kind=kind,
        title=title,
        body=body,
        tags=tags,
        decision_meta=decision_meta,
    )


def _req(
    *,
    children: list[MemDecomposeChild] | None = None,
    mode: str = "derive",
    expected_version: int | None = None,
    idempotency_key: str | None = None,
) -> MemDecomposeRequest:
    if children is None:
        children = [_child(body="a"), _child(body="b")]
    return MemDecomposeRequest(
        source_id=uuid4(),
        children=children,
        mode=mode,  # type: ignore[arg-type]
        expected_version=expected_version,
        idempotency_key=idempotency_key,
    )


def _src(
    *,
    env_id: UUID | None = None,
    kind: str = MemoryKind.fact.value,
    status: str = MemoryStatus.active.value,
    version: int = 1,
):
    """Return a duck-typed ``Memory`` stand-in.

    ``_validate_source`` only reads ``.env_id`` / ``.kind`` / ``.status`` /
    ``.version`` / ``.id``, so a SimpleNamespace mirrors the ORM rowfaithfully
    without needing a session.
    """
    return SimpleNamespace(
        id=uuid4(),
        env_id=env_id or uuid4(),
        kind=kind,
        status=status,
        version=version,
    )


def _ctx(*, attached: list[UUID] | None = None):
    """Return a duck-typed ``AgentContext`` stand-in.

    ``_validate_source`` only reads ``ctx.attached_env_ids``.
    """
    return SimpleNamespace(attached_env_ids=attached or [])


# ===========================================================================
# _validate_children
# ===========================================================================


# ---------------------------------------------------------------------------
# decision_meta gating
# ---------------------------------------------------------------------------


def test_validate_children_rejects_decision_meta_on_fact_child() -> None:
    req = _req(
        children=[
            _child(body="a", kind=MemoryKind.fact, decision_meta={"choice": "X"}),
            _child(body="b"),
        ]
    )
    with pytest.raises(InvalidInputError, match="decision_meta only valid for kind=decision"):
        _validate_children(req)


def test_validate_children_rejects_decision_meta_on_procedure_child() -> None:
    req = _req(
        children=[
            _child(body="a"),
            _child(body="b", kind=MemoryKind.procedure, decision_meta={"choice": "X"}),
        ]
    )
    with pytest.raises(InvalidInputError, match=r"children\[1\]"):
        _validate_children(req)


def test_validate_children_accepts_decision_meta_on_decision_child() -> None:
    req = _req(
        children=[
            _child(body="a"),
            _child(body="b", kind=MemoryKind.decision, decision_meta={"choice": "X"}),
        ]
    )
    _validate_children(req)  # no raise


def test_validate_children_accepts_none_decision_meta_everywhere() -> None:
    """Baseline: every kind without decision_meta passes."""
    req = _req(
        children=[
            _child(body="a", kind=MemoryKind.fact),
            _child(body="b", kind=MemoryKind.procedure),
        ]
    )
    _validate_children(req)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_validate_children_rejects_two_identical_children() -> None:
    """Same kind + same body + same tags + same metadata → duplicate."""
    c = _child(body="a", tags=["t1"])
    req = _req(children=[c, _child(body="a", tags=["t1"])])
    with pytest.raises(InvalidInputError, match="duplicate child content"):
        _validate_children(req)


def test_validate_children_rejects_three_with_one_duplicate_pair() -> None:
    """A 3-child request where children[2] dupes children[0]."""
    c0 = _child(body="a", tags=["x"])
    c1 = _child(body="b", tags=["y"])
    c2 = _child(body="a", tags=["x"])
    req = _req(children=[c0, c1, c2])
    with pytest.raises(InvalidInputError, match=r"children\[2\]"):
        _validate_children(req)


def test_validate_children_accepts_two_children_with_different_body() -> None:
    req = _req(children=[_child(body="a"), _child(body="b")])
    _validate_children(req)


def test_validate_children_accepts_two_children_with_different_tags() -> None:
    req = _req(children=[_child(body="a", tags=["x"]), _child(body="a", tags=["y"])])
    _validate_children(req)


def test_validate_children_accepts_two_children_with_different_kind() -> None:
    req = _req(
        children=[
            _child(body="a", kind=MemoryKind.fact),
            _child(body="a", kind=MemoryKind.procedure),
        ]
    )
    _validate_children(req)


def test_validate_children_duplicate_ignores_trigger_description() -> None:
    """``trigger_description`` is not in the dedupe canonical payload —
    two children that differ only in trigger_description ARE duplicates."""
    c0 = MemDecomposeChild(kind=MemoryKind.fact, body="a", trigger_description="when X")
    c1 = MemDecomposeChild(kind=MemoryKind.fact, body="a", trigger_description="when Y")
    req = _req(children=[c0, c1])
    with pytest.raises(InvalidInputError, match="duplicate child content"):
        _validate_children(req)


# ===========================================================================
# _validate_source
# ===========================================================================


# ---------------------------------------------------------------------------
# Env visibility (enforced on BOTH first call and replay)
# ---------------------------------------------------------------------------


def test_validate_source_no_attached_envs_skips_visibility() -> None:
    """Empty attached set means 'tool has no env scoping' — allow."""
    src = _src()
    ctx = _ctx(attached=[])
    _validate_source(src, _req(), ctx, is_replay=False)


def test_validate_source_attached_envs_match_passes() -> None:
    env = uuid4()
    src = _src(env_id=env)
    ctx = _ctx(attached=[env])
    _validate_source(src, _req(), ctx, is_replay=False)


def test_validate_source_attached_envs_mismatch_raises_notfound() -> None:
    src = _src(env_id=uuid4())
    ctx = _ctx(attached=[uuid4()])
    with pytest.raises(NotFoundError, match="not visible in attached envs"):
        _validate_source(src, _req(), ctx, is_replay=False)


def test_validate_source_env_check_enforced_on_replay_too() -> None:
    """Even on replay, an external caller can't fish for op rows in
    envs they don't see."""
    src = _src(env_id=uuid4())
    ctx = _ctx(attached=[uuid4()])
    with pytest.raises(NotFoundError):
        _validate_source(src, _req(), ctx, is_replay=True)


# ---------------------------------------------------------------------------
# Playbook source rejection (enforced on BOTH paths)
# ---------------------------------------------------------------------------


def test_validate_source_rejects_playbook_kind_first_call() -> None:
    src = _src(kind=MemoryKind.playbook.value)
    with pytest.raises(InvalidInputError, match="cannot decompose a playbook source"):
        _validate_source(src, _req(), _ctx(), is_replay=False)


def test_validate_source_rejects_playbook_kind_on_replay() -> None:
    """Replay still surfaces the structural error — playbook decompose
    never made sense; the original call should have failed too."""
    src = _src(kind=MemoryKind.playbook.value)
    with pytest.raises(InvalidInputError):
        _validate_source(src, _req(), _ctx(), is_replay=True)


# ---------------------------------------------------------------------------
# Status gating (first-call ONLY)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [MemoryStatus.active.value, MemoryStatus.stale.value],
)
def test_validate_source_accepts_active_and_stale(status: str) -> None:
    src = _src(status=status)
    _validate_source(src, _req(), _ctx(), is_replay=False)


@pytest.mark.parametrize(
    "status",
    [
        MemoryStatus.proposed.value,
        MemoryStatus.retired.value,
        MemoryStatus.archived.value,
        MemoryStatus.superseded.value,
    ],
)
def test_validate_source_rejects_unacceptable_status_on_first_call(status: str) -> None:
    src = _src(status=status)
    with pytest.raises(InvalidTransitionError) as exc_info:
        _validate_source(src, _req(), _ctx(), is_replay=False)
    assert exc_info.value.src == status
    assert exc_info.value.dst == "decomposed"


@pytest.mark.parametrize(
    "status",
    [
        MemoryStatus.proposed.value,
        MemoryStatus.retired.value,
        MemoryStatus.archived.value,
        MemoryStatus.superseded.value,
    ],
)
def test_validate_source_status_check_skipped_on_replay(status: str) -> None:
    """Per RD A.2: replay survives source retirement. Status checks
    must NOT fire on replay."""
    src = _src(status=status)
    _validate_source(src, _req(), _ctx(), is_replay=True)


# ---------------------------------------------------------------------------
# expected_version gating (first-call ONLY)
# ---------------------------------------------------------------------------


def test_validate_source_accepts_matching_expected_version() -> None:
    src = _src(version=5)
    _validate_source(src, _req(expected_version=5), _ctx(), is_replay=False)


def test_validate_source_accepts_no_expected_version() -> None:
    src = _src(version=5)
    _validate_source(src, _req(expected_version=None), _ctx(), is_replay=False)


def test_validate_source_rejects_mismatched_expected_version() -> None:
    src = _src(version=5)
    with pytest.raises(VersionConflictError) as exc_info:
        _validate_source(src, _req(expected_version=3), _ctx(), is_replay=False)
    assert exc_info.value.expected == 3
    assert exc_info.value.actual == 5


def test_validate_source_expected_version_check_skipped_on_replay() -> None:
    """The precondition gates the original mutating call only. On replay
    the source's version may have moved on; the call still succeeds."""
    src = _src(version=99)
    _validate_source(src, _req(expected_version=1), _ctx(), is_replay=True)
