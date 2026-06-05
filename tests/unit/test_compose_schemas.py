"""Schema-level tests for ``memory_mcp_schemas.compose``.

These cover the request validators only — the transaction body (lock
order, dedupe-key computation, outbox, audit, lineage) lives in
``composers.py`` and gets DB-integration coverage in ``tests/integration/``.
"""

from __future__ import annotations

from datetime import UTC
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.compose import (
    ComposeLineageRow,
    MemComposeRequest,
    MemComposeResponse,
    MemComposeTarget,
)
from memory_mcp_schemas.enums import MemoryKind, MemoryStatus
from memory_mcp_schemas.memories import MemoryResponse
from pydantic import ValidationError


def _good_target() -> MemComposeTarget:
    return MemComposeTarget(kind=MemoryKind.fact, body="combined body")


def _good_sources() -> list[UUID]:
    return [uuid4(), uuid4()]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_compose_request_defaults_promote_no_tag_policy() -> None:
    req = MemComposeRequest(source_ids=_good_sources(), target=_good_target())
    assert req.mode == "promote"
    assert req.tag_policy is None  # server resolves per-mode default
    assert req.idempotency_key is None
    assert req.expected_versions is None
    assert req.target.tags is None  # distinct from []


def test_compose_target_empty_tags_vs_none() -> None:
    """``tags=None`` defers to policy; ``tags=[]`` is intentionally empty."""
    t_none = MemComposeTarget(kind=MemoryKind.fact, body="x", tags=None)
    t_empty = MemComposeTarget(kind=MemoryKind.fact, body="x", tags=[])
    assert t_none.tags is None
    assert t_empty.tags == []


# ---------------------------------------------------------------------------
# Source-id validation
# ---------------------------------------------------------------------------


def test_compose_rejects_single_source() -> None:
    with pytest.raises(ValidationError, match="at least 2"):
        MemComposeRequest(source_ids=[uuid4()], target=_good_target())


def test_compose_rejects_more_than_twenty_sources() -> None:
    with pytest.raises(ValidationError, match="at most 20"):
        MemComposeRequest(source_ids=[uuid4() for _ in range(21)], target=_good_target())


def test_compose_rejects_duplicate_source_ids() -> None:
    shared = uuid4()
    with pytest.raises(ValidationError, match="duplicates"):
        MemComposeRequest(source_ids=[shared, shared], target=_good_target())


# ---------------------------------------------------------------------------
# Mode + tag-policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["promote", "merge"])
def test_compose_accepts_both_modes(mode: str) -> None:
    req = MemComposeRequest(source_ids=_good_sources(), target=_good_target(), mode=mode)
    assert req.mode == mode


def test_compose_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        MemComposeRequest(source_ids=_good_sources(), target=_good_target(), mode="frobnicate")


@pytest.mark.parametrize("policy", ["target", "union", "target_plus_union"])
def test_compose_accepts_valid_tag_policy(policy: str) -> None:
    req = MemComposeRequest(source_ids=_good_sources(), target=_good_target(), tag_policy=policy)
    assert req.tag_policy == policy


def test_compose_rejects_unknown_tag_policy() -> None:
    with pytest.raises(ValidationError):
        MemComposeRequest(source_ids=_good_sources(), target=_good_target(), tag_policy="exclusive")


# ---------------------------------------------------------------------------
# Expected versions
# ---------------------------------------------------------------------------


def test_compose_expected_versions_must_subset_source_ids() -> None:
    src = _good_sources()
    unknown = uuid4()
    with pytest.raises(ValidationError, match="ids not in source_ids"):
        MemComposeRequest(
            source_ids=src,
            target=_good_target(),
            expected_versions={unknown: 1},
        )


def test_compose_expected_versions_subset_ok() -> None:
    src = _good_sources()
    req = MemComposeRequest(
        source_ids=src,
        target=_good_target(),
        expected_versions={src[0]: 1},  # subset is allowed
    )
    assert req.expected_versions == {src[0]: 1}


# ---------------------------------------------------------------------------
# Env refs (re-uses validate_optional_env_ref_pair from schemas)
# ---------------------------------------------------------------------------


def test_compose_rejects_both_env_id_and_env_name() -> None:
    with pytest.raises(ValidationError):
        MemComposeRequest(
            source_ids=_good_sources(),
            target=_good_target(),
            env_id=uuid4(),
            env_name="foo",
        )


def test_compose_accepts_env_id_only() -> None:
    e = uuid4()
    req = MemComposeRequest(source_ids=_good_sources(), target=_good_target(), env_id=e)
    assert req.env_id == e


def test_compose_accepts_env_name_only() -> None:
    req = MemComposeRequest(source_ids=_good_sources(), target=_good_target(), env_name="cdp")
    assert req.env_name == "cdp"


# ---------------------------------------------------------------------------
# Extra-field forbidden
# ---------------------------------------------------------------------------


def test_compose_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MemComposeRequest(source_ids=_good_sources(), target=_good_target(), unknown_field=1)


def test_compose_target_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        # ``env_id`` is one of the fields explicitly kept off the target so
        # callers can't tunnel a destination env through ``target``.
        MemComposeTarget(kind=MemoryKind.fact, body="x", env_id=uuid4())


# ---------------------------------------------------------------------------
# Idempotency-key length cap
# ---------------------------------------------------------------------------


def test_compose_idempotency_key_length_cap() -> None:
    with pytest.raises(ValidationError):
        MemComposeRequest(
            source_ids=_good_sources(),
            target=_good_target(),
            idempotency_key="x" * 129,
        )


def test_compose_idempotency_key_accepted() -> None:
    req = MemComposeRequest(
        source_ids=_good_sources(),
        target=_good_target(),
        idempotency_key="caller-supplied-handle-001",
    )
    assert req.idempotency_key == "caller-supplied-handle-001"


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


def _make_memory_response() -> MemoryResponse:
    from datetime import datetime

    return MemoryResponse(
        id=uuid4(),
        env_id=uuid4(),
        kind=MemoryKind.fact,
        status=MemoryStatus.active,
        title="combined",
        body="combined body",
        tags=[],
        metadata={},
        salience=0.5,
        confidence=0.7,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def test_compose_response_round_trip() -> None:
    src = _good_sources()
    mem = _make_memory_response()
    resp = MemComposeResponse(
        memory=mem,
        mode="promote",
        source_ids=src,
        lineage_rows=[
            ComposeLineageRow(
                parent_memory_id=src[0],
                child_memory_id=mem.id,
                relation="promoted_from",
            ),
            ComposeLineageRow(
                parent_memory_id=src[1],
                child_memory_id=mem.id,
                relation="promoted_from",
            ),
        ],
        retired_source_ids=[],
        auto_wired=[],
        idempotency_replay=False,
        tag_policy_applied="target",
        dedupe_key="a" * 32,
    )
    dumped = resp.model_dump()
    revived = MemComposeResponse.model_validate(dumped)
    assert revived.mode == "promote"
    assert len(revived.lineage_rows) == 2
    assert all(r.relation == "promoted_from" for r in revived.lineage_rows)


def test_compose_lineage_row_rejects_unknown_relation() -> None:
    with pytest.raises(ValidationError):
        ComposeLineageRow(
            parent_memory_id=uuid4(),
            child_memory_id=uuid4(),
            relation="derives_from",  # not in compose's allowed set
        )
