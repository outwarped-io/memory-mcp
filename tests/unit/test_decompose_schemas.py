"""Schema-level tests for ``memory_mcp_schemas.decompose``.

These cover the request validators only — the transaction body (lock
order, dedupe-key computation, outbox, audit, lineage) lives in
``decomposers.py`` and gets DB-integration coverage in
``tests/integration/`` once the C6 transaction body lands.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from memory_mcp_schemas.decompose import (
    DecomposeLineageRow,
    MemDecomposeChild,
    MemDecomposeRequest,
    MemDecomposeResponse,
)
from memory_mcp_schemas.enums import MemoryKind, MemoryStatus
from memory_mcp_schemas.memories import MemoryResponse


def _good_child(kind: MemoryKind = MemoryKind.fact, body: str = "atom") -> MemDecomposeChild:
    return MemDecomposeChild(kind=kind, body=body)


def _good_children(n: int = 2) -> list[MemDecomposeChild]:
    return [_good_child(body=f"atom-{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_decompose_request_defaults_derive_mode() -> None:
    req = MemDecomposeRequest(source_id=uuid4(), children=_good_children())
    assert req.mode == "derive"
    assert req.expected_version is None
    assert req.idempotency_key is None
    assert req.children[0].tags is None  # distinct from []


def test_decompose_child_empty_tags_vs_none() -> None:
    """``tags=None`` defers to server defaults; ``tags=[]`` is intentionally empty."""
    c_none = MemDecomposeChild(kind=MemoryKind.fact, body="x", tags=None)
    c_empty = MemDecomposeChild(kind=MemoryKind.fact, body="x", tags=[])
    assert c_none.tags is None
    assert c_empty.tags == []


# ---------------------------------------------------------------------------
# Children cardinality
# ---------------------------------------------------------------------------


def test_decompose_rejects_single_child() -> None:
    with pytest.raises(ValidationError, match="at least 2"):
        MemDecomposeRequest(source_id=uuid4(), children=[_good_child()])


def test_decompose_rejects_zero_children() -> None:
    with pytest.raises(ValidationError, match="at least 2"):
        MemDecomposeRequest(source_id=uuid4(), children=[])


def test_decompose_rejects_more_than_twenty_children() -> None:
    with pytest.raises(ValidationError, match="at most 20"):
        MemDecomposeRequest(source_id=uuid4(), children=_good_children(21))


def test_decompose_accepts_two_children_boundary() -> None:
    req = MemDecomposeRequest(source_id=uuid4(), children=_good_children(2))
    assert len(req.children) == 2


def test_decompose_accepts_twenty_children_boundary() -> None:
    req = MemDecomposeRequest(source_id=uuid4(), children=_good_children(20))
    assert len(req.children) == 20


# ---------------------------------------------------------------------------
# Mode literal
# ---------------------------------------------------------------------------


def test_decompose_accepts_split_mode() -> None:
    req = MemDecomposeRequest(
        source_id=uuid4(), children=_good_children(), mode="split"
    )
    assert req.mode == "split"


def test_decompose_accepts_derive_mode() -> None:
    req = MemDecomposeRequest(
        source_id=uuid4(), children=_good_children(), mode="derive"
    )
    assert req.mode == "derive"


def test_decompose_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        MemDecomposeRequest(
            source_id=uuid4(), children=_good_children(), mode="frobnicate"
        )


# ---------------------------------------------------------------------------
# Extra-field rejection
# ---------------------------------------------------------------------------


def test_decompose_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="(?i)extra"):
        MemDecomposeRequest.model_validate(
            {
                "source_id": str(uuid4()),
                "children": [
                    {"kind": "fact", "body": "a"},
                    {"kind": "fact", "body": "b"},
                ],
                "rogue_field": "x",
            }
        )


def test_decompose_child_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="(?i)extra"):
        MemDecomposeChild.model_validate(
            {"kind": "fact", "body": "x", "rogue_field": "x"}
        )


# ---------------------------------------------------------------------------
# Idempotency-key handling
# ---------------------------------------------------------------------------


def test_decompose_idempotency_key_length_cap() -> None:
    with pytest.raises(ValidationError):
        MemDecomposeRequest(
            source_id=uuid4(),
            children=_good_children(),
            idempotency_key="x" * 129,
        )


def test_decompose_idempotency_key_accepted() -> None:
    req = MemDecomposeRequest(
        source_id=uuid4(),
        children=_good_children(),
        idempotency_key="x" * 128,
    )
    assert req.idempotency_key == "x" * 128


# ---------------------------------------------------------------------------
# expected_version validation
# ---------------------------------------------------------------------------


def test_decompose_expected_version_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        MemDecomposeRequest(
            source_id=uuid4(), children=_good_children(), expected_version=-1
        )


def test_decompose_expected_version_zero_accepted() -> None:
    req = MemDecomposeRequest(
        source_id=uuid4(), children=_good_children(), expected_version=0
    )
    assert req.expected_version == 0


# ---------------------------------------------------------------------------
# Per-child validators
# ---------------------------------------------------------------------------


def test_decompose_child_kind_playbook_rejected() -> None:
    with pytest.raises(ValidationError, match="(?i)playbook"):
        MemDecomposeChild(kind=MemoryKind.playbook, body="x")


def test_decompose_child_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        MemDecomposeChild(kind=MemoryKind.fact, body="x", confidence=1.5)
    ok = MemDecomposeChild(kind=MemoryKind.fact, body="x", confidence=0.5)
    assert ok.confidence == 0.5


def test_decompose_child_salience_bounds() -> None:
    with pytest.raises(ValidationError):
        MemDecomposeChild(kind=MemoryKind.fact, body="x", salience=-0.1)
    ok = MemDecomposeChild(kind=MemoryKind.fact, body="x", salience=0.5)
    assert ok.salience == 0.5


def test_decompose_child_title_length_cap() -> None:
    with pytest.raises(ValidationError):
        MemDecomposeChild(kind=MemoryKind.fact, title="t" * 401, body="x")
    ok = MemDecomposeChild(kind=MemoryKind.fact, title="t" * 400, body="x")
    assert ok.title is not None and len(ok.title) == 400


def test_decompose_child_body_min_length() -> None:
    with pytest.raises(ValidationError):
        MemDecomposeChild(kind=MemoryKind.fact, body="")


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


def _make_memory_response(body: str = "x") -> MemoryResponse:
    return MemoryResponse(
        id=uuid4(),
        env_id=uuid4(),
        kind=MemoryKind.fact,
        status=MemoryStatus.active,
        title="t",
        body=body,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def test_decompose_response_round_trip() -> None:
    source = _make_memory_response("source body")
    children = [_make_memory_response(f"child-{i}") for i in range(2)]
    op_id = uuid4()
    resp = MemDecomposeResponse(
        source=source,
        children=children,
        mode="derive",
        lineage_rows=[
            DecomposeLineageRow(
                parent_memory_id=source.id,
                child_memory_id=children[0].id,
                relation="derived_from",
            ),
            DecomposeLineageRow(
                parent_memory_id=source.id,
                child_memory_id=children[1].id,
                relation="derived_from",
            ),
        ],
        auto_wired=[],
        idempotency_replay=False,
        dedupe_key="a" * 32,
        operation_id=op_id,
    )
    dumped = resp.model_dump()
    revived = MemDecomposeResponse.model_validate(dumped)
    assert revived.mode == "derive"
    assert len(revived.children) == 2
    assert all(r.relation == "derived_from" for r in revived.lineage_rows)
    assert revived.operation_id == op_id


def test_decompose_response_idempotency_replay_default_false() -> None:
    resp = MemDecomposeResponse(
        source=_make_memory_response(),
        children=[_make_memory_response(), _make_memory_response()],
        mode="split",
        lineage_rows=[],
        dedupe_key="d" * 32,
        operation_id=uuid4(),
    )
    assert resp.idempotency_replay is False
    assert resp.auto_wired == []


def test_decompose_lineage_row_rejects_unknown_relation() -> None:
    with pytest.raises(ValidationError):
        DecomposeLineageRow(
            parent_memory_id=uuid4(),
            child_memory_id=uuid4(),
            relation="supersedes",  # not in decompose's allowed set
        )


def test_decompose_lineage_row_rejects_promoted_from() -> None:
    # promoted_from belongs to compose, not decompose.
    with pytest.raises(ValidationError):
        DecomposeLineageRow(
            parent_memory_id=uuid4(),
            child_memory_id=uuid4(),
            relation="promoted_from",
        )


# ---------------------------------------------------------------------------
# v0.16 — per-child auto-wire mapping (Stage H1)
# ---------------------------------------------------------------------------


def test_decompose_response_auto_wired_by_child_defaults_to_none() -> None:
    """Default is ``None`` so callers can distinguish feature OFF
    (None) from feature ON-but-empty (``{}`` or ``{child: []}``)."""
    resp = MemDecomposeResponse(
        source=_make_memory_response(),
        children=[_make_memory_response(), _make_memory_response()],
        mode="derive",
        lineage_rows=[],
        dedupe_key="d" * 32,
        operation_id=uuid4(),
    )
    assert resp.auto_wired_by_child is None
    assert resp.auto_wired == []


def test_decompose_response_auto_wired_by_child_accepts_empty_dict() -> None:
    """Feature ON, no children produced candidates → ``{}``."""
    resp = MemDecomposeResponse(
        source=_make_memory_response(),
        children=[_make_memory_response(), _make_memory_response()],
        mode="derive",
        lineage_rows=[],
        auto_wired_by_child={},
        dedupe_key="d" * 32,
        operation_id=uuid4(),
    )
    assert resp.auto_wired_by_child == {}


def test_decompose_response_auto_wired_by_child_accepts_populated_mapping() -> None:
    child_a = uuid4()
    child_b = uuid4()
    dst_1 = uuid4()
    dst_2 = uuid4()
    dst_3 = uuid4()
    payload = {
        child_a: [dst_1, dst_2],
        child_b: [dst_3],
    }
    resp = MemDecomposeResponse(
        source=_make_memory_response(),
        children=[_make_memory_response(), _make_memory_response()],
        mode="derive",
        lineage_rows=[],
        auto_wired=[dst_1, dst_2, dst_3],
        auto_wired_by_child=payload,
        dedupe_key="d" * 32,
        operation_id=uuid4(),
    )
    assert resp.auto_wired_by_child == payload
    revived = MemDecomposeResponse.model_validate(resp.model_dump())
    assert revived.auto_wired_by_child == payload
    assert set(revived.auto_wired) == {dst_1, dst_2, dst_3}


def test_decompose_response_auto_wired_by_child_rejects_non_uuid_keys() -> None:
    """Strict-validate UUID keys so misuse surfaces at the boundary."""
    with pytest.raises(ValidationError):
        MemDecomposeResponse(
            source=_make_memory_response(),
            children=[_make_memory_response(), _make_memory_response()],
            mode="derive",
            lineage_rows=[],
            auto_wired_by_child={"not-a-uuid": [uuid4()]},
            dedupe_key="d" * 32,
            operation_id=uuid4(),
        )
