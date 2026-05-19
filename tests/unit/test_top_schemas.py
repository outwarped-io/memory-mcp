"""Schema-level unit tests for ``mem_top`` request and response shapes.

Covers the wire-level contract additions introduced in Phase 1e-e:

* ``MemTopBy`` Literal accepts the new ``reference_authority`` value.
* Unknown ``by`` values are rejected by Pydantic.
* ``MemoryResponse.reference_authority`` defaults to ``0.0`` so callers
  not aware of the field still get a well-formed response.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from memory_mcp_schemas.memories import MemoryResponse
from memory_mcp_schemas.top import MemTopRequest


def test_memtop_request_accepts_reference_authority() -> None:
    """``by="reference_authority"`` is a valid metric value."""
    req = MemTopRequest(by="reference_authority")
    assert req.by == "reference_authority"


def test_memtop_request_rejects_unknown_by() -> None:
    """A bogus ``by`` value fails Pydantic validation — the new
    metric must not loosen the closed set.
    """
    with pytest.raises(ValidationError):
        MemTopRequest(by="bogus_metric")


def test_memory_response_reference_authority_default_zero() -> None:
    """``reference_authority`` is additive and defaults to ``0.0`` so
    old callers building ``MemoryResponse`` without the field still
    validate.
    """
    import datetime as dt
    from uuid import uuid4

    resp = MemoryResponse(
        id=uuid4(),
        env_id=uuid4(),
        kind="fact",
        status="active",
        title=None,
        body="x",
        tags=[],
        metadata={},
        salience=0.5,
        confidence=0.5,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        version=1,
        created_at=dt.datetime.now(dt.UTC),
        updated_at=dt.datetime.now(dt.UTC),
    )
    assert resp.reference_authority == 0.0
