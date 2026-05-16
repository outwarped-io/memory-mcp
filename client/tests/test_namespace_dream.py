"""Happy-path tests for the dream client namespace."""

from __future__ import annotations

from uuid import uuid4

import pytest

from memory_mcp_schemas.dream import (
    DreamProposalsListRequest,
    DreamProposalsListResponse,
    DreamReviewRequest,
    DreamReviewResponse,
    DreamRunRequest,
    DreamRunResponse,
    DreamStatusRequest,
    DreamStatusResponse,
)


pytestmark = pytest.mark.asyncio


def _proposal_payload(**overrides):
    base = {
        "id": "00000000-0000-0000-0000-00000000d001",
        "env_id": "00000000-0000-0000-0000-00000000e001",
        "kind": "promotion_candidate",
        "status": "open",
        "summarizer_kind": "template",
        "llm_failed": False,
        "payload": {},
        "dream_run_id": None,
        "created_at": "2026-05-13T00:00:00Z",
        "updated_at": "2026-05-13T00:00:00Z",
        "reviewed_at": None,
        "reviewed_by_agent_id": None,
        "review_action": None,
        "review_notes": None,
    }
    base.update(overrides)
    return base


async def test_run(client, fake_session) -> None:
    request = DreamRunRequest()
    fake_session.set_response("dream_run_", {})

    out = await client.dream.run(request)

    assert fake_session.calls == [
        ("dream_run_", {"request": request.model_dump(mode="json")})
    ]
    assert isinstance(out, DreamRunResponse)


async def test_status(client, fake_session) -> None:
    request = DreamStatusRequest(runs_per_mode=1)
    fake_session.set_response(
        "dream_status_",
        {
            "last_runs": [],
            "open_proposal_counts": {},
            "summarizer_kind": "template",
            "llm_backend": "disabled",
            "llm_status": {},
            "heartbeats": [],
        },
    )

    out = await client.dream.status(request)

    assert fake_session.calls == [
        ("dream_status_", {"request": request.model_dump(mode="json")})
    ]
    assert isinstance(out, DreamStatusResponse)


async def test_proposals_list(client, fake_session) -> None:
    request = DreamProposalsListRequest(limit=5)
    fake_session.set_response(
        "dream_proposals_list_",
        {
            "items": [],
            "next_cursor": None,
        },
    )

    out = await client.dream.proposals_list(request)

    assert fake_session.calls == [
        ("dream_proposals_list_", {"request": request.model_dump(mode="json")})
    ]
    assert isinstance(out, DreamProposalsListResponse)


async def test_review(client, fake_session) -> None:
    request = DreamReviewRequest(proposal_id=uuid4(), action="reject")
    fake_session.set_response(
        "dream_review_",
        {
            "proposal": _proposal_payload(id=str(request.proposal_id)),
            "accepted_memory": None,
            "superseded_memory_ids": [],
        },
    )

    out = await client.dream.review(request)

    assert fake_session.calls == [
        ("dream_review_", {"request": request.model_dump(mode="json")})
    ]
    assert isinstance(out, DreamReviewResponse)
