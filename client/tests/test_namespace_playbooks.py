"""Happy-path tests for the playbooks client namespace."""

from __future__ import annotations

from uuid import uuid4

import pytest
from memory_mcp_schemas.playbooks import PlaybookInvokeResponse

pytestmark = pytest.mark.asyncio


def _memory_payload(**overrides):
    base = {
        "id": "00000000-0000-0000-0000-00000000cafe",
        "env_id": "00000000-0000-0000-0000-00000000e001",
        "kind": "playbook",
        "status": "active",
        "title": "fake playbook",
        "body": "fake body",
        "trigger_description": None,
        "steps": ["do thing"],
        "macro": "foo",
        "tags": [],
        "metadata": {},
        "salience": 0.5,
        "confidence": 0.9,
        "pinned": False,
        "access_count": 0,
        "last_accessed_at": None,
        "negative_feedback_count": 0,
        "verified_at": None,
        "expires_at": None,
        "superseded_by": None,
        "decision_meta": None,
        "version": 1,
        "created_at": "2026-05-13T00:00:00Z",
        "updated_at": "2026-05-13T00:00:00Z",
    }
    base.update(overrides)
    return base


async def test_invoke(client, fake_session) -> None:
    env_id = uuid4()
    fake_session.set_response(
        "playbook_invoke",
        {
            "playbook": _memory_payload(),
            "steps": ["do thing"],
            "referenced_memories": [],
            "missing_refs": [],
        },
    )

    out = await client.playbooks.invoke(macro="foo", env_id=env_id)

    name, args = fake_session.calls[0]
    assert name == "playbook_invoke"
    assert args["macro"] == "foo"
    assert args["env_id"] == str(env_id)
    assert isinstance(out, PlaybookInvokeResponse)
