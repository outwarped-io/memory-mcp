"""Sanity-check that MemoryClient dispatches tool calls via the fake session.

Smoke test that ensures the Phase 2 dispatcher + Phase 3 namespaces +
Phase 4 fake-session conftest all line up end-to-end. The detailed
per-namespace happy-path coverage lives in ``test_namespace_*.py``.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from tests.conftest import make_memory_payload

pytestmark = pytest.mark.asyncio


async def test_client_dispatches_through_fake_session(client, fake_session) -> None:
    """A single tool call lands in the fake session with the right payload."""
    memory_id = str(uuid4())
    response_payload = make_memory_payload(id=memory_id)
    fake_session.set_response("mem_get", response_payload)

    out = await client.memories.get(memory_id)

    # Tool name + payload landed in the fake.
    assert fake_session.calls == [("mem_get", {"memory_id": memory_id})]
    # Response validated into a MemoryResponse Pydantic model.
    assert str(out.id) == memory_id
    assert out.version == 1


async def test_identity_defaults_are_merged(client_with_identity, fake_session) -> None:
    """Client-level agent_id + env_ids merge into every call payload."""
    fake_session.set_response("env_list_", [])

    out = await client_with_identity.envs.list_()
    assert out == []

    # Payload should carry the defaults.
    name, args = fake_session.calls[0]
    assert name == "env_list_"
    assert args["agent_id"] == "00000000-0000-0000-0000-000000000001"
    assert args["attached_env_ids"] == ["00000000-0000-0000-0000-000000000002"]


async def test_call_without_open_raises() -> None:
    """Forgetting ``async with`` should fail loudly, not crash mid-call."""
    from memory_mcp_client import MemoryClient

    c = MemoryClient("http://fake.local/mcp")
    with pytest.raises(RuntimeError, match="must be opened"):
        await c.memories.get("00000000-0000-0000-0000-000000000001")
