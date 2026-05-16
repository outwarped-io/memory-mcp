"""Identity-default payload merging tests."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from memory_mcp_client import MemoryClient
from tests.conftest import make_memory_payload


pytestmark = pytest.mark.asyncio


async def test_no_identity_no_passthrough(client, fake_session) -> None:
    fake_session.set_response("mem_get", make_memory_payload())

    await client.memories.get(uuid4())

    _, payload = fake_session.calls[0]
    assert "agent_id" not in payload
    assert "attached_env_ids" not in payload
    assert "attached_env_names" not in payload


async def test_client_level_agent_id_merged(client_with_identity, fake_session) -> None:
    fake_session.set_response("mem_get", make_memory_payload())

    await client_with_identity.memories.get(uuid4())

    _, payload = fake_session.calls[0]
    assert payload["agent_id"] == "00000000-0000-0000-0000-000000000001"


async def test_client_level_env_ids_merged(client_with_identity, fake_session) -> None:
    fake_session.set_response("mem_get", make_memory_payload())

    await client_with_identity.memories.get(uuid4())

    _, payload = fake_session.calls[0]
    assert payload["attached_env_ids"] == ["00000000-0000-0000-0000-000000000002"]


async def test_client_level_env_names_merged(session_factory, fake_session) -> None:
    client = MemoryClient(
        "http://fake.local/mcp",
        session_factory=session_factory,
        default_env_names=["cdp"],
    )
    fake_session.set_response("mem_get", make_memory_payload())

    async with client:
        await client.memories.get(uuid4())

    _, payload = fake_session.calls[0]
    assert payload["attached_env_names"] == ["cdp"]
    assert "attached_env_ids" not in payload


async def test_client_constructor_accepts_attached_env_names_alias(session_factory, fake_session) -> None:
    client = MemoryClient(
        "http://fake.local/mcp",
        session_factory=session_factory,
        attached_env_names=["cdp"],
    )
    fake_session.set_response("mem_get", make_memory_payload())

    async with client:
        await client.memories.get(uuid4())

    _, payload = fake_session.calls[0]
    assert payload["attached_env_names"] == ["cdp"]


async def test_default_env_ids_and_names_are_mutually_exclusive(session_factory) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        MemoryClient(
            "http://fake.local/mcp",
            session_factory=session_factory,
            default_env_ids=["00000000-0000-0000-0000-000000000002"],
            default_env_names=["cdp"],
        )


async def test_per_call_override_takes_precedence(client_with_identity, fake_session) -> None:
    agent_id = str(uuid4())
    fake_session.set_response("mem_get", make_memory_payload())

    await client_with_identity.memories.get(uuid4(), agent_id=agent_id)

    _, payload = fake_session.calls[0]
    assert payload["agent_id"] == agent_id


async def test_per_call_env_ids_override(client_with_identity, fake_session) -> None:
    env_ids = [str(uuid4()), str(uuid4())]
    fake_session.set_response("mem_get", make_memory_payload())

    await client_with_identity.memories.get(uuid4(), attached_env_ids=env_ids)

    _, payload = fake_session.calls[0]
    assert payload["attached_env_ids"] == env_ids


async def test_uuid_objects_serialize_as_strings(client, fake_session) -> None:
    agent_id = UUID("00000000-0000-0000-0000-000000000011")
    fake_session.set_response("mem_get", make_memory_payload())

    await client.memories.get(uuid4(), agent_id=agent_id)

    _, payload = fake_session.calls[0]
    assert payload["agent_id"] == str(agent_id)


async def test_list_of_uuid_objects_serialize(client, fake_session) -> None:
    env_ids = [
        UUID("00000000-0000-0000-0000-000000000012"),
        UUID("00000000-0000-0000-0000-000000000013"),
    ]
    fake_session.set_response("mem_get", make_memory_payload())

    await client.memories.get(uuid4(), attached_env_ids=env_ids)

    _, payload = fake_session.calls[0]
    assert payload["attached_env_ids"] == [str(env_id) for env_id in env_ids]


async def test_empty_env_list_default_not_injected(session_factory, fake_session) -> None:
    c = MemoryClient(
        "http://fake.local/mcp",
        session_factory=session_factory,
        default_env_ids=[],
    )
    fake_session.set_response("mem_get", make_memory_payload())

    async with c:
        await c.memories.get(uuid4())

    _, payload = fake_session.calls[0]
    assert "attached_env_ids" not in payload
