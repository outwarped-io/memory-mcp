"""Happy-path tests for the envs client namespace."""

from __future__ import annotations

from uuid import uuid4

import pytest

from memory_mcp_schemas.envs import (
    AttachedEnvsResponse,
    EnvCreateRequest,
    EnvResponse,
)
from tests.conftest import make_env_payload


pytestmark = pytest.mark.asyncio


def _attached_payload(session_id, **env_overrides):
    return {
        "session_id": str(session_id),
        "attached": [make_env_payload(**env_overrides)],
    }


async def test_create(client, fake_session) -> None:
    request = EnvCreateRequest(name="foo", default_embedding_model_id="model-1")
    fake_session.set_response("env_create_", make_env_payload(name="foo"))

    out = await client.envs.create(request)

    assert fake_session.calls == [
        ("env_create_", {"request": request.model_dump(mode="json")})
    ]
    assert isinstance(out, EnvResponse)
    assert out.name == "foo"


async def test_list_returns_list(client, fake_session) -> None:
    fake_session.set_response(
        "env_list_",
        [make_env_payload(), make_env_payload(id="00000000-0000-0000-0000-0000000000e1", name="other")],
    )

    out = await client.envs.list_()

    assert fake_session.calls == [("env_list_", {})]
    assert len(out) == 2
    assert all(isinstance(env, EnvResponse) for env in out)


async def test_list_empty(client, fake_session) -> None:
    fake_session.set_response("env_list_", [])

    out = await client.envs.list_()

    assert fake_session.calls == [("env_list_", {})]
    assert out == []


async def test_get_by_name(client, fake_session) -> None:
    fake_session.set_response("env_get_", make_env_payload(name="foo"))

    out = await client.envs.get(name="foo")

    name, args = fake_session.calls[0]
    assert name == "env_get_"
    assert args["name"] == "foo"
    assert "env_id" not in args
    assert isinstance(out, EnvResponse)
    assert out.name == "foo"


async def test_get_by_id(client, fake_session) -> None:
    env_id = uuid4()
    fake_session.set_response("env_get_", make_env_payload(id=str(env_id)))

    out = await client.envs.get(env_id=env_id)

    name, args = fake_session.calls[0]
    assert name == "env_get_"
    assert args["env_id"] == str(env_id)
    assert "name" not in args
    assert isinstance(out, EnvResponse)
    assert str(out.id) == str(env_id)


async def test_attach(client, fake_session) -> None:
    session_id = uuid4()
    fake_session.set_response("env_attach_", _attached_payload(session_id, name="foo"))

    out = await client.envs.attach(name="foo", session_id=session_id)

    assert fake_session.calls == [
        ("env_attach_", {"name": "foo", "session_id": str(session_id)})
    ]
    assert isinstance(out, AttachedEnvsResponse)
    assert str(out.session_id) == str(session_id)
    assert out.attached[0].name == "foo"


async def test_detach(client, fake_session) -> None:
    session_id = uuid4()
    fake_session.set_response("env_detach_", _attached_payload(session_id, name="foo"))

    out = await client.envs.detach(name="foo", session_id=session_id)

    assert fake_session.calls == [
        ("env_detach_", {"name": "foo", "session_id": str(session_id)})
    ]
    assert isinstance(out, AttachedEnvsResponse)
    assert str(out.session_id) == str(session_id)
    assert out.attached[0].name == "foo"
