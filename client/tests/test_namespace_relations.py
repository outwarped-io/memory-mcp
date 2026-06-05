"""Happy-path tests for the relations client namespace."""

from __future__ import annotations

from uuid import uuid4

import pytest
from memory_mcp_schemas.relations import (
    RelationBrowseRequest,
    RelationBrowseResponse,
    RelationLinkRequest,
    RelationResponse,
)

pytestmark = pytest.mark.asyncio


def _endpoint_payload(record_id):
    return {"kind": "entity", "id": str(record_id)}


def _relation_payload(**overrides):
    src_id = overrides.pop("src_id", "00000000-0000-0000-0000-00000000e101")
    dst_id = overrides.pop("dst_id", "00000000-0000-0000-0000-00000000e102")
    base = {
        "id": "00000000-0000-0000-0000-00000000f001",
        "env_id": "00000000-0000-0000-0000-0000000000e0",
        "src": _endpoint_payload(src_id),
        "dst": _endpoint_payload(dst_id),
        "src_node_id": "00000000-0000-0000-0000-00000000a101",
        "dst_node_id": "00000000-0000-0000-0000-00000000a102",
        "type": "depends_on",
        "properties": {},
        "version": 1,
        "created_at": "2026-05-13T00:00:00Z",
        "updated_at": "2026-05-13T00:00:00Z",
    }
    base.update(overrides)
    return base


def _relation_browse_hit(**overrides):
    payload = _relation_payload(**overrides)
    return {
        "id": payload["id"],
        "env_id": payload["env_id"],
        "type": payload["type"],
        "src_kind": payload["src"]["kind"],
        "src_id": payload["src"]["id"],
        "dst_kind": payload["dst"]["kind"],
        "dst_id": payload["dst"]["id"],
        "properties": payload["properties"],
        "created_at": payload["created_at"],
        "updated_at": payload["updated_at"],
    }


async def test_link(client, fake_session) -> None:
    src_id = uuid4()
    dst_id = uuid4()
    request = RelationLinkRequest(
        src={"kind": "entity", "id": src_id},
        dst={"kind": "entity", "id": dst_id},
        type="depends_on",
        properties={"weight": 1},
    )
    fake_session.set_response(
        "rel_link",
        _relation_payload(src_id=str(src_id), dst_id=str(dst_id), properties={"weight": 1}),
    )

    out = await client.relations.link(request)

    assert fake_session.calls == [("rel_link", {"request": request.model_dump(mode="json")})]
    name, args = fake_session.calls[0]
    assert name == "rel_link"
    assert args["request"]["src"] == {"kind": "entity", "id": str(src_id)}
    assert args["request"]["dst"] == {"kind": "entity", "id": str(dst_id)}
    assert args["request"]["type"] == "depends_on"
    assert args["request"]["properties"] == {"weight": 1}
    assert isinstance(out, RelationResponse)


async def test_browse(client, fake_session) -> None:
    env_id = uuid4()
    request = RelationBrowseRequest(env_ids=[env_id], types=["depends_on"], limit=5)
    fake_session.set_response(
        "rel_browse",
        {
            "hits": [_relation_browse_hit(env_id=str(env_id))],
            "next_cursor": None,
            "has_more": False,
        },
    )

    out = await client.relations.browse(request)

    assert fake_session.calls == [("rel_browse", {"request": request.model_dump(mode="json")})]
    assert isinstance(out, RelationBrowseResponse)
    assert len(out.hits) == 1
