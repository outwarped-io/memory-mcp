"""Happy-path tests for the entities client namespace."""

from __future__ import annotations

from uuid import uuid4

import pytest

from memory_mcp_schemas.entities import (
    EntityBrowseRequest,
    EntityBrowseResponse,
    EntityMergeRequest,
    EntityResolveRequest,
    EntityResponse,
    EntityUpsertRequest,
)
from memory_mcp_schemas.graph import EntityNeighborsRequest, EntityNeighborsResponse


pytestmark = pytest.mark.asyncio


def _entity_payload(**overrides):
    base = {
        "id": "00000000-0000-0000-0000-00000000e101",
        "env_id": "00000000-0000-0000-0000-0000000000e0",
        "kind": "service",
        "canonical_name": "Service A",
        "normalized_name": "service a",
        "aliases": [],
        "metadata": {},
        "version": 1,
        "created_at": "2026-05-13T00:00:00Z",
        "updated_at": "2026-05-13T00:00:00Z",
    }
    base.update(overrides)
    return base


def _neighbors_payload(entity_id, env_id):
    return {
        "hits": [
            {
                "node": {
                    "kind": "entity",
                    "id": str(entity_id),
                    "name": "Service B",
                    "env_id": str(env_id),
                },
                "path_length": 1,
                "path": [
                    {
                        "src_kind": "entity",
                        "src_id": "00000000-0000-0000-0000-00000000e101",
                        "dst_kind": "entity",
                        "dst_id": str(entity_id),
                        "edge_type": "depends_on",
                    }
                ],
                "score": 1.0,
            }
        ],
        "next_cursor": None,
    }


async def test_upsert(client, fake_session) -> None:
    request = EntityUpsertRequest(kind="service", canonical_name="Service A")
    fake_session.set_response("ent_upsert", _entity_payload())

    out = await client.entities.upsert(request)

    assert fake_session.calls == [
        ("ent_upsert", {"request": request.model_dump(mode="json")})
    ]
    assert isinstance(out, EntityResponse)
    assert out.canonical_name == "Service A"


async def test_resolve_returns_list(client, fake_session) -> None:
    request = EntityResolveRequest(name="Service")
    fake_session.set_response(
        "ent_resolve",
        [
            _entity_payload(),
            _entity_payload(
                id="00000000-0000-0000-0000-00000000e102",
                canonical_name="Service B",
                normalized_name="service b",
            ),
        ],
    )

    out = await client.entities.resolve(request)

    assert fake_session.calls == [
        ("ent_resolve", {"request": request.model_dump(mode="json")})
    ]
    assert len(out) == 2
    assert all(isinstance(entity, EntityResponse) for entity in out)


async def test_merge(client, fake_session) -> None:
    keep_id = uuid4()
    merge_id = uuid4()
    request = EntityMergeRequest(
        keep_id=keep_id,
        merge_ids=[merge_id],
        expected_versions={keep_id: 1, merge_id: 1},
    )
    fake_session.set_response("ent_merge", _entity_payload(id=str(keep_id)))

    out = await client.entities.merge(request)

    assert fake_session.calls == [
        ("ent_merge", {"request": request.model_dump(mode="json")})
    ]
    assert isinstance(out, EntityResponse)
    assert str(out.id) == str(keep_id)


async def test_neighbors(client, fake_session) -> None:
    env_id = uuid4()
    entity_id = uuid4()
    request = EntityNeighborsRequest(entity_id=entity_id, env_id=env_id, hops=1)
    fake_session.set_response("ent_neighbors", _neighbors_payload(entity_id, env_id))

    out = await client.entities.neighbors(request)

    assert fake_session.calls == [
        ("ent_neighbors", {"request": request.model_dump(mode="json")})
    ]
    assert isinstance(out, EntityNeighborsResponse)
    assert str(out.hits[0].node.id) == str(entity_id)


async def test_browse(client, fake_session) -> None:
    request = EntityBrowseRequest(kinds=["service"], limit=5)
    fake_session.set_response(
        "ent_browse",
        {
            "hits": [_entity_payload()],
            "next_cursor": None,
            "has_more": False,
        },
    )

    out = await client.entities.browse(request)

    assert fake_session.calls == [
        ("ent_browse", {"request": request.model_dump(mode="json")})
    ]
    assert isinstance(out, EntityBrowseResponse)
    assert len(out.hits) == 1
