"""Happy-path coverage for the memories namespace."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from memory_mcp_schemas.browse import (
    MemBrowseResponse,
    MemFacetsResponse,
)
from memory_mcp_schemas.context_pack import ContextPackResponse
from memory_mcp_schemas.digest import DigestResponse, ResumeResponse
from memory_mcp_schemas.env_ops import (
    MemCopyRequest,
    MemCopyResponse,
    MemMoveRequest,
    MemMoveResponse,
)
from memory_mcp_schemas.graph import MemNeighborsResponse, MemRelatedResponse
from memory_mcp_schemas.journal import JournalRequest
from memory_mcp_schemas.memories import (
    MemoryKind,
    MemoryResponse,
    MemorySupersedeRequest,
    MemorySupersedeResponse,
    MemoryUpdatePatch,
    MemoryWriteRequest,
)
from memory_mcp_schemas.provenance import (
    MemLineageResponse,
    MemSourcesBrowseResponse,
)
from memory_mcp_schemas.search import (
    AutoContextResponse,
    MemorySearchResponse,
)

from tests.conftest import make_memory_payload

pytestmark = pytest.mark.asyncio

TS = "2026-05-13T00:00:00Z"


def memory_payload(
    *, memory_id: UUID | None = None, env_id: UUID | None = None, **overrides: Any
) -> dict[str, Any]:
    """Return a valid MemoryResponse-shaped payload."""
    payload = make_memory_payload(
        id=str(memory_id or uuid4()),
        env_id=str(env_id or uuid4()),
    )
    payload.update(overrides)
    return payload


def digest_sections(**overrides: str) -> dict[str, str]:
    sections = {
        "brief": "brief",
        "active_context": "active",
        "system_patterns": "patterns",
        "tech_context": "tech",
        "progress": "progress",
        "open_questions": "questions",
    }
    sections.update(overrides)
    return sections


async def test_write(client, fake_session) -> None:
    env_id = uuid4()
    payload = memory_payload(env_id=env_id, title="hello", body="world")
    fake_session.set_response("mem_write", payload)

    out = await client.memories.write(
        env_id=env_id,
        kind=MemoryKind.fact,
        title="hello",
        body="world",
    )

    name, args = fake_session.calls[0]
    assert name == "mem_write"
    assert isinstance(args["request"], dict)
    assert args["request"]["env_id"] == str(env_id)
    assert args["request"]["title"] == "hello"
    assert args["request"]["kind"] == "fact"
    assert isinstance(out, MemoryResponse)
    assert out.title == "hello"


async def test_get(client, fake_session) -> None:
    memory_id = uuid4()
    fake_session.set_response("mem_get", memory_payload(memory_id=memory_id))

    out = await client.memories.get(memory_id)

    name, args = fake_session.calls[0]
    assert name == "mem_get"
    assert args["memory_id"] == str(memory_id)
    assert isinstance(out, MemoryResponse)
    assert out.id == memory_id


async def test_get_many(client, fake_session) -> None:
    first_id = uuid4()
    second_id = uuid4()
    fake_session.set_response(
        "mem_get_many",
        [
            memory_payload(memory_id=first_id, title="one"),
            memory_payload(memory_id=second_id, title="two"),
        ],
    )

    out = await client.memories.get_many([first_id, second_id])

    name, args = fake_session.calls[0]
    assert name == "mem_get_many"
    assert args["memory_ids"] == [str(first_id), str(second_id)]
    assert len(out) == 2
    assert all(isinstance(item, MemoryResponse) for item in out)
    assert [item.title for item in out] == ["one", "two"]


async def test_update(client, fake_session) -> None:
    memory_id = uuid4()
    patch = MemoryUpdatePatch(expected_version=1, title="new")
    fake_session.set_response(
        "mem_update",
        memory_payload(memory_id=memory_id, title="new", version=2),
    )

    out = await client.memories.update(memory_id, patch)

    name, args = fake_session.calls[0]
    assert name == "mem_update"
    assert args["memory_id"] == str(memory_id)
    assert args["patch"] == {"expected_version": 1, "title": "new"}
    assert isinstance(out, MemoryResponse)
    assert out.title == "new"
    assert out.version == 2


async def test_archive(client, fake_session) -> None:
    memory_id = uuid4()
    fake_session.set_response(
        "mem_archive",
        memory_payload(memory_id=memory_id, status="archived", version=2),
    )

    out = await client.memories.archive(memory_id, expected_version=1)

    name, args = fake_session.calls[0]
    assert name == "mem_archive"
    assert args["memory_id"] == str(memory_id)
    assert args["expected_version"] == 1
    assert isinstance(out, MemoryResponse)
    assert out.status == "archived"


async def test_retire(client, fake_session) -> None:
    memory_id = uuid4()
    fake_session.set_response(
        "mem_retire",
        memory_payload(memory_id=memory_id, status="retired", version=2),
    )

    out = await client.memories.retire(
        memory_id,
        expected_version=1,
        reason="obsolete",
    )

    name, args = fake_session.calls[0]
    assert name == "mem_retire"
    assert args["memory_id"] == str(memory_id)
    assert args["expected_version"] == 1
    assert args["reason"] == "obsolete"
    assert isinstance(out, MemoryResponse)
    assert out.status == "retired"


async def test_supersede(client, fake_session) -> None:
    old_id = uuid4()
    new_id = uuid4()
    request = MemorySupersedeRequest(
        expected_version=1,
        new=MemoryWriteRequest(kind=MemoryKind.fact, title="new", body="new body"),
    )
    fake_session.set_response(
        "mem_supersede",
        {
            "old": memory_payload(memory_id=old_id, status="superseded"),
            "new": memory_payload(memory_id=new_id, title="new", body="new body"),
        },
    )

    out = await client.memories.supersede(old_id, request)

    name, args = fake_session.calls[0]
    assert name == "mem_supersede"
    assert args["old_memory_id"] == str(old_id)
    assert isinstance(args["request"], dict)
    assert args["request"]["expected_version"] == 1
    assert args["request"]["new"]["title"] == "new"
    assert isinstance(out, MemorySupersedeResponse)
    assert isinstance(out.old, MemoryResponse)
    assert isinstance(out.new, MemoryResponse)
    assert out.old.id == old_id
    assert out.new.id == new_id


async def test_hard_delete(client, fake_session) -> None:
    from memory_mcp_schemas.memories import (
        MemoryHardDeleteResponse,
    )

    memory_id = uuid4()
    tombstone_id = uuid4()
    fake_session.set_response(
        "mem_hard_delete",
        {
            "deleted_id": str(memory_id),
            "deleted_at": "2026-05-14T18:30:00+00:00",
            "canonical_deleted": True,
            "projection_eviction": {"qdrant": "pending", "neo4j": "pending"},
            "tombstone_id": str(tombstone_id),
        },
    )

    out = await client.memories.hard_delete(
        memory_id,
        expected_version=1,
        reason="leaked secret recovery",
        confirm_destroy=True,
    )

    name, args = fake_session.calls[0]
    assert name == "mem_hard_delete"
    assert args["memory_id"] == str(memory_id)
    assert isinstance(args["request"], dict)
    assert args["request"]["expected_version"] == 1
    assert args["request"]["confirm_destroy"] is True
    assert args["request"]["reason"] == "leaked secret recovery"
    assert isinstance(out, MemoryHardDeleteResponse)
    assert out.deleted_id == memory_id
    assert out.canonical_deleted is True
    assert out.tombstone_id == tombstone_id
    assert out.projection_eviction.qdrant == "pending"
    assert out.projection_eviction.neo4j == "pending"


async def test_journal(client, fake_session) -> None:
    env_id = uuid4()
    request = JournalRequest(env_id=env_id, content="note", tags=["daily"])
    fake_session.set_response(
        "mem_journal",
        memory_payload(env_id=env_id, kind="journal_entry", body="note"),
    )

    out = await client.memories.journal(request)

    name, args = fake_session.calls[0]
    assert name == "mem_journal"
    assert isinstance(args["request"], dict)
    assert args["request"]["env_id"] == str(env_id)
    assert args["request"]["content"] == "note"
    assert isinstance(out, MemoryResponse)
    assert out.kind == "journal_entry"


async def test_digest(client, fake_session) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    since_ts = datetime(2026, 1, 1, tzinfo=UTC)
    fake_session.set_response(
        "mem_digest",
        {
            "memory_id": str(memory_id),
            "sections": digest_sections(brief="digest"),
            "summarizer_kind": "template",
            "source_type": "test",
        },
    )

    out = await client.memories.digest(env_id, since_ts=since_ts)

    name, args = fake_session.calls[0]
    assert name == "mem_digest"
    assert args["env_id"] == str(env_id)
    assert args["since_ts"] == since_ts.isoformat()
    assert isinstance(out, DigestResponse)
    assert out.memory_id == memory_id
    assert out.sections.brief == "digest"


async def test_resume(client, fake_session) -> None:
    env_id = uuid4()
    journal_id = uuid4()
    fake_session.set_response(
        "mem_resume",
        {
            "latest_digest": digest_sections(brief="latest"),
            "recent_journal": [
                {
                    "id": str(journal_id),
                    "env_id": str(env_id),
                    "kind": "journal_entry",
                    "title": "entry",
                    "body": "note",
                    "salience": 0.5,
                    "created_at": TS,
                    "updated_at": TS,
                }
            ],
            "summary_stats": {
                "memory_count": 1,
                "entity_count": 0,
                "last_journal_ts": TS,
            },
        },
    )

    out = await client.memories.resume(env_id, journal_tail=3)

    name, args = fake_session.calls[0]
    assert name == "mem_resume"
    assert args["env_id"] == str(env_id)
    assert args["journal_tail"] == 3
    assert isinstance(out, ResumeResponse)
    assert out.latest_digest is not None
    assert out.latest_digest.brief == "latest"
    assert out.recent_journal[0].id == journal_id


async def test_search(client, fake_session) -> None:
    env_id = uuid4()
    payload = memory_payload(env_id=env_id, title="needle")
    fake_session.set_response(
        "mem_search",
        {
            "hits": [
                {
                    "memory": payload,
                    "score": 0.9,
                    "sources": ["lex"],
                    "raw_scores": {"lex": 0.9},
                }
            ],
            "mode": "hybrid",
            "effective_mode": "hybrid",
            "consistency_used": "fresh",
            "projection_status": [],
            "truncated": False,
        },
    )

    out = await client.memories.search(query="needle", env_ids=[env_id], limit=5)

    name, args = fake_session.calls[0]
    assert name == "mem_search"
    assert isinstance(args["request"], dict)
    assert args["request"]["query"] == "needle"
    assert args["request"]["env_ids"] == [str(env_id)]
    assert isinstance(out, MemorySearchResponse)
    assert out.hits[0].memory.title == "needle"


async def test_search_passes_fallback_and_min_score(client, fake_session) -> None:
    """v0.12: relax/tighten knobs round-trip through the SDK."""
    env_id = uuid4()
    fake_session.set_response(
        "mem_search",
        {
            "hits": [],
            "mode": "hybrid",
            "effective_mode": "hybrid",
            "consistency_used": "default",
            "projection_status": [],
            "truncated": False,
            "fallback_used": ["mode->hybrid", "drop_filters"],
        },
    )

    out = await client.memories.search(
        query="needle",
        env_ids=[env_id],
        fallback=True,
        min_score=0.025,
    )

    _name, args = fake_session.calls[0]
    assert args["request"]["fallback"] is True
    assert args["request"]["min_score"] == 0.025
    assert out.fallback_used == ["mode->hybrid", "drop_filters"]


async def test_auto_context(client, fake_session) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    fake_session.set_response(
        "mem_auto_context",
        {
            "hits": [
                {
                    "memory_id": str(memory_id),
                    "title": "matched",
                    "body": "body",
                    "trigger_description": "task",
                    "score": 0.8,
                    "salience": 0.7,
                    "kind": "fact",
                }
            ],
            "task_desc_used": "task",
        },
    )

    out = await client.memories.auto_context("task", env_id, top_k=8)

    name, args = fake_session.calls[0]
    assert name == "mem_auto_context"
    assert args["task_desc"] == "task"
    assert args["env_id"] == str(env_id)
    assert args["top_k"] == 8
    assert isinstance(out, AutoContextResponse)
    assert out.hits[0].memory_id == memory_id


async def test_neighbors(client, fake_session) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    entity_id = uuid4()
    fake_session.set_response(
        "mem_neighbors",
        {
            "hits": [
                {
                    "node": {
                        "kind": "memory",
                        "id": str(memory_id),
                        "name": "neighbor",
                        "env_id": str(env_id),
                    },
                    "path_length": 1,
                    "path": [
                        {
                            "src_kind": "memory",
                            "src_id": str(memory_id),
                            "dst_kind": "entity",
                            "dst_id": str(entity_id),
                            "edge_type": "mentions",
                        }
                    ],
                    "score": 1.0,
                }
            ],
            "next_cursor": None,
        },
    )

    out = await client.memories.neighbors(memory_id=memory_id, env_id=env_id)

    name, args = fake_session.calls[0]
    assert name == "mem_neighbors"
    assert isinstance(args["request"], dict)
    assert args["request"]["memory_id"] == str(memory_id)
    assert args["request"]["env_id"] == str(env_id)
    assert isinstance(out, MemNeighborsResponse)
    assert out.hits[0].node.id == memory_id


async def test_related(client, fake_session) -> None:
    memory_id = uuid4()
    related_id = uuid4()
    entity_id = uuid4()
    fake_session.set_response(
        "mem_related",
        {
            "hits": [
                {
                    "memory_id": str(related_id),
                    "score": 0.7,
                    "shared_entity_ids": [str(entity_id)],
                    "memory": memory_payload(memory_id=related_id, title="related"),
                }
            ],
            "next_cursor": None,
            "note": "ok",
        },
    )

    out = await client.memories.related(
        memory_id=memory_id,
        relation="shared_entity",
        limit=3,
    )

    name, args = fake_session.calls[0]
    assert name == "mem_related"
    assert isinstance(args["request"], dict)
    assert args["request"]["memory_id"] == str(memory_id)
    assert args["request"]["limit"] == 3
    assert isinstance(out, MemRelatedResponse)
    assert out.hits[0].memory.title == "related"


async def test_lineage(client, fake_session) -> None:
    seed_id = uuid4()
    child_id = uuid4()
    seed = memory_payload(memory_id=seed_id, title="seed")
    fake_session.set_response(
        "mem_lineage",
        {
            "seed": seed,
            "ancestors": [],
            "descendants": [
                {
                    "parent_memory_id": str(seed_id),
                    "child_memory_id": str(child_id),
                    "relation": "supersedes",
                    "created_at": TS,
                    "depth": 1,
                }
            ],
            "nodes": {str(seed_id): seed},
            "truncated": False,
        },
    )

    out = await client.memories.lineage(
        memory_id=seed_id,
        direction="descendants",
        max_depth=2,
    )

    name, args = fake_session.calls[0]
    assert name == "mem_lineage"
    assert isinstance(args["request"], dict)
    assert args["request"]["memory_id"] == str(seed_id)
    assert args["request"]["direction"] == "descendants"
    assert isinstance(out, MemLineageResponse)
    assert out.seed.id == seed_id
    assert out.descendants[0].child_memory_id == child_id


async def test_sources_browse(client, fake_session) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    memory = memory_payload(memory_id=memory_id, env_id=env_id)
    fake_session.set_response(
        "mem_sources_browse",
        {
            "hits": [
                {
                    "id": 1,
                    "memory_id": str(memory_id),
                    "env_id": str(env_id),
                    "source_type": "agent",
                    "source_ref": "tests",
                    "agent_id": None,
                    "created_at": TS,
                    "evidence_span": "span",
                }
            ],
            "next_cursor": None,
            "nodes": {str(memory_id): memory},
        },
    )

    out = await client.memories.sources_browse(
        env_ids=[env_id],
        memory_ids=[memory_id],
        hydrate_memories=True,
    )

    name, args = fake_session.calls[0]
    assert name == "mem_sources_browse"
    assert isinstance(args["request"], dict)
    assert args["request"]["env_ids"] == [str(env_id)]
    assert args["request"]["memory_ids"] == [str(memory_id)]
    assert isinstance(out, MemSourcesBrowseResponse)
    assert out.hits[0].memory_id == memory_id
    assert out.nodes is not None
    assert out.nodes[memory_id].id == memory_id


async def test_browse(client, fake_session) -> None:
    env_id = uuid4()
    payload = memory_payload(env_id=env_id, title="browsed")
    fake_session.set_response(
        "mem_browse",
        {
            "hits": [payload],
            "next_cursor": None,
            "has_more": False,
            "schema_version": 1,
        },
    )

    out = await client.memories.browse(env_ids=[env_id], tags=["tag"], limit=10)

    name, args = fake_session.calls[0]
    assert name == "mem_browse"
    assert isinstance(args["request"], dict)
    assert args["request"]["env_ids"] == [str(env_id)]
    assert args["request"]["tags"] == ["tag"]
    assert isinstance(out, MemBrowseResponse)
    assert out.hits[0].title == "browsed"


async def test_facets(client, fake_session) -> None:
    env_id = uuid4()
    fake_session.set_response(
        "mem_facets",
        {
            "total": 1,
            "by_env": {str(env_id): 1},
            "facets": {
                "kind": [{"value": "fact", "count": 1}],
                "status": [{"value": "active", "count": 1}],
                "tag": [{"value": "tag", "count": 1}],
            },
            "approximate": False,
            "sampled_rows": 0,
            "schema_version": 1,
        },
    )

    out = await client.memories.facets(env_ids=[env_id], facets=["kind", "tag"])

    name, args = fake_session.calls[0]
    assert name == "mem_facets"
    assert isinstance(args["request"], dict)
    assert args["request"]["env_ids"] == [str(env_id)]
    assert args["request"]["facets"] == ["kind", "tag"]
    assert isinstance(out, MemFacetsResponse)
    assert out.total == 1
    assert out.by_env[env_id] == 1


async def test_context_pack(client, fake_session) -> None:
    env_id = uuid4()
    memory_id = uuid4()
    fake_session.set_response(
        "mem_context_pack",
        {
            "sections": [
                {
                    "name": "trigger_matched",
                    "items": [
                        {
                            "memory_id": str(memory_id),
                            "title": "packed",
                            "body": "body",
                            "kind": "fact",
                            "salience": 0.5,
                            "tokens_used": 5,
                            "body_truncated": False,
                        }
                    ],
                    "tokens_used": 5,
                    "cap_tokens": 100,
                    "truncation_count": 0,
                }
            ],
            "total_tokens": 5,
            "budget_used_pct": 0.05,
            "sections_skipped": [],
            "task_desc_used": "task",
        },
    )

    out = await client.memories.context_pack(
        "task",
        env_id,
        token_budget=100,
        include_core=True,
        include_journal=False,
    )

    name, args = fake_session.calls[0]
    assert name == "mem_context_pack"
    assert args["task_desc"] == "task"
    assert args["env_id"] == str(env_id)
    assert args["token_budget"] == 100
    assert args["include_core"] is True
    assert args["include_journal"] is False
    assert isinstance(out, ContextPackResponse)
    assert out.sections[0].items[0].memory_id == memory_id


async def test_memories_copy_calls_correct_tool(client, fake_session) -> None:
    memory_id = uuid4()
    dst_env_id = uuid4()
    dst_memory_id = uuid4()
    request = MemCopyRequest(memory_id=memory_id, dst_env_id=dst_env_id)
    fake_session.set_response(
        "mem_copy_",
        {
            "dst_memory_id": str(dst_memory_id),
            "dst_env_id": str(dst_env_id),
            "lineage_edge_id": None,
            "pending_vector_rebuild": 0,
        },
    )

    out = await client.memories.copy(request)

    assert fake_session.calls == [("mem_copy_", {"request": request.model_dump(mode="json")})]
    assert isinstance(out, MemCopyResponse)
    assert out.dst_memory_id == dst_memory_id


async def test_memories_move_calls_correct_tool(client, fake_session) -> None:
    memory_id = uuid4()
    dst_env_id = uuid4()
    dst_memory_id = uuid4()
    request = MemMoveRequest(memory_id=memory_id, dst_env_id=dst_env_id)
    fake_session.set_response(
        "mem_move_",
        {
            "dst_memory_id": str(dst_memory_id),
            "dst_env_id": str(dst_env_id),
            "lineage_edge_id": "edge-1",
            "pending_vector_rebuild": 0,
            "source_memory_status": "superseded",
        },
    )

    out = await client.memories.move(request)

    assert fake_session.calls == [("mem_move_", {"request": request.model_dump(mode="json")})]
    assert isinstance(out, MemMoveResponse)
    assert out.source_memory_status == "superseded"


async def test_memories_copy_passes_options(client, fake_session) -> None:
    memory_id = uuid4()
    dst_env_id = uuid4()
    request = MemCopyRequest(
        memory_id=memory_id,
        dst_env_id=dst_env_id,
        copy_tags=False,
        copy_provenance=False,
        create_lineage_edge=False,
        preserve_timestamps=True,
        re_embed_if_model_mismatch=True,
        copy_lineage=True,
        copy_entities="always_create",
    )
    fake_session.set_response(
        "mem_copy_",
        {
            "dst_memory_id": str(uuid4()),
            "dst_env_id": str(dst_env_id),
            "lineage_edge_id": None,
            "pending_vector_rebuild": 1,
        },
    )

    await client.memories.copy(request)

    name, args = fake_session.calls[0]
    assert name == "mem_copy_"
    assert args["request"]["copy_tags"] is False
    assert args["request"]["copy_provenance"] is False
    assert args["request"]["create_lineage_edge"] is False
    assert args["request"]["preserve_timestamps"] is True
    assert args["request"]["re_embed_if_model_mismatch"] is True
    assert args["request"]["copy_lineage"] is True
    assert args["request"]["copy_entities"] == "always_create"
