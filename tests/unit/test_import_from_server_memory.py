from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, uuid5, NAMESPACE_DNS

import pytest

from scripts.import_from_server_memory import import_records, observation_title, parse_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "server_memory_sample.jsonl"
ENV_ID = "11111111-1111-4111-8111-111111111111"


class MockMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.entities: dict[str, str] = {}
        self.memories: list[dict[str, Any]] = []
        self.relations: set[tuple[str, str, str]] = set()

    async def call_tool(self, name: str, request: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        self.calls.append((name, request))
        if name == "ent_upsert":
            ent_id = self.entities.setdefault(request["canonical_name"], str(uuid5(NAMESPACE_DNS, request["canonical_name"])))
            return self._payload(ent_id, version=1)
        if name == "mem_search":
            hits = []
            query = request["query"]
            for memory in self.memories:
                if memory["body"] == query:
                    hits.append({"memory": memory, "score": 1.0, "sources": ["lex"], "raw_scores": {"lex": 1.0}})
            return {"hits": hits, "mode": "lex", "effective_mode": "lex", "consistency_used": "canonical", "projection_status": []}
        if name == "mem_write":
            mem_id = str(uuid5(NAMESPACE_DNS, f"memory:{len(self.memories)}:{request['body']}"))
            memory = {
                "id": mem_id,
                "env_id": request["env_id"],
                "kind": request["kind"],
                "status": "active",
                "title": request["title"],
                "body": request["body"],
                "metadata": dict(request.get("metadata") or {}),
                "version": 1,
                "created_at": "2026-05-11T00:00:00+00:00",
                "updated_at": "2026-05-11T00:00:00+00:00",
            }
            self.memories.append(memory)
            return memory
        if name == "rel_link":
            key = (request["src"]["id"], request["dst"]["id"], request["type"])
            self.relations.add(key)
            return self._payload(str(uuid5(NAMESPACE_DNS, ":".join(key))), version=1)
        if name == "ent_resolve":
            name_arg = request["name"]
            return [{"id": self.entities[name_arg], "canonical_name": name_arg}]
        raise AssertionError(f"unexpected tool {name}")

    @staticmethod
    def _payload(record_id: str, *, version: int) -> dict[str, Any]:
        return {
            "id": record_id,
            "version": version,
            "created_at": "2026-05-11T00:00:00+00:00",
            "updated_at": "2026-05-11T00:00:00+00:00",
        }


def mutating_calls(client: MockMcpClient) -> list[tuple[str, dict[str, Any]]]:
    return [(name, request) for name, request in client.calls if name in {"ent_upsert", "mem_write", "rel_link"}]


@pytest.mark.asyncio
async def test_happy_path_imports_entities_observations_and_relations_in_order() -> None:
    parsed = parse_jsonl(FIXTURE)
    client = MockMcpClient()

    summary = await import_records(parsed, client=client, input_path=FIXTURE, env_id=ENV_ID)

    calls = mutating_calls(client)
    assert [name for name, _ in calls] == [
        "ent_upsert", "mem_write", "mem_write",
        "ent_upsert", "mem_write", "mem_write",
        "ent_upsert", "mem_write",
        "rel_link", "rel_link",
    ]
    assert calls[0][1] == {"kind": "person", "canonical_name": "John_Smith", "env_id": ENV_ID}
    assert calls[3][1] == {"kind": "organization", "canonical_name": "Anthropic", "env_id": ENV_ID}
    assert calls[6][1] == {"kind": "event", "canonical_name": "MCP_Summit_2026", "env_id": ENV_ID}
    assert calls[-2][1]["type"] == "works_at"
    assert calls[-1][1]["type"] == "attended"
    assert summary.entities_upserted == 3
    assert summary.observations_written == 5
    assert summary.relations_linked == 2
    assert summary.error_count == 0


def test_observation_title_truncation() -> None:
    body = "x" * 61
    assert observation_title("Anthropic", body) == f"Anthropic: {'x' * 60}…"


@pytest.mark.asyncio
async def test_source_type_import_threaded_to_every_mem_write() -> None:
    client = MockMcpClient()

    await import_records(parse_jsonl(FIXTURE), client=client, input_path=FIXTURE, env_id=ENV_ID)

    mem_writes = [request for name, request in client.calls if name == "mem_write"]
    assert len(mem_writes) == 5
    assert {request["source_type"] for request in mem_writes} == {"import"}
    assert {request["source_ref"] for request in mem_writes} == {"server-memory:server_memory_sample.jsonl"}


@pytest.mark.asyncio
async def test_idempotency_second_run_skips_existing_observations() -> None:
    parsed = parse_jsonl(FIXTURE)
    client = MockMcpClient()

    first = await import_records(parsed, client=client, input_path=FIXTURE, env_id=ENV_ID)
    second = await import_records(parsed, client=client, input_path=FIXTURE, env_id=ENV_ID)

    assert first.observations_written == 5
    assert second.observations_written == 0
    assert second.observations_skipped == 5
    assert len([name for name, _ in client.calls if name == "mem_write"]) == 5


@pytest.mark.asyncio
async def test_dry_run_prints_intended_calls_without_real_client(capsys: pytest.CaptureFixture[str]) -> None:
    parsed = parse_jsonl(FIXTURE)

    summary = await import_records(parsed, client=None, input_path=FIXTURE, env_id=ENV_ID, dry_run=True)

    output = capsys.readouterr().out
    assert "ent_upsert:" in output
    assert "mem_write:" in output
    assert "rel_link:" in output
    assert "server-memory:server_memory_sample.jsonl" in output
    assert summary.observations_seen == 5
    assert summary.entities_upserted == 3
    assert summary.relations_linked == 2


def test_blank_and_comment_lines_are_skipped_without_counting_as_records() -> None:
    parsed = parse_jsonl(FIXTURE)

    assert parsed.skipped == 2
    assert len(parsed.entities) == 3
    assert sum(len(entity.observations) for entity in parsed.entities) == 5
    assert len(parsed.relations) == 2
    assert parsed.errors == []


def test_fixture_env_id_is_uuid() -> None:
    assert str(UUID(ENV_ID)) == ENV_ID
