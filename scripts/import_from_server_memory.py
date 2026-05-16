"""Import memory data from @modelcontextprotocol/server-memory JSONL into memory-mcp.

Usage:
    python -m scripts.import_from_server_memory \
        --input ~/.../memory.jsonl \
        --base-url http://127.0.0.1:8080/mcp \
        --env-id <uuid> \
        [--agent-id <uuid>] \
        [--dry-run] \
        [--batch-size 50]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityRecord:
    line_no: int
    name: str
    kind: str
    observations: tuple[str, ...]


@dataclass(frozen=True)
class RelationRecord:
    line_no: int
    src_name: str
    dst_name: str
    kind: str


@dataclass
class ParsedJsonl:
    entities: list[EntityRecord] = field(default_factory=list)
    relations: list[RelationRecord] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ImportSummary:
    entities_upserted: int = 0
    entities_new: int = 0
    observations_seen: int = 0
    observations_written: int = 0
    observations_new: int = 0
    observations_skipped: int = 0
    relations_seen: int = 0
    relations_linked: int = 0
    relations_new: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)


class ToolClient(Protocol):
    async def call_tool(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        """Call one memory-mcp tool and return its structured payload."""


class McpSdkToolClient:
    """Tiny wrapper over the MCP SDK streamable-HTTP client."""

    def __init__(self, base_url: str, *, agent_id: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.agent_id = agent_id
        self.timeout = timeout
        self._stack: contextlib.AsyncExitStack | None = None
        self._session: Any | None = None

    async def __aenter__(self) -> "McpSdkToolClient":
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self._stack = contextlib.AsyncExitStack()
        headers = {"X-Agent-Id": self.agent_id} if self.agent_id else None
        read, write, _ = await self._stack.enter_async_context(
            streamablehttp_client(self.base_url, headers=headers, timeout=self.timeout)
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def call_tool(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("MCP client is not connected")
        arguments: dict[str, Any] = {"request": request}
        if self.agent_id:
            arguments["agent_id"] = self.agent_id
        result = await self._session.call_tool(name, arguments=arguments)
        if getattr(result, "isError", False):
            raise RuntimeError(_error_text(result) or f"{name} returned an MCP error")
        return _result_payload(result)


def _result_payload(call_result: Any) -> dict[str, Any]:
    if getattr(call_result, "structuredContent", None):
        return dict(call_result.structuredContent)
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            with contextlib.suppress(Exception):
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    return loaded
    return {}


def _error_text(call_result: Any) -> str:
    parts: list[str] = []
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def parse_jsonl(path: Path) -> ParsedJsonl:
    parsed = ParsedJsonl()
    with path.expanduser().open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                parsed.skipped += 1
                log.debug("Skipping blank JSONL line %s", line_no)
                continue
            if stripped.startswith("//"):
                parsed.skipped += 1
                log.debug("Skipping comment JSONL line %s", line_no)
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                parsed.errors.append(f"line {line_no}: malformed JSON: {exc.msg}")
                continue
            if not isinstance(record, dict):
                parsed.errors.append(f"line {line_no}: record must be a JSON object")
                continue
            try:
                _append_record(parsed, line_no, record)
            except (TypeError, ValueError) as exc:
                parsed.errors.append(f"line {line_no}: {exc}")
    return parsed


def _append_record(parsed: ParsedJsonl, line_no: int, record: dict[str, Any]) -> None:
    record_type = record.get("type")
    if record_type == "entity":
        name = _required_str(record, "name")
        kind = _required_str(record, "entityType")
        observations_raw = record.get("observations", [])
        if not isinstance(observations_raw, list) or not all(isinstance(o, str) for o in observations_raw):
            raise ValueError("entity observations must be a list of strings")
        parsed.entities.append(EntityRecord(line_no, name, kind, tuple(observations_raw)))
        return
    if record_type == "relation":
        parsed.relations.append(
            RelationRecord(
                line_no=line_no,
                src_name=_required_str(record, "from"),
                dst_name=_required_str(record, "to"),
                kind=_required_str(record, "relationType"),
            )
        )
        return
    raise ValueError(f"unknown record type {record_type!r}")


def _required_str(record: dict[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name!r} must be a non-empty string")
    return value


def observation_title(entity_name: str, observation: str) -> str:
    return f"{entity_name}: {observation[:60]}{'…' if len(observation) > 60 else ''}"


def observation_hash(env_id: str, entity_id: str, body: str) -> str:
    key = json.dumps([env_id, body, [entity_id]], ensure_ascii=False, separators=(",", ":"))
    return sha256(key.encode("utf-8")).hexdigest()


def _looks_new(payload: dict[str, Any]) -> bool:
    return payload.get("version") == 1 and payload.get("created_at") == payload.get("updated_at")


def _payload_id(payload: dict[str, Any]) -> str:
    value = payload.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"tool response did not include an id: {payload!r}")
    return value


async def import_records(
    parsed: ParsedJsonl,
    *,
    client: ToolClient | None,
    input_path: Path,
    env_id: str,
    agent_id: str | None = None,
    dry_run: bool = False,
) -> ImportSummary:
    summary = ImportSummary(errors=list(parsed.errors))
    source_ref = f"server-memory:{os.path.basename(str(input_path))}"
    entity_ids: dict[str, str] = {}

    for entity in parsed.entities:
        request = {"kind": entity.kind, "canonical_name": entity.name, "env_id": env_id}
        if dry_run:
            print(f"ent_upsert: {json.dumps(request, ensure_ascii=False, sort_keys=True)}")
            fake_id = str(uuid5(NAMESPACE_URL, f"server-memory:{env_id}:{entity.name}"))
            entity_ids[entity.name] = fake_id
            summary.entities_upserted += 1
        else:
            assert client is not None
            try:
                payload = await client.call_tool("ent_upsert", request)
                entity_ids[entity.name] = _payload_id(payload)
                summary.entities_upserted += 1
                summary.entities_new += int(_looks_new(payload))
            except Exception as exc:  # noqa: BLE001 - per-record import must continue
                summary.errors.append(f"line {entity.line_no}: ent_upsert failed for {entity.name!r}: {exc}")
                continue

        ent_id = entity_ids[entity.name]
        for observation in entity.observations:
            summary.observations_seen += 1
            mem_request = {
                "kind": "observation",
                "title": observation_title(entity.name, observation),
                "body": observation,
                "env_id": env_id,
                "entity_links": [ent_id],
                "salience": 0.5,
                "source_type": "import",
                "source_ref": source_ref,
                "metadata": {"server_memory_import_hash": observation_hash(env_id, ent_id, observation)},
            }
            if dry_run:
                print(f"mem_write: {json.dumps(mem_request, ensure_ascii=False, sort_keys=True)}")
                summary.observations_written += 1
                continue
            assert client is not None
            try:
                if await _observation_exists(client, env_id=env_id, entity_id=ent_id, body=observation):
                    summary.observations_skipped += 1
                    continue
                payload = await client.call_tool("mem_write", mem_request)
                summary.observations_written += 1
                summary.observations_new += int(_looks_new(payload)) or 1
            except Exception as exc:  # noqa: BLE001
                summary.errors.append(f"line {entity.line_no}: mem_write failed for {entity.name!r}: {exc}")

    summary.relations_seen = len(parsed.relations)
    for relation in parsed.relations:
        try:
            src_id = entity_ids.get(relation.src_name)
            if src_id is None and not dry_run:
                assert client is not None
                src_id = await _resolve_entity(client, env_id=env_id, name=relation.src_name)
                entity_ids[relation.src_name] = src_id
            dst_id = entity_ids.get(relation.dst_name)
            if dst_id is None and not dry_run:
                assert client is not None
                dst_id = await _resolve_entity(client, env_id=env_id, name=relation.dst_name)
                entity_ids[relation.dst_name] = dst_id
            if src_id is None or dst_id is None:
                raise ValueError("relation references an entity that was not present in the input")
            rel_request = {
                "src": {"kind": "entity", "id": src_id},
                "dst": {"kind": "entity", "id": dst_id},
                "type": relation.kind,
                "env_id": env_id,
            }
            if dry_run:
                print(f"rel_link: {json.dumps(rel_request, ensure_ascii=False, sort_keys=True)}")
                summary.relations_linked += 1
                continue
            assert client is not None
            payload = await client.call_tool("rel_link", rel_request)
            summary.relations_linked += 1
            summary.relations_new += int(_looks_new(payload))
        except Exception as exc:  # noqa: BLE001
            summary.errors.append(f"line {relation.line_no}: rel_link failed for {relation.kind!r}: {exc}")

    return summary


async def _observation_exists(client: ToolClient, *, env_id: str, entity_id: str, body: str) -> bool:
    content_hash = observation_hash(env_id, entity_id, body)
    request = {
        "query": body,
        "env_ids": [env_id],
        "kinds": ["observation"],
        "mode": "lex",
        "limit": 5,
        "consistency": "canonical",
    }
    payload = await client.call_tool("mem_search", request)
    for hit in payload.get("hits", []) or []:
        memory = hit.get("memory", {}) if isinstance(hit, dict) else {}
        if memory.get("body") != body:
            continue
        metadata = memory.get("metadata") or {}
        if metadata.get("server_memory_import_hash") in {None, content_hash}:
            return True
    return False


async def _resolve_entity(client: ToolClient, *, env_id: str, name: str) -> str:
    payload = await client.call_tool("ent_resolve", {"name": name, "env_ids": [env_id], "limit": 1})
    hits = payload if isinstance(payload, list) else payload.get("hits", []) or payload.get("entities", []) or []
    if not hits:
        raise ValueError(f"could not resolve entity {name!r}")
    first = hits[0]
    if not isinstance(first, dict) or not first.get("id"):
        raise ValueError(f"entity resolve response did not include an id for {name!r}: {payload!r}")
    return str(first["id"])


def print_summary(summary: ImportSummary) -> None:
    print(f"Entities upserted: {summary.entities_upserted} (new: {summary.entities_new})")
    print(f"Observations written: {summary.observations_seen} (new: {summary.observations_new})")
    if summary.observations_skipped:
        print(f"Observations skipped as existing: {summary.observations_skipped}")
    print(f"Relations linked: {summary.relations_linked} (new: {summary.relations_new})")
    print(f"Errors: {summary.error_count}")
    for error in summary.errors:
        print(f"ERROR: {error}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to server-memory JSONL file")
    parser.add_argument("--base-url", required=True, help="memory-mcp streamable HTTP URL, e.g. http://127.0.0.1:8080/mcp")
    parser.add_argument("--env-id", required=True, type=UUID, help="Target memory-mcp environment UUID")
    parser.add_argument("--agent-id", type=UUID, help="Optional importing agent UUID")
    parser.add_argument("--dry-run", action="store_true", help="Print intended tool calls without contacting MCP")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved concurrency knob; default keeps calls sequential")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    input_path: Path = args.input.expanduser()
    if not input_path.exists():
        print(f"fatal: input file does not exist: {input_path}", file=sys.stderr)
        return 2

    parsed = parse_jsonl(input_path)
    env_id = str(args.env_id)
    agent_id = str(args.agent_id) if args.agent_id else None

    try:
        if args.dry_run:
            summary = await import_records(parsed, client=None, input_path=input_path, env_id=env_id, agent_id=agent_id, dry_run=True)
        else:
            async with McpSdkToolClient(args.base_url, agent_id=agent_id) as client:
                summary = await import_records(parsed, client=client, input_path=input_path, env_id=env_id, agent_id=agent_id)
    except Exception as exc:  # noqa: BLE001 - top-level fatal connection/init errors
        print(f"fatal: {exc}", file=sys.stderr)
        return 2

    print_summary(summary)
    return 1 if summary.error_count else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
