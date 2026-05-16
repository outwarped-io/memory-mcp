from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from memory_mcp.db.models import Environment, Memory, Tag
from memory_mcp.env_ops import import_ as importer
from memory_mcp.identity import AgentContext
from memory_mcp_schemas.env_ops import EnvImportRequest, ImportMode


class _ScalarResult:
    def __init__(self, value: object = None) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object:
        return self._value


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.tags_by_id: dict[UUID, Tag] = {}
        self.existing_tags: set[tuple[UUID, str]] = set()

    def add(self, obj: object) -> None:
        self.added.append(obj)
        if isinstance(obj, Tag):
            self.tags_by_id[obj.id] = obj
            self.existing_tags.add((obj.env_id, obj.name))

    async def flush(self) -> None:
        return None

    async def execute(self, stmt: object) -> _ScalarResult:
        return _ScalarResult(None)


@pytest.fixture
def ctx() -> AgentContext:
    return AgentContext(agent_id=uuid4(), agent_name="test-agent")


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeSession]:
    session = _FakeSession()

    @asynccontextmanager
    async def fake_session_scope() -> AsyncIterator[_FakeSession]:
        yield session

    async def fake_tag_exists(sess: _FakeSession, env_id: UUID, name: str) -> bool:
        return (env_id, name) in sess.existing_tags

    async def fake_tag_name(sess: _FakeSession, tag_id: UUID) -> str | None:
        tag = sess.tags_by_id.get(tag_id)
        return None if tag is None else tag.name

    monkeypatch.setattr(importer, "session_scope", fake_session_scope)
    monkeypatch.setattr(importer, "_tag_exists", fake_tag_exists)
    monkeypatch.setattr(importer, "_tag_name", fake_tag_name)
    monkeypatch.setattr(importer, "_entity_exists", _false_exists)
    monkeypatch.setattr(importer, "_entity_alias_exists", _false_exists)
    monkeypatch.setattr(importer, "_relation_exists", _false_exists)
    yield session


async def _false_exists(*_args: object, **_kwargs: object) -> bool:
    return False


@pytest.mark.asyncio
async def test_import_dry_run_reports_counts(tmp_path: Path, fake_session: _FakeSession, ctx: AgentContext) -> None:
    archive = _build_fake_archive(tmp_path, memories_count=1)

    report = await importer.import_env(
        EnvImportRequest(source_path=str(archive), target_env_name="dry-run-env", dry_run=True),
        ctx=ctx,
    )

    assert report.dry_run is True
    assert report.counts["memories"] == 1
    assert report.conflicts["memories"] == 0
    assert fake_session.added == []


@pytest.mark.asyncio
async def test_import_full_into_new_env_fail_mode(tmp_path: Path, fake_session: _FakeSession, ctx: AgentContext) -> None:
    source_memory_id = uuid4()
    archive = _build_fake_archive(tmp_path, memory_ids=[source_memory_id])

    report = await importer.import_env(
        EnvImportRequest(
            source_path=str(archive),
            target_env_name="imported-env",
            dry_run=False,
            mode=ImportMode.fail,
        ),
        ctx=ctx,
    )

    memories = [obj for obj in fake_session.added if isinstance(obj, Memory)]
    assert report.counts["memories"] == 1
    assert len(memories) == 1
    assert memories[0].env_id == report.target_env_id
    assert memories[0].id != source_memory_id


@pytest.mark.asyncio
async def test_import_rejects_bad_checksum(tmp_path: Path, fake_session: _FakeSession, ctx: AgentContext) -> None:
    archive = _build_fake_archive(tmp_path, memories_count=1)
    (archive / "memories.jsonl").write_text('{"corrupt": true}\n', encoding="utf-8")

    with pytest.raises(importer.ChecksumMismatchError) as exc:
        await importer.import_env(
            EnvImportRequest(source_path=str(archive), target_env_name="bad-checksum", dry_run=True),
            ctx=ctx,
        )

    assert exc.value.code == "IMPORT_CHECKSUM_MISMATCH"


@pytest.mark.asyncio
async def test_import_rejects_future_version(tmp_path: Path, fake_session: _FakeSession, ctx: AgentContext) -> None:
    archive = _build_fake_archive(tmp_path, memories_count=1, schema_version="9.9.9")

    with pytest.raises(importer.ArchiveVersionError) as exc:
        await importer.import_env(
            EnvImportRequest(source_path=str(archive), target_env_name="future-version", dry_run=True),
            ctx=ctx,
        )

    assert exc.value.details["decision"] == "reject_too_new"


@pytest.mark.asyncio
async def test_import_bulk_reembed_blocked(
    tmp_path: Path,
    fake_session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
    ctx: AgentContext,
) -> None:
    target_env_id = uuid4()
    target_env = Environment(
        id=target_env_id,
        name="existing",
        kind=None,
        retention_policy={},
        default_embedding_model_id="target-model",
    )
    monkeypatch.setattr(importer, "get_env_by_id", _get_env(target_env))
    archive = _build_fake_archive(
        tmp_path,
        memories_count=0,
        manifest_counts={"memories": 20_000},
        source_model_id="source-model",
    )

    with pytest.raises(importer.BulkReembedBlocked) as exc:
        await importer.import_env(
            EnvImportRequest(source_path=str(archive), target_env_id=target_env_id, dry_run=True),
            ctx=ctx,
        )
    assert "20,000" in str(exc.value) or "20000" in str(exc.value)

    report = await importer.import_env(
        EnvImportRequest(
            source_path=str(archive),
            target_env_id=target_env_id,
            dry_run=True,
            allow_bulk_reembed=True,
        ),
        ctx=ctx,
    )
    assert report.target_env_id == target_env_id


@pytest.mark.asyncio
async def test_import_two_pass_supersession(tmp_path: Path, fake_session: _FakeSession, ctx: AgentContext) -> None:
    memory_a = uuid4()
    memory_b = uuid4()
    archive = _build_fake_archive(tmp_path, memory_ids=[memory_a, memory_b], superseded_by={memory_a: memory_b})

    await importer.import_env(
        EnvImportRequest(source_path=str(archive), target_env_name="supersession", dry_run=False),
        ctx=ctx,
    )

    imported = {obj.body: obj for obj in fake_session.added if isinstance(obj, Memory)}
    assert imported["memory 0"].superseded_by == imported["memory 1"].id
    assert imported["memory 0"].superseded_by != memory_b


@pytest.mark.asyncio
async def test_import_skip_mode_skips_collisions(tmp_path: Path, fake_session: _FakeSession, ctx: AgentContext) -> None:
    dst_env_id = uuid4()
    fake_session.existing_tags.add((dst_env_id, "foo"))
    archive = _build_fake_archive(tmp_path, memories_count=0, tags=[(uuid4(), "foo")])

    target_env = Environment(
        id=dst_env_id,
        name="existing",
        kind=None,
        retention_policy={},
        default_embedding_model_id="model-a",
    )

    async def fake_get_env_by_id(env_id: UUID, *, include_deleted: bool = False) -> Environment | None:
        assert env_id == dst_env_id
        assert include_deleted is False
        return target_env

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(importer, "get_env_by_id", fake_get_env_by_id)
        report = await importer.import_env(
            EnvImportRequest(
                source_path=str(archive),
                target_env_id=dst_env_id,
                dry_run=False,
                mode=ImportMode.skip,
            ),
            ctx=ctx,
        )

    inserted_foo = [obj for obj in fake_session.added if isinstance(obj, Tag) and obj.name == "foo"]
    assert report.conflicts["tags"] == 1
    assert inserted_foo == []


def _get_env(env: Environment):
    async def fake_get_env_by_id(env_id: UUID, *, include_deleted: bool = False) -> Environment | None:
        assert env_id == env.id
        assert include_deleted is False
        return env

    return fake_get_env_by_id


def _build_fake_archive(
    tmp_path: Path,
    *,
    memories_count: int | None = None,
    memory_ids: list[UUID] | None = None,
    superseded_by: dict[UUID, UUID] | None = None,
    tags: list[tuple[UUID, str]] | None = None,
    schema_version: str = "0.8.0",
    source_model_id: str = "model-a",
    manifest_counts: dict[str, int] | None = None,
) -> Path:
    root = tmp_path / f"archive-{uuid4().hex}"
    root.mkdir()
    ids = memory_ids or [uuid4() for _ in range(memories_count or 0)]
    superseded_by = superseded_by or {}
    tags = tags or []

    _write_jsonl(root / "tags.jsonl", [{"id": tag_id, "env_id": uuid4(), "name": name} for tag_id, name in tags])
    _write_jsonl(
        root / "memories.jsonl",
        [
            {
                "id": memory_id,
                "env_id": uuid4(),
                "kind": "fact",
                "status": "active",
                "title": None,
                "body": f"memory {index}",
                "metadata": {},
                "salience": 0.5,
                "confidence": 0.5,
                "pinned": False,
                "version": 1,
                "superseded_by": superseded_by.get(memory_id),
            }
            for index, memory_id in enumerate(ids)
        ],
    )
    for name in (
        "entities",
        "entity_aliases",
        "memory_tags",
        "memory_sources",
        "graph_nodes",
        "relations",
        "memory_lineage",
        "tasks",
        "dream_runs",
        "dream_proposals",
    ):
        _write_jsonl(root / f"{name}.jsonl", [])

    counts = {
        "tags": len(tags),
        "memories": len(ids),
        "entities": 0,
        "entity_aliases": 0,
        "memory_tags": 0,
        "memory_sources": 0,
        "graph_nodes": 0,
        "relations": 0,
        "memory_lineage": 0,
        "tasks": 0,
        "dream_runs": 0,
        "dream_proposals": 0,
    }
    if manifest_counts:
        counts.update(manifest_counts)

    manifest = {
        "schema_version": schema_version,
        "memory_mcp_version": "0.8.0",
        "source": {
            "env_id": str(uuid4()),
            "env_name": "source",
            "default_embedding_model_id": source_model_id,
            "instance_fingerprint": "fake",
        },
        "exported_at": dt.datetime(2026, 5, 13, tzinfo=dt.UTC).isoformat(),
        "exported_by_agent": "test",
        "include_flags": {"embeddings": False, "provenance": True, "dream_history": False, "grants": False},
        "counts": counts,
        "checksums": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, default=str), encoding="utf-8")
    checksums = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    (root / "checksums.sha256").write_text(
        "".join(f"{digest}  {rel_path}\n" for rel_path, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    return root


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, default=str, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
