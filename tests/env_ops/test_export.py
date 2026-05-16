from __future__ import annotations

import datetime as dt
import json
import tarfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from memory_mcp.db.models import (
    DreamProposal,
    DreamRun,
    Entity,
    EnvGrant,
    Environment,
    Memory,
    Tag,
)
from memory_mcp.env_ops.export import export_env
from memory_mcp.errors import NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp_schemas.env_ops import EnvExportRequest, ExportFormat, ExportManifest


class _ScalarResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows

    def first(self) -> object | None:
        return self._rows[0] if self._rows else None


class _ExecuteResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _ScalarResult:
        return _ScalarResult(self._rows)

    def all(self) -> list[object]:
        return self._rows


class _Session:
    def __init__(self, fixture: "_Fixture") -> None:
        self._fixture = fixture

    async def connection(self, **_kwargs: object) -> None:
        return None

    async def execute(self, stmt: object) -> _ExecuteResult:
        sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))  # type: ignore[attr-defined]
        rows: list[object]
        if "count(" in sql and "FROM memories" in sql:
            by_kind: dict[str, int] = {}
            for memory in self._fixture.memories:
                by_kind[memory.kind] = by_kind.get(memory.kind, 0) + 1
            rows = [(kind, count) for kind, count in by_kind.items()]
        elif "FROM environments" in sql:
            rows = [self._fixture.env] if self._fixture.env is not None else []
        elif "FROM memory_tags" in sql:
            rows = self._fixture.memory_tags
        elif "FROM memory_sources" in sql:
            rows = self._fixture.memory_sources
        elif "FROM memory_lineage" in sql:
            memory_ids = {memory.id for memory in self._fixture.memories}
            rows = [
                row
                for row in self._fixture.memory_lineage
                if row.parent_memory_id in memory_ids and row.child_memory_id in memory_ids
            ]
        elif "FROM entity_aliases" in sql:
            rows = self._fixture.entity_aliases
        elif "FROM entities" in sql:
            rows = self._fixture.entities
        elif "FROM relations" in sql:
            rows = self._fixture.relations
        elif "FROM graph_nodes" in sql:
            rows = self._fixture.graph_nodes
        elif "FROM tags" in sql:
            rows = self._fixture.tags
        elif "FROM tasks" in sql:
            rows = self._fixture.tasks
        elif "FROM env_grants" in sql:
            rows = self._fixture.grants
        elif "FROM dream_runs" in sql:
            rows = self._fixture.dream_runs
        elif "FROM dream_proposals" in sql:
            rows = self._fixture.dream_proposals
        elif "FROM memories" in sql:
            rows = self._fixture.memories
        else:
            rows = []
        return _ExecuteResult(rows)


class _VectorStore:
    def __init__(self, vectors: dict[tuple[object, str], list[float]]) -> None:
        self._vectors = vectors

    async def get_vectors(self, *, env_id: object, ids: list[object], vector_name: str) -> dict[object, list[float] | None]:
        return {memory_id: self._vectors.get((memory_id, vector_name)) for memory_id in ids}

    async def close(self) -> None:
        return None


class _Fixture:
    def __init__(self, *, deleted: bool = False, superseded: bool = False) -> None:
        now = dt.datetime.now(dt.UTC)
        self.env = Environment(
            id=uuid4(),
            name="export-env",
            kind=None,
            retention_policy={},
            default_embedding_model_id="test-model",
            created_at=now,
            status="deleted" if deleted else "active",
            deleted_at=now if deleted else None,
        )
        self.memory = Memory(
            id=uuid4(),
            env_id=self.env.id,
            kind="fact",
            status="superseded" if superseded else "active",
            title="hello",
            body="world",
            salience=0.5,
            confidence=0.5,
            pinned=False,
            metadata_={},
            version=2 if superseded else 1,
            created_at=now,
            updated_at=now,
        )
        self.tag = Tag(id=uuid4(), env_id=self.env.id, name="tag")
        self.entity = Entity(
            id=uuid4(),
            env_id=self.env.id,
            kind="thing",
            canonical_name="Entity",
            normalized_name="entity",
            metadata_={},
            created_at=now,
            updated_at=now,
            version=1,
        )
        self.memories = [self.memory]
        self.tags = [self.tag]
        self.entities = [self.entity]
        self.memory_tags: list[object] = []
        self.memory_sources: list[object] = []
        self.memory_lineage: list[object] = []
        self.entity_aliases: list[object] = []
        self.relations: list[object] = []
        self.graph_nodes: list[object] = []
        self.tasks: list[object] = []
        self.grants: list[EnvGrant] = []
        self.dream_runs: list[DreamRun] = []
        self.dream_proposals: list[DreamProposal] = []
        self.vectors = {(self.memory.id, "body"): [0.1, 0.2, 0.3]}

    def add_dream(self) -> None:
        run = DreamRun(
            id=uuid4(),
            env_id=self.env.id,
            mode="maintenance",
            status="succeeded",
            started_at=dt.datetime.now(dt.UTC),
            ended_at=dt.datetime.now(dt.UTC),
            triggered_by="test",
            summary={},
        )
        proposal = DreamProposal(
            id=uuid4(),
            env_id=self.env.id,
            kind="merge",
            status="reviewed",
            payload={},
            llm_failed=False,
            dream_run_id=run.id,
            reviewed_by_agent_id=uuid4(),
            created_at=dt.datetime.now(dt.UTC),
            updated_at=dt.datetime.now(dt.UTC),
        )
        self.dream_runs.append(run)
        self.dream_proposals.append(proposal)

    def add_grant(self) -> None:
        self.grants.append(
            EnvGrant(
                env_id=self.env.id,
                agent_id=uuid4(),
                role="read",
                granted_at=dt.datetime.now(dt.UTC),
            )
        )


def _install_fixture(monkeypatch: pytest.MonkeyPatch, fixture: _Fixture) -> None:
    @asynccontextmanager
    async def fake_session_scope() -> AsyncIterator[_Session]:
        yield _Session(fixture)

    monkeypatch.setattr("memory_mcp.env_ops.export.session_scope", fake_session_scope)
    monkeypatch.setattr(
        "memory_mcp.env_ops.export._default_vector_store",
        lambda: _VectorStore(fixture.vectors),
    )


def _ctx() -> AgentContext:
    return AgentContext(agent_id=uuid4())


async def _run_export(tmp_path: Path, fixture: _Fixture, **kwargs: object):
    request = EnvExportRequest(
        env_id=fixture.env.id,
        format=kwargs.pop("format", ExportFormat.directory),  # type: ignore[arg-type]
        target_path=str(tmp_path / "export"),
        **kwargs,
    )
    return await export_env(request, ctx=_ctx())


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_export_smallest_env_directory_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    _install_fixture(monkeypatch, fixture)

    response = await _run_export(tmp_path, fixture)
    root = Path(response.output_path)

    assert root.is_dir()
    assert len(_jsonl(root / "env.json")) == 1
    assert len(_jsonl(root / "memories.jsonl")) == 1
    ExportManifest.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))
    assert (root / "checksums.sha256").is_file()


@pytest.mark.asyncio
async def test_export_archive_format_creates_tar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    _install_fixture(monkeypatch, fixture)

    response = await _run_export(tmp_path, fixture, format=ExportFormat.archive)
    archive = Path(response.output_path)

    assert archive.suffixes[-2:] == [".tar", ".gz"]
    assert archive.is_file()
    assert not (tmp_path / "export").exists()
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert {"env.json", "memories.jsonl", "manifest.json", "checksums.sha256"} <= names


@pytest.mark.asyncio
async def test_export_excludes_dream_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    fixture.add_dream()
    _install_fixture(monkeypatch, fixture)

    response = await _run_export(tmp_path, fixture)

    assert not (Path(response.output_path) / "dream_runs.jsonl").exists()


@pytest.mark.asyncio
async def test_export_includes_dream_when_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    fixture.add_dream()
    _install_fixture(monkeypatch, fixture)

    response = await _run_export(tmp_path, fixture, include_dream_history=True)
    root = Path(response.output_path)

    assert len(_jsonl(root / "dream_runs.jsonl")) == 1
    proposals = _jsonl(root / "dream_proposals.jsonl")
    assert len(proposals) == 1
    assert proposals[0]["reviewed_by_agent_id"] is None


@pytest.mark.asyncio
async def test_export_excludes_grants_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    fixture.add_grant()
    _install_fixture(monkeypatch, fixture)

    response = await _run_export(tmp_path, fixture)

    assert not (Path(response.output_path) / "grants.jsonl").exists()


@pytest.mark.asyncio
async def test_export_excludes_deleted_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture(deleted=True)
    _install_fixture(monkeypatch, fixture)

    with pytest.raises(NotFoundError) as exc:
        await _run_export(tmp_path, fixture)
    assert exc.value.code == "ENV_DELETED"


@pytest.mark.asyncio
async def test_export_skips_superseded_memory_embedding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture(superseded=True)
    _install_fixture(monkeypatch, fixture)

    response = await _run_export(tmp_path, fixture)
    vectors = _jsonl(Path(response.output_path) / "embeddings" / "memory_vectors.jsonl")

    assert len(vectors) <= len(fixture.memories)
    assert response.manifest.counts["memory_vectors_skipped"] == 1


@pytest.mark.asyncio
async def test_export_manifest_counts_match_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _Fixture()
    _install_fixture(monkeypatch, fixture)

    response = await _run_export(tmp_path, fixture, include_dream_history=True, include_grants=True)
    root = Path(response.output_path)
    counts = response.manifest.counts

    for file_path in root.rglob("*.jsonl"):
        key = file_path.stem
        assert counts[key] == len(_jsonl(file_path))
    assert counts["env"] == len(_jsonl(root / "env.json"))
    assert response.counts_by_kind == {"fact": 1}
