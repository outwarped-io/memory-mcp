"""Direct service coverage for wave-2 friendly env-name resolution."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from memory_mcp import env_resolve, entities as entities_api, graph as graph_api, provenance as provenance_api
from memory_mcp import relations as relations_api
from memory_mcp.digest import api as digest_api
from memory_mcp.dream import api as dream_api
from memory_mcp.env_ops import delete as env_delete_api
from memory_mcp.env_ops import export as env_export_api
from memory_mcp.env_ops import rename as env_rename_api
from memory_mcp.env_ops import snapshot as env_snapshot_api
from memory_mcp.errors import EnvNotAttachedError, EnvNotFoundError, MemoryMCPError, NotFoundError
from memory_mcp.identity import AgentContext
from memory_mcp.tasks import api as tasks_api
from memory_mcp.env_resolve import _resolve_env_refs
from memory_mcp_schemas.digest import DigestRequest, ResumeRequest
from memory_mcp_schemas.dream import DreamMode, DreamProposalsListRequest, DreamRunRequest, DreamStatusRequest
from memory_mcp_schemas.entities import EntityBrowseRequest, EntityResponse
from memory_mcp_schemas.env_ops import EnvDeleteRequest, EnvExportRequest, EnvRenameRequest, EnvSnapshotRequest, ExportFormat
from memory_mcp_schemas.enums import MemoryKind, MemoryStatus, TaskStatus
from memory_mcp_schemas.graph import EntityNeighborsRequest, MemNeighborsRequest, MemRelatedRequest, MemRelatedResponse
from memory_mcp_schemas.memories import MemoryResponse
from memory_mcp_schemas.provenance import MemLineageRequest, MemLineageResponse, MemSourcesBrowseRequest
from memory_mcp_schemas.relations import RelationBrowseRequest
from memory_mcp_schemas.tasks import TaskCreateRequest, TaskListRequest, TaskListResponse, TaskResponse

FIXED_NOW = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
ENV_ID = UUID('00000000-0000-0000-0000-000000000111')
OTHER_ENV_ID = UUID('00000000-0000-0000-0000-000000000222')
MEMORY_ID = UUID('00000000-0000-0000-0000-000000000333')
ENTITY_ID = UUID('00000000-0000-0000-0000-000000000444')
SNAPSHOT_ID = UUID('00000000-0000-0000-0000-000000000555')
TASK_ID = UUID('00000000-0000-0000-0000-000000000666')
AGENT_ID = UUID('00000000-0000-0000-0000-000000000777')
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH = REPO_ROOT / '.tmp' / 'wave2-env-name-tests'
SCRATCH.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Wave2Case:
    name: str
    make_request: Any
    invoke: Any
    plural: bool = False
    allow_deleted: bool = False


class FakeScalars:
    def __init__(self, rows: list[Any]):
        self._rows = rows

    def all(self) -> list[Any]:
        return list(self._rows)


class FakeExecuteResult:
    def __init__(self, rows: list[Any] | None = None, scalar: Any = None):
        self._rows = list(rows or [])
        self._scalar = scalar

    def scalars(self) -> FakeScalars:
        return FakeScalars(self._rows)

    def all(self) -> list[Any]:
        return list(self._rows)

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def scalar_one(self) -> Any:
        return self._scalar


class FakeSessionContext:
    def __init__(self, session: Any):
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeEmptySession:
    async def execute(self, _stmt: Any) -> FakeExecuteResult:
        return FakeExecuteResult(rows=[])


class FakeGetSession:
    def __init__(self, value: Any):
        self._value = value

    async def get(self, _model: Any, _key: Any) -> Any:
        return self._value


class FakeGraphMemorySession(FakeGetSession):
    async def execute(self, _stmt: Any) -> FakeExecuteResult:
        return FakeExecuteResult(scalar=SimpleNamespace())


class FakeTaskCreateSession:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, value: Any) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        for task in self.added:
            task.id = TASK_ID
            task.version = 1
            task.status = TaskStatus.pending.value
            task.created_at = FIXED_NOW
            task.updated_at = FIXED_NOW

    async def refresh(self, _value: Any) -> None:
        return None

    async def execute(self, _stmt: Any) -> FakeExecuteResult:
        return FakeExecuteResult(rows=[])


class FakeSnapshotSession:
    def add(self, _value: Any) -> None:
        return None

    async def flush(self) -> None:
        return None

    async def refresh(self, value: Any) -> None:
        value.created_at = FIXED_NOW


class FakeGraphStore:
    async def neighbors(self, *_args: Any, **_kwargs: Any) -> tuple[list[Any], None]:
        return [], None


class FixedDateTime:
    @staticmethod
    def now(_tz: Any = None) -> dt.datetime:
        return FIXED_NOW


def _ctx(*, attached: bool) -> AgentContext:
    return AgentContext(
        agent_id=AGENT_ID,
        agent_name='test',
        attached_env_ids=[ENV_ID] if attached else [OTHER_ENV_ID],
    )


def _memory_response(env_id: UUID = ENV_ID) -> MemoryResponse:
    return MemoryResponse(
        id=MEMORY_ID,
        env_id=env_id,
        kind=MemoryKind.fact,
        status=MemoryStatus.active,
        title='wave2',
        body='body',
        tags=[],
        metadata={},
        salience=0.5,
        confidence=0.9,
        pinned=False,
        access_count=0,
        last_accessed_at=None,
        negative_feedback_count=0,
        verified_at=None,
        expires_at=None,
        superseded_by=None,
        version=1,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )


def _entity_response(env_id: UUID = ENV_ID) -> EntityResponse:
    return EntityResponse(
        id=ENTITY_ID,
        env_id=env_id,
        kind='repo',
        canonical_name='memory-mcp',
        normalized_name='memory mcp',
        aliases=[],
        metadata={},
        version=1,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )


def _task_response(env_id: UUID = ENV_ID, *, title: str = 'task') -> TaskResponse:
    return TaskResponse(
        id=TASK_ID,
        env_id=env_id,
        title=title,
        description=None,
        status=TaskStatus.pending,
        priority=50,
        playbook_id=None,
        version=1,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
    )


def _patch_lookup(monkeypatch: pytest.MonkeyPatch, *, env_id: UUID = ENV_ID, deleted_name: str | None = None) -> None:
    async def fake_lookup(name: str, *, include_deleted: bool = False):
        if deleted_name is not None and name == deleted_name:
            if include_deleted:
                return SimpleNamespace(id=env_id)
            raise EnvNotFoundError(name=name)
        if name == 'does-not-exist':
            raise EnvNotFoundError(name=name)
        return SimpleNamespace(id=env_id)

    monkeypatch.setattr(env_resolve, 'get_env_by_name_ci', fake_lookup)


def _patch_attached_rbac(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    def fake_require(_role: str, env_id: UUID | None, ctx: AgentContext) -> None:
        if env_id is None:
            return None
        if env_id not in set(ctx.attached_env_ids):
            raise EnvNotAttachedError(
                f'ENV_NOT_ATTACHED: env {env_id} is not attached to this session',
                env_id=str(env_id),
                attached_env_ids=[str(e) for e in ctx.attached_env_ids],
            )
        return None

    monkeypatch.setattr(module.rbac, 'require', fake_require)


def _deleted_env_error(env_id: UUID) -> NotFoundError:
    exc = NotFoundError(f'environment {env_id} is deleted', env_id=str(env_id))
    exc.code = 'ENV_DELETED'
    return exc


async def _invoke_digest(request: DigestRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, digest_api)

    async def fake_load_digest_inputs(*_args: Any, **_kwargs: Any):
        return digest_api.DigestInputs(
            memories=[], journals=[], latest_digest=None, memory_count=0, entity_count=0, last_journal_ts=None,
        )

    monkeypatch.setattr(digest_api, '_load_digest_inputs', fake_load_digest_inputs)

    async def fake_summarize(**_kwargs: Any):
        return digest_api.DigestSections(brief='brief', active_context='context'), 'template', SimpleNamespace(value='digest_template')

    async def fake_write_digest_memory(**_kwargs: Any):
        return _memory_response(request.env_id)

    monkeypatch.setattr(digest_api, '_summarize_digest', fake_summarize)
    monkeypatch.setattr(digest_api, '_write_digest_memory', fake_write_digest_memory)
    return await digest_api.digest_for_env(request.env_id, since_ts=request.since_ts, ctx=_ctx(attached=attached), settings=SimpleNamespace())


async def _invoke_resume(request: ResumeRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, digest_api)

    async def fake_load_resume_inputs(*_args: Any, **_kwargs: Any):
        return None, [], digest_api.ResumeStats(memory_count=0, entity_count=0, last_journal_ts=None)

    monkeypatch.setattr(digest_api, '_load_resume_inputs', fake_load_resume_inputs)
    return await digest_api.resume_for_env(request.env_id, journal_tail=request.journal_tail, ctx=_ctx(attached=attached))


async def _invoke_dream_run(request: DreamRunRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, dream_api)

    async def fake_run_pairs_with_resources(*_args: Any, **_kwargs: Any):
        return [
            SimpleNamespace(
                env_id=request.env_id,
                mode=DreamMode.decay,
                outcome=dream_api.DreamPassOutcome.done,
                dream_run_id=None,
                summary={},
                last_error=None,
                duration_seconds=0.0,
            )
        ]

    monkeypatch.setattr(dream_api, '_run_pairs_with_resources', fake_run_pairs_with_resources)
    return await dream_api.dream_run(request, ctx=_ctx(attached=attached), settings=SimpleNamespace())


async def _invoke_dream_status(request: DreamStatusRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, dream_api)
    monkeypatch.setattr(dream_api, 'session_scope', lambda: FakeSessionContext(SimpleNamespace()))

    async def fake_last_runs(*_args: Any, **_kwargs: Any):
        return []

    async def fake_open_counts(*_args: Any, **_kwargs: Any):
        return {
            'merge_candidate': 0,
            'promotion_candidate': 0,
            'decay_candidate': 0,
            'decision_conflict_candidate': 0,
        }

    async def fake_heartbeats(*_args: Any, **_kwargs: Any):
        return []

    async def fake_probe(*_args: Any, **_kwargs: Any):
        return {'status': 'ok'}

    monkeypatch.setattr(dream_api, '_load_last_runs_per_mode', fake_last_runs)
    monkeypatch.setattr(dream_api, '_load_open_proposal_counts', fake_open_counts)
    monkeypatch.setattr(dream_api, '_load_dream_heartbeats', fake_heartbeats)
    monkeypatch.setattr(dream_api, '_bounded_llm_probe', fake_probe)
    settings = SimpleNamespace(dream_summarizer='template', llm_backend='test')
    return await dream_api.dream_status(request, ctx=_ctx(attached=attached), settings=settings)


async def _invoke_dream_proposals(request: DreamProposalsListRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, dream_api)
    monkeypatch.setattr(dream_api, 'session_scope', lambda: FakeSessionContext(FakeEmptySession()))
    return await dream_api.dream_proposals_list(request, ctx=_ctx(attached=attached))


async def _invoke_entity_browse(request: EntityBrowseRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, entities_api)
    monkeypatch.setattr(entities_api, 'session_scope', lambda: FakeSessionContext(FakeEmptySession()))
    return await entities_api.entity_browse(request, ctx=_ctx(attached=attached), settings=SimpleNamespace())


async def _invoke_entity_neighbors(request: EntityNeighborsRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, graph_api)
    monkeypatch.setattr(
        graph_api,
        'session_scope',
        lambda: FakeSessionContext(FakeGetSession(SimpleNamespace(env_id=ENV_ID))),
    )

    async def fake_finalize(*_args: Any, **_kwargs: Any):
        return []

    monkeypatch.setattr(graph_api, '_finalize_neighbor_hits', fake_finalize)
    return await graph_api.entity_neighbors(
        request,
        ctx=_ctx(attached=attached),
        settings=SimpleNamespace(graph_backend='postgres', search_fresh_max_wait_seconds=0),
        graph_store=FakeGraphStore(),
    )


async def _invoke_env_delete(request: EnvDeleteRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    def fake_require_delete(env_id: UUID, ctx: AgentContext) -> None:
        if env_id not in set(ctx.attached_env_ids):
            raise EnvNotAttachedError(
                f'ENV_NOT_ATTACHED: env {env_id} is not attached to this session',
                env_id=str(env_id),
                attached_env_ids=[str(e) for e in ctx.attached_env_ids],
            )

    monkeypatch.setattr(env_delete_api, '_require_delete', fake_require_delete)
    monkeypatch.setattr(
        env_delete_api,
        'session_scope',
        lambda: FakeSessionContext(FakeGetSession(SimpleNamespace(status='deleted'))),
    )
    return await env_delete_api.delete_env(request, ctx=_ctx(attached=attached))


async def _invoke_env_export(request: EnvExportRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, env_export_api)
    monkeypatch.setattr(env_export_api, 'session_scope', lambda: FakeSessionContext(SimpleNamespace()))

    async def fake_load_environment(_session: Any, env_id: UUID):
        if deleted:
            raise _deleted_env_error(env_id)
        return SimpleNamespace(id=env_id, name='cdp', default_embedding_model_id='embedder')

    async def fake_stream_table(*_args: Any, **_kwargs: Any) -> int:
        return 0

    async def fake_counts_by_kind(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {}

    async def fake_vectors(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {}

    async def fake_manifest_checksums(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        return {}

    async def fake_compute_checksums(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        return {}

    async def fake_write_checksums_file(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def fake_set_repeatable_read(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(env_export_api, '_set_repeatable_read', fake_set_repeatable_read)
    monkeypatch.setattr(env_export_api, '_load_environment', fake_load_environment)
    monkeypatch.setattr(env_export_api, '_prepare_target_dir', lambda _request: SCRATCH / 'env-export')
    monkeypatch.setattr(env_export_api, '_write_single_row', lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(env_export_api, '_stream_table', fake_stream_table)
    monkeypatch.setattr(env_export_api, '_count_memories_by_kind', fake_counts_by_kind)
    monkeypatch.setattr(env_export_api, '_export_memory_vectors', fake_vectors)
    monkeypatch.setattr(env_export_api, '_write_manifest_and_checksums', fake_manifest_checksums)
    monkeypatch.setattr(env_export_api, '_write_manifest', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_export_api, '_compute_checksums', fake_compute_checksums)
    monkeypatch.setattr(env_export_api, 'write_checksums_file', fake_write_checksums_file)
    monkeypatch.setattr(env_export_api, '_byte_size', lambda _path: 0)
    monkeypatch.setattr(env_export_api, '_read_project_version', lambda: '0.13.0-test')
    monkeypatch.setattr(env_export_api, 'datetime', FixedDateTime)
    return await env_export_api.export_env(request, ctx=_ctx(attached=attached))


async def _invoke_env_rename(request: EnvRenameRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, env_rename_api)

    class FakeRenameSession(FakeGetSession):
        async def get(self, _model: Any, _key: Any) -> Any:
            return SimpleNamespace(
                id=request.env_id,
                name='cdp',
                status='deleted' if deleted else 'active',
                default_embedding_model_id='embedder',
                retention_policy={},
            )

    async def fake_name_available(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def fake_emit(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(env_rename_api, 'session_scope', lambda: FakeSessionContext(FakeRenameSession(None)))
    monkeypatch.setattr(env_rename_api, '_ensure_name_available', fake_name_available)
    monkeypatch.setattr(env_rename_api, '_emit_env_renamed', fake_emit)
    return await env_rename_api.rename_env(request, ctx=_ctx(attached=attached))


async def _invoke_env_snapshot(request: EnvSnapshotRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, env_snapshot_api)
    archive_path = SCRATCH / 'snapshot-archive.tar.gz'
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b'snapshot')

    async def fake_load_active_env(_session: Any, env_id: UUID):
        if deleted:
            raise _deleted_env_error(env_id)
        return SimpleNamespace(id=env_id)

    async def fake_export_env(_request: Any, *, ctx: AgentContext):
        return SimpleNamespace(output_path=str(archive_path), manifest=SimpleNamespace(schema_version='0.8.0'))

    async def fake_sha256_file(_path: Path) -> str:
        return 'checksum'

    async def fake_augment(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def fake_snapshot_tree_size(*_args: Any, **_kwargs: Any) -> int:
        return 0

    monkeypatch.setattr(env_snapshot_api, 'get_settings', lambda: SimpleNamespace(data_root=str(SCRATCH)))
    monkeypatch.setattr(env_snapshot_api, 'session_scope', lambda: FakeSessionContext(FakeSnapshotSession()))
    monkeypatch.setattr(env_snapshot_api, '_load_active_env', fake_load_active_env)
    monkeypatch.setattr(env_snapshot_api, 'export_env', fake_export_env)
    monkeypatch.setattr(env_snapshot_api, '_augment_archive_with_external_lineage', fake_augment)
    monkeypatch.setattr(env_snapshot_api, 'sha256_file', fake_sha256_file)
    monkeypatch.setattr(env_snapshot_api, '_snapshot_tree_size', fake_snapshot_tree_size)
    monkeypatch.setattr(env_snapshot_api, 'uuid4', lambda: SNAPSHOT_ID)
    return await env_snapshot_api.create_snapshot(request, ctx=_ctx(attached=attached))


async def _invoke_mem_lineage(request: MemLineageRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, provenance_api)
    monkeypatch.setattr(
        provenance_api,
        'session_scope',
        lambda: FakeSessionContext(FakeGetSession(SimpleNamespace(env_id=ENV_ID))),
    )

    async def fake_lineage_edges(*_args: Any, **_kwargs: Any):
        return [], False

    def fake_cap(ancestors: list[Any], descendants: list[Any], *, max_edges: int):
        return ancestors, descendants, False

    async def fake_hydrate(_memory_ids: list[UUID], *, statuses: list[str] | None = None):
        return {MEMORY_ID: _memory_response(ENV_ID)}

    monkeypatch.setattr(provenance_api, '_lineage_edges', fake_lineage_edges)
    monkeypatch.setattr(provenance_api, '_apply_lineage_edge_cap', fake_cap)
    monkeypatch.setattr(provenance_api, '_hydrate_memory_responses', fake_hydrate)
    return await provenance_api.memory_lineage(request, ctx=_ctx(attached=attached), settings=SimpleNamespace())


async def _invoke_mem_neighbors(request: MemNeighborsRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, graph_api)
    monkeypatch.setattr(
        graph_api,
        'session_scope',
        lambda: FakeSessionContext(FakeGraphMemorySession(SimpleNamespace(env_id=ENV_ID))),
    )

    async def fake_finalize(*_args: Any, **_kwargs: Any):
        return []

    monkeypatch.setattr(graph_api, '_finalize_neighbor_hits', fake_finalize)
    return await graph_api.memory_neighbors(
        request,
        ctx=_ctx(attached=attached),
        settings=SimpleNamespace(graph_backend='postgres', search_fresh_max_wait_seconds=0),
        graph_store=FakeGraphStore(),
    )


async def _invoke_mem_related(request: MemRelatedRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, graph_api)

    async def fake_resolve_seed(_memory_id: UUID, *, env_id: UUID | None, ctx: AgentContext):
        if env_id is not None and env_id != ENV_ID:
            raise NotFoundError(f'memory {MEMORY_ID} not found', memory_id=str(MEMORY_ID))
        graph_api.rbac.require('read', ENV_ID, ctx)
        return SimpleNamespace(env_id=ENV_ID)

    async def fake_related(*_args: Any, **_kwargs: Any):
        return MemRelatedResponse(hits=[], next_cursor=None, note='ok')

    monkeypatch.setattr(graph_api, '_resolve_seed_memory', fake_resolve_seed)
    monkeypatch.setattr(graph_api, '_memory_related_shared_entity', fake_related)
    monkeypatch.setattr(graph_api, '_memory_related_semantic', fake_related)
    return await graph_api.memory_related(request, ctx=_ctx(attached=attached), settings=SimpleNamespace(), vector_store=None)


async def _invoke_mem_sources(request: MemSourcesBrowseRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, provenance_api)
    monkeypatch.setattr(provenance_api, 'session_scope', lambda: FakeSessionContext(FakeEmptySession()))
    return await provenance_api.memory_sources_browse(request, ctx=_ctx(attached=attached), settings=SimpleNamespace())


async def _invoke_relation_browse(request: RelationBrowseRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, relations_api)
    monkeypatch.setattr(relations_api, 'session_scope', lambda: FakeSessionContext(FakeEmptySession()))
    return await relations_api.relation_browse(request, ctx=_ctx(attached=attached), settings=SimpleNamespace())


async def _invoke_task_create(request: TaskCreateRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, tasks_api)
    monkeypatch.setattr(tasks_api, 'session_scope', lambda: FakeSessionContext(FakeTaskCreateSession()))

    async def fake_ensure_task_graph_node(*_args: Any, **_kwargs: Any):
        return None

    async def fake_enqueue_task(*_args: Any, **_kwargs: Any):
        return None

    def fake_task_to_response(task: Any) -> TaskResponse:
        return TaskResponse(
            id=TASK_ID,
            env_id=task.env_id,
            title=task.title,
            description=task.description,
            status=TaskStatus.pending,
            priority=task.priority,
            playbook_id=task.playbook_id,
            version=1,
            created_at=FIXED_NOW,
            updated_at=FIXED_NOW,
        )

    monkeypatch.setattr(tasks_api, '_ensure_task_graph_node', fake_ensure_task_graph_node)
    monkeypatch.setattr(tasks_api, '_enqueue_task', fake_enqueue_task)
    monkeypatch.setattr(tasks_api, '_task_to_response', fake_task_to_response)
    return await tasks_api.task_create(request, ctx=_ctx(attached=attached), settings=SimpleNamespace())


async def _invoke_task_list(request: TaskListRequest, monkeypatch: pytest.MonkeyPatch, *, attached: bool, deleted: bool = False) -> Any:
    _patch_attached_rbac(monkeypatch, tasks_api)
    monkeypatch.setattr(tasks_api, 'session_scope', lambda: FakeSessionContext(FakeEmptySession()))
    return await tasks_api.task_list(request, ctx=_ctx(attached=attached), settings=SimpleNamespace())


CASES = [
    Wave2Case('DigestRequest', lambda env_id=None, env_name=None: DigestRequest(env_id=env_id, env_name=env_name), _invoke_digest),
    Wave2Case('ResumeRequest', lambda env_id=None, env_name=None: ResumeRequest(env_id=env_id, env_name=env_name), _invoke_resume),
    Wave2Case(
        'DreamRunRequest',
        lambda env_id=None, env_name=None: DreamRunRequest(env_id=env_id, env_name=env_name, modes=[DreamMode.decay], wait=True),
        _invoke_dream_run,
    ),
    Wave2Case('DreamStatusRequest', lambda env_id=None, env_name=None: DreamStatusRequest(env_id=env_id, env_name=env_name), _invoke_dream_status),
    Wave2Case('DreamProposalsListRequest', lambda env_id=None, env_name=None: DreamProposalsListRequest(env_id=env_id, env_name=env_name), _invoke_dream_proposals),
    Wave2Case('EntityBrowseRequest', lambda env_id=None, env_name=None: EntityBrowseRequest(env_ids=[env_id] if env_id else None, env_names=[env_name] if env_name else None), _invoke_entity_browse, plural=True),
    Wave2Case('EntityNeighborsRequest', lambda env_id=None, env_name=None: EntityNeighborsRequest(entity_id=ENTITY_ID, env_id=env_id, env_name=env_name), _invoke_entity_neighbors),
    Wave2Case('EnvDeleteRequest', lambda env_id=None, env_name=None: EnvDeleteRequest(env_id=env_id, env_name=env_name, confirm_destroy=True), _invoke_env_delete, allow_deleted=True),
    Wave2Case('EnvExportRequest', lambda env_id=None, env_name=None: EnvExportRequest(env_id=env_id, env_name=env_name, format=ExportFormat.directory, target_path=str(SCRATCH / 'export-target')), _invoke_env_export, allow_deleted=True),
    Wave2Case('EnvRenameRequest', lambda env_id=None, env_name=None: EnvRenameRequest(env_id=env_id, env_name=env_name, new_name='renamed-env'), _invoke_env_rename, allow_deleted=True),
    Wave2Case('EnvSnapshotRequest', lambda env_id=None, env_name=None: EnvSnapshotRequest(env_id=env_id, env_name=env_name, label='snapshot'), _invoke_env_snapshot, allow_deleted=True),
    Wave2Case('MemLineageRequest', lambda env_id=None, env_name=None: MemLineageRequest(memory_id=MEMORY_ID, env_id=env_id, env_name=env_name), _invoke_mem_lineage, allow_deleted=True),
    Wave2Case('MemNeighborsRequest', lambda env_id=None, env_name=None: MemNeighborsRequest(memory_id=MEMORY_ID, env_id=env_id, env_name=env_name), _invoke_mem_neighbors),
    Wave2Case('MemRelatedRequest', lambda env_id=None, env_name=None: MemRelatedRequest(memory_id=MEMORY_ID, env_id=env_id, env_name=env_name), _invoke_mem_related),
    Wave2Case('MemSourcesBrowseRequest', lambda env_id=None, env_name=None: MemSourcesBrowseRequest(env_ids=[env_id] if env_id else None, env_names=[env_name] if env_name else None), _invoke_mem_sources, plural=True),
    Wave2Case('RelationBrowseRequest', lambda env_id=None, env_name=None: RelationBrowseRequest(env_ids=[env_id] if env_id else None, env_names=[env_name] if env_name else None), _invoke_relation_browse, plural=True),
    Wave2Case('TaskCreateRequest', lambda env_id=None, env_name=None: TaskCreateRequest(env_id=env_id, env_name=env_name, title='task'), _invoke_task_create),
    Wave2Case('TaskListRequest', lambda env_id=None, env_name=None: TaskListRequest(env_id=env_id, env_name=env_name), _invoke_task_list),
]


def _assert_resolved(request: Any, *, plural: bool) -> None:
    if plural:
        assert request.env_ids == [ENV_ID]
        assert request.env_names is None
    else:
        assert request.env_id == ENV_ID
        assert request.env_name is None


async def _assert_same_failure(
    case: Wave2Case,
    monkeypatch: pytest.MonkeyPatch,
    *,
    direct_request: Any,
    resolved_request: Any,
    deleted: bool = False,
) -> None:
    with pytest.raises(Exception) as direct_exc:
        await case.invoke(direct_request, monkeypatch, attached=False, deleted=deleted)
    with pytest.raises(type(direct_exc.value)) as resolved_exc:
        await case.invoke(resolved_request, monkeypatch, attached=False, deleted=deleted)
    if isinstance(direct_exc.value, MemoryMCPError):
        assert isinstance(resolved_exc.value, MemoryMCPError)
        assert resolved_exc.value.code == direct_exc.value.code
        assert resolved_exc.value.details == direct_exc.value.details
    else:
        assert str(resolved_exc.value) == str(direct_exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.name)
async def test_env_name_roundtrip_matches_env_id(case: Wave2Case, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_lookup(monkeypatch)
    resolved_request = await _resolve_env_refs(case.make_request(env_name='cdp'))
    _assert_resolved(resolved_request, plural=case.plural)

    direct_request = case.make_request(env_id=ENV_ID)
    resolved_result = await case.invoke(resolved_request, monkeypatch, attached=True)
    direct_result = await case.invoke(direct_request, monkeypatch, attached=True)

    assert resolved_result == direct_result


@pytest.mark.asyncio
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.name)
async def test_both_env_id_and_env_name_rejected(case: Wave2Case) -> None:
    with pytest.raises(ValidationError):
        case.make_request(env_id=ENV_ID, env_name='cdp')


@pytest.mark.asyncio
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.name)
async def test_unknown_env_name_raises_not_found(case: Wave2Case, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_lookup(monkeypatch)
    with pytest.raises(EnvNotFoundError) as exc_info:
        await _resolve_env_refs(case.make_request(env_name='does-not-exist'))
    assert exc_info.value.code == 'ENV_NOT_FOUND'
    assert exc_info.value.details == {'name': 'does-not-exist'}


@pytest.mark.asyncio
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.name)
async def test_unattached_env_name_matches_unattached_env_id(case: Wave2Case, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_lookup(monkeypatch)
    resolved_request = await _resolve_env_refs(case.make_request(env_name='cdp'))
    await _assert_same_failure(
        case,
        monkeypatch,
        direct_request=case.make_request(env_id=ENV_ID),
        resolved_request=resolved_request,
    )


ALLOW_DELETED_CASES = [case for case in CASES if case.allow_deleted]


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ALLOW_DELETED_CASES, ids=lambda case: case.name)
async def test_deleted_env_name_requires_allow_deleted(case: Wave2Case, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_lookup(monkeypatch, deleted_name='deleted-env')

    with pytest.raises(EnvNotFoundError):
        await _resolve_env_refs(case.make_request(env_name='deleted-env'))

    resolved_request = await _resolve_env_refs(
        case.make_request(env_name='deleted-env'),
        allow_deleted=True,
    )
    _assert_resolved(resolved_request, plural=case.plural)

    direct_request = case.make_request(env_id=ENV_ID)
    try:
        resolved_result = await case.invoke(resolved_request, monkeypatch, attached=True, deleted=True)
    except Exception as resolved_exc:
        with pytest.raises(type(resolved_exc)) as direct_exc:
            await case.invoke(direct_request, monkeypatch, attached=True, deleted=True)
        if isinstance(resolved_exc, MemoryMCPError):
            assert isinstance(direct_exc.value, MemoryMCPError)
            assert direct_exc.value.code == resolved_exc.code
            assert direct_exc.value.details == resolved_exc.details
    else:
        direct_result = await case.invoke(direct_request, monkeypatch, attached=True, deleted=True)
        assert resolved_result == direct_result
