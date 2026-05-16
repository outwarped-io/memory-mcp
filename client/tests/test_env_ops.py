"""Happy-path and validation coverage for the env_ops client namespace."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError as PydanticValidationError

from memory_mcp_client.errors import InvalidInputError
from memory_mcp_schemas.env_ops import (
    ArchiveVersionDecision,
    DiffGranularity,
    EnvCloneRequest,
    EnvCloneResponse,
    EnvDiffRequest,
    EnvExportRequest,
    EnvImportReport,
    EnvImportRequest,
    EnvMergeRequest,
    EnvMergeResponse,
    EnvMigrateRequest,
    EnvMigrateResponse,
    EnvRenameRequest,
    EnvRestoreRequest,
    EnvSnapshotRequest,
    ExportFormat,
    ImportMode,
    MigrationMode,
    RestoreMode,
)


pytestmark = pytest.mark.asyncio

TS = "2026-05-13T00:00:00Z"


def export_payload(env_id) -> dict:
    return {
        "manifest": {
            "schema_version": "0.8.0",
            "memory_mcp_version": "0.8.0",
            "source": {
                "env_id": str(env_id),
                "env_name": "source",
                "default_embedding_model_id": "model-1",
                "instance_fingerprint": "fingerprint",
            },
            "exported_at": TS,
            "exported_by_agent": None,
            "include_flags": {
                "embeddings": True,
                "provenance": True,
                "dream_history": False,
                "grants": False,
            },
            "counts": {"memories": 1},
            "checksums": {"manifest.json": "abc"},
        },
        "output_path": "exports/source.memarchive.tar.gz",
        "byte_size": 128,
        "counts_by_kind": {"fact": 1},
    }


def import_report_payload(target_env_id) -> dict:
    return {
        "target_env_id": str(target_env_id),
        "dry_run": False,
        "mode": "merge",
        "counts": {"memories": 2},
        "conflicts": {},
        "sample_conflicts": {},
        "remap_table_size": 7,
        "pending_vector_rebuild": 0,
        "re_embed_count": 0,
        "entity_merges_performed": 0,
        "archive_version_decision": "accept",
    }


async def test_env_ops_export_calls_correct_tool(client, fake_session) -> None:
    env_id = uuid4()
    request = EnvExportRequest(
        env_id=env_id,
        format=ExportFormat.archive,
        target_path="exports/source.memarchive.tar.gz",
    )
    fake_session.set_response("env_export_", export_payload(env_id))

    out = await client.env_ops.export(request)

    assert fake_session.calls == [
        ("env_export_", {"request": request.model_dump(mode="json")})
    ]
    assert out.output_path == "exports/source.memarchive.tar.gz"


async def test_env_ops_import_returns_typed_response(client, fake_session) -> None:
    target_env_id = uuid4()
    request = EnvImportRequest(
        source_path="exports/source.memarchive.tar.gz",
        target_env_id=target_env_id,
        mode=ImportMode.merge,
        dry_run=False,
    )
    fake_session.set_response("env_import_", import_report_payload(target_env_id))

    out = await client.env_ops.import_(request)

    assert isinstance(out, EnvImportReport)
    assert out.target_env_id == target_env_id
    assert out.archive_version_decision == ArchiveVersionDecision.accept


async def test_env_ops_diff_validates_request(fake_session) -> None:
    with pytest.raises(PydanticValidationError):
        EnvDiffRequest(env_a_id=uuid4())

    assert fake_session.calls == []


async def test_env_ops_clone_passes_through_options(client, fake_session) -> None:
    src_env_id = uuid4()
    dst_env_id = uuid4()
    request = EnvCloneRequest(
        src_env_id=src_env_id,
        new_name="clone",
        include_embeddings=False,
        lineage_depth=3,
        include_referenced_entities=False,
    )
    fake_session.set_response(
        "env_clone_",
        {
            "dst_env_id": str(dst_env_id),
            "dst_env_name": "clone",
            "new_env_id": str(dst_env_id),
            "counts": {"memories": 1},
            "closure_inclusions": {"lineage": 2},
            "pending_vector_rebuild": 0,
            "remap_table_size": 3,
        },
    )

    out = await client.env_ops.clone(request)

    name, args = fake_session.calls[0]
    assert name == "env_clone_"
    assert args["request"]["include_embeddings"] is False
    assert args["request"]["lineage_depth"] == 3
    assert args["request"]["include_referenced_entities"] is False
    assert isinstance(out, EnvCloneResponse)
    assert out.remap_table_size == 3


async def test_env_ops_merge_returns_remap_table_size(client, fake_session) -> None:
    src_env_id = uuid4()
    dst_env_id = uuid4()
    request = EnvMergeRequest(src_env_id=src_env_id, dst_env_id=dst_env_id)
    fake_session.set_response(
        "env_merge_",
        {
            "dst_env_id": str(dst_env_id),
            "src_env_id": str(src_env_id),
            "delete_src_after": True,
            "counts": {"memories": 2},
            "entity_merges_performed": 1,
            "external_refs_rewritten": 0,
            "pending_vector_rebuild": 0,
            "remap_table_size": 9,
        },
    )

    out = await client.env_ops.merge(request)

    assert isinstance(out, EnvMergeResponse)
    assert out.remap_table_size == 9


async def test_env_ops_migrate_with_failures(client, fake_session) -> None:
    src_env_id = uuid4()
    dst_env_id = uuid4()
    request = EnvMigrateRequest(
        src_env_id=src_env_id,
        dst_env_id=dst_env_id,
        mode=MigrationMode.move,
        fail_fast=False,
    )
    failed_memory_id = uuid4()
    fake_session.set_response(
        "env_migrate_",
        {
            "src_env_id": str(src_env_id),
            "dst_env_id": str(dst_env_id),
            "mode": "move",
            "attempted": 2,
            "succeeded": 1,
            "failed": 1,
            "remap": {},
            "failures": [
                {
                    "id": str(failed_memory_id),
                    "message": "blocked",
                    "memory_id": str(failed_memory_id),
                    "code": "INVALID_INPUT",
                }
            ],
            "truncated": False,
            "pending_vector_rebuild": 0,
        },
    )

    out = await client.env_ops.migrate(request)

    assert isinstance(out, EnvMigrateResponse)
    assert out.failed == 1
    assert out.failures[0].memory_id == failed_memory_id


async def test_env_ops_snapshot_label_required(fake_session) -> None:
    with pytest.raises(PydanticValidationError):
        EnvSnapshotRequest(env_id=uuid4())

    assert fake_session.calls == []


async def test_env_ops_restore_in_place_requires_confirm_destroy(
    client, fake_session
) -> None:
    request = EnvRestoreRequest(
        snapshot_id=uuid4(),
        mode=RestoreMode.replace_env_in_place,
        confirm_destroy=False,
    )
    fake_session.set_error("env_restore_", "[INVALID_INPUT] confirm_destroy required")

    with pytest.raises(InvalidInputError):
        await client.env_ops.restore(request)

    assert fake_session.calls[0][0] == "env_restore_"
    assert fake_session.calls[0][1]["request"]["confirm_destroy"] is False


async def test_env_ops_delete_requires_confirm_destroy(fake_session) -> None:
    with pytest.raises(PydanticValidationError):
        client_request = {"env_id": uuid4()}
        _ = client_request
        from memory_mcp_schemas.env_ops import EnvDeleteRequest

        EnvDeleteRequest(**client_request)

    assert fake_session.calls == []


async def test_env_ops_rename_at_least_one_field_required(
    client, fake_session
) -> None:
    request = EnvRenameRequest(env_id=uuid4())
    fake_session.set_error("env_rename_", "[INVALID_INPUT] NOTHING_TO_RENAME")

    with pytest.raises(InvalidInputError):
        await client.env_ops.rename(request)

    assert fake_session.calls[0][0] == "env_rename_"
