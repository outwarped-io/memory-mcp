"""Tests for v0.8 env ops manifest schemas."""

from __future__ import annotations

import datetime as dt
from uuid import uuid4

import pytest
from memory_mcp_schemas.env_ops import ExportManifest, IncludeFlags, MemoryVectorRecord, RemapTable, SourceMetadata
from pydantic import ValidationError


def _manifest() -> ExportManifest:
    return ExportManifest(
        memory_mcp_version="0.8.0",
        source=SourceMetadata(
            env_id=uuid4(),
            env_name="prod",
            default_embedding_model_id="model-a",
            instance_fingerprint="fingerprint",
        ),
        exported_at=dt.datetime(2026, 5, 13, 3, 24, tzinfo=dt.UTC),
        exported_by_agent="agent",
        include_flags=IncludeFlags(embeddings=True, provenance=True, dream_history=False, grants=False),
        counts={"memories": 10},
        checksums={"memories.jsonl": "abc"},
    )


def test_manifest_roundtrip() -> None:
    manifest = _manifest()

    parsed = ExportManifest.model_validate_json(manifest.model_dump_json())

    assert parsed == manifest


def test_manifest_rejects_invalid_schema_version() -> None:
    payload = _manifest().model_dump()
    payload["schema_version"] = "0.9.0"

    with pytest.raises(ValidationError):
        ExportManifest.model_validate(payload)


def test_memory_vector_record_roundtrip() -> None:
    record = MemoryVectorRecord(
        memory_id=uuid4(),
        memory_version=1,
        model_id="model-a",
        vector_name="body",
        dimension=3,
        vector=[0.1, 0.2, 0.3],
    )

    parsed = MemoryVectorRecord.model_validate_json(record.model_dump_json())

    assert parsed == record


def test_memory_vector_record_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValidationError):
        MemoryVectorRecord(
            memory_id=uuid4(),
            memory_version=1,
            model_id="model-a",
            vector_name="body",
            dimension=3,
            vector=[0.1, 0.2],
        )


def test_memory_vector_record_rejects_unknown_vector_name() -> None:
    with pytest.raises(ValidationError):
        MemoryVectorRecord(
            memory_id=uuid4(),
            memory_version=1,
            model_id="model-a",
            vector_name="summary",  # type: ignore[arg-type]
            dimension=1,
            vector=[0.1],
        )


def test_remap_table_roundtrip() -> None:
    src_memory = uuid4()
    dst_memory = uuid4()
    src_tag = uuid4()
    dst_tag = uuid4()
    table = RemapTable(memories={src_memory: dst_memory}, tags={src_tag: dst_tag})

    parsed = RemapTable.model_validate_json(table.model_dump_json())

    assert parsed == table
