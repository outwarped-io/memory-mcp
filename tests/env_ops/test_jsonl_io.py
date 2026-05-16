"""Tests for JSONL streaming helpers."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from memory_mcp.env_ops._io import JsonlReader, stream_jsonl


@pytest.fixture
def artifact_dir() -> Iterator[Path]:
    path = Path("tests/env_ops/.artifacts/jsonl")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path)


@pytest.mark.asyncio
async def test_stream_jsonl_round_trips_10k_rows(artifact_dir: Path) -> None:
    path = artifact_dir / "rows.jsonl"
    rows = [{"id": i, "body": f"body {i}"} for i in range(10_000)]

    written = stream_jsonl(rows, path)
    reader = JsonlReader(path)
    read_rows = [row async for row in reader]

    assert written == 10_000
    assert reader.count == 10_000
    assert read_rows == rows


@pytest.mark.asyncio
async def test_jsonl_reader_skips_empty_trailing_lines(artifact_dir: Path) -> None:
    path = artifact_dir / "trailing.jsonl"
    path.write_text('{"id": 1}\n\n\n', encoding="utf-8")

    reader = JsonlReader(path)
    rows = [row async for row in reader]

    assert rows == [{"id": 1}]
    assert reader.count == 1
