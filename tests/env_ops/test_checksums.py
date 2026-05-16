"""Tests for checksum helpers."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from memory_mcp.env_ops._checksums import sha256_file, verify_checksums_file, write_checksums_file


@pytest.fixture
def artifact_dir() -> Iterator[Path]:
    path = Path("tests/env_ops/.artifacts/checksums")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path)


@pytest.mark.asyncio
async def test_sha256_file_matches_known_fixture(artifact_dir: Path) -> None:
    path = artifact_dir / "hello.txt"
    path.write_text("hello", encoding="utf-8")

    digest = await sha256_file(path)

    assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


@pytest.mark.asyncio
async def test_checksums_file_round_trip_verifies(artifact_dir: Path) -> None:
    payload = artifact_dir / "payload.txt"
    payload.write_text("hello", encoding="utf-8")
    digest = await sha256_file(payload)
    checksums_path = artifact_dir / "SHA256SUMS"

    await write_checksums_file({"payload.txt": digest}, checksums_path)

    assert await verify_checksums_file(checksums_path, artifact_dir) is True


@pytest.mark.asyncio
async def test_checksums_file_detects_mismatch(artifact_dir: Path) -> None:
    payload = artifact_dir / "payload.txt"
    payload.write_text("hello", encoding="utf-8")
    checksums_path = artifact_dir / "SHA256SUMS"
    await write_checksums_file({"payload.txt": "0" * 64}, checksums_path)

    assert await verify_checksums_file(checksums_path, artifact_dir) is False
