"""SHA-256 checksum helpers for environment operation artifacts."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

_CHUNK_SIZE = 64 * 1024


def _sha256_file_sync(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


async def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest for ``path`` using chunked reads."""

    return await asyncio.to_thread(_sha256_file_sync, path)


def _write_checksums_file_sync(checksums: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rel_path, digest in sorted(checksums.items()):
            f.write(f"{digest}  {rel_path}\n")


async def write_checksums_file(checksums: dict[str, str], path: Path) -> None:
    """Write a BSD-style checksums file."""

    await asyncio.to_thread(_write_checksums_file_sync, checksums, path)


async def verify_checksums_file(checksums_path: Path, base_dir: Path) -> bool:
    """Return True iff every checksum entry matches its file under ``base_dir``."""

    try:
        lines = await asyncio.to_thread(checksums_path.read_text, encoding="utf-8")
    except FileNotFoundError:
        return False

    for raw_line in lines.splitlines():
        if not raw_line.strip():
            continue
        try:
            expected, rel_path = raw_line.split("  ", 1)
        except ValueError:
            return False
        file_path = base_dir / rel_path
        if not file_path.is_file():
            return False
        actual = await sha256_file(file_path)
        if actual != expected:
            return False
    return True


__all__ = ["sha256_file", "verify_checksums_file", "write_checksums_file"]
