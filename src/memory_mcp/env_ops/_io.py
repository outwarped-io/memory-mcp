"""Streaming JSONL helpers for environment operations."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel

try:  # pragma: no cover - exercised only when optional dependency is installed
    import aiofiles
except ImportError:  # pragma: no cover - default test environment
    aiofiles = None  # type: ignore[assignment]


class JsonlWriter:
    """Synchronous JSONL writer for dicts and Pydantic models."""

    def __init__(self, file_or_path: str | Path | TextIO) -> None:
        self._file_or_path = file_or_path
        self._file: TextIO | None = None
        self._owns_file = False
        self._count = 0

    def __enter__(self) -> JsonlWriter:
        if isinstance(self._file_or_path, str | Path):
            self._file = Path(self._file_or_path).open("w", encoding="utf-8")
            self._owns_file = True
        else:
            self._file = self._file_or_path
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._owns_file and self._file is not None:
            self._file.close()
        self._file = None

    @property
    def count(self) -> int:
        """Number of rows written."""

        return self._count

    def write(self, obj: dict[str, Any] | BaseModel) -> None:
        """Serialize ``obj`` as a single JSONL row."""

        if self._file is None:
            raise RuntimeError("JsonlWriter must be used as a context manager")
        line = obj.model_dump_json() if isinstance(obj, BaseModel) else json.dumps(obj, default=str, ensure_ascii=False)
        self._file.write(line)
        self._file.write("\n")
        self._count += 1


class JsonlReader:
    """Async iterator over parsed JSONL dictionaries."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file: Any | None = None
        self._count = 0
        self._closed = False

    @property
    def count(self) -> int:
        """Number of rows yielded so far."""

        return self._count

    def __aiter__(self) -> JsonlReader:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._closed:
            raise StopAsyncIteration
        await self._ensure_open()
        assert self._file is not None

        while True:
            line = await self._readline()
            if line == "":
                await self._close()
                raise StopAsyncIteration
            if not line.strip():
                continue
            self._count += 1
            return json.loads(line)

    async def _ensure_open(self) -> None:
        if self._file is not None:
            return
        if aiofiles is not None:
            self._file = await aiofiles.open(self._path, encoding="utf-8")
        else:
            self._file = await asyncio.to_thread(self._path.open, encoding="utf-8")

    async def _readline(self) -> str:
        assert self._file is not None
        if aiofiles is not None:
            return await self._file.readline()
        return await asyncio.to_thread(self._file.readline)

    async def _close(self) -> None:
        if self._file is None:
            self._closed = True
            return
        if aiofiles is not None:
            await self._file.close()
        else:
            await asyncio.to_thread(self._file.close)
        self._file = None
        self._closed = True


def stream_jsonl(rows: Iterable[Any], path: Path) -> int:
    """Write ``rows`` to ``path`` and return the number of rows written."""

    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)
        return writer.count


__all__ = ["JsonlReader", "JsonlWriter", "stream_jsonl"]
