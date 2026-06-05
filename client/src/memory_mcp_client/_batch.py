"""Batch helpers for fan-out SDK operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchFailure[T]:
    """One failed item from a bounded batch run."""

    index: int
    item: T
    exception: BaseException


@dataclass
class BatchResult[T, R]:
    """Split batch outcome for resumable callers."""

    successes: list[R] = field(default_factory=list)
    failures: list[BatchFailure[T]] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.successes)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def is_partial(self) -> bool:
        return bool(self.failures) and bool(self.successes)


async def run_bounded[T, R](
    items: list[T],
    handler: Callable[[T], Awaitable[R]],
    *,
    max_concurrency: int = 8,
) -> BatchResult[T, R]:
    """Execute ``handler(item)`` for each item with bounded concurrency."""

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    sem = asyncio.Semaphore(max_concurrency)
    result: BatchResult[T, R] = BatchResult()
    outcomes: list[tuple[int, R | None, BaseException | None]] = [
        (i, None, None) for i in range(len(items))
    ]

    async def _one(i: int, item: T) -> None:
        async with sem:
            try:
                response = await handler(item)
                outcomes[i] = (i, response, None)
            except BaseException as exc:  # noqa: BLE001
                outcomes[i] = (i, None, exc)

    await asyncio.gather(*(_one(i, item) for i, item in enumerate(items)))

    for i, response, exc in outcomes:
        if exc is not None:
            result.failures.append(BatchFailure(index=i, item=items[i], exception=exc))
            continue
        result.successes.append(cast(R, response))

    return result
