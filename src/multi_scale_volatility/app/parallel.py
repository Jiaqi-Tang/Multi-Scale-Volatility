"""Small process-pool helpers shared by long-running research stages."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
import os
from typing import TypeVar

TArg = TypeVar("TArg")
TResult = TypeVar("TResult")


def effective_worker_count(max_workers: int | None, default_cap: int = 4) -> int:
    if max_workers is not None:
        return max_workers
    return min(default_cap, os.cpu_count() or 1)


def process_pool_map(
    worker: Callable[[TArg], TResult],
    worker_args: Iterable[TArg],
    *,
    max_workers: int | None = None,
    default_cap: int = 4,
) -> Iterator[tuple[int, TResult]]:
    effective_max_workers = effective_worker_count(max_workers, default_cap)
    with ProcessPoolExecutor(max_workers=effective_max_workers) as executor:
        yield from enumerate(executor.map(worker, worker_args), start=1)
