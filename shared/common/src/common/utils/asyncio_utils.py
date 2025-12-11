import asyncio
from typing import Awaitable, TypeVar, List

T = TypeVar("T")


async def gather_parallel(*coros: Awaitable[T], max_concurrency: int = 32) -> List[T]:
    """
    Like asyncio.gather but limits concurrency using a semaphore.
    Accepts varargs instead of a single iterable.
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def run_with_sem(coro: Awaitable[T]):
        async with sem:
            return await coro

    if not coros:
        return []

    return await asyncio.gather(*[run_with_sem(c) for c in coros])
