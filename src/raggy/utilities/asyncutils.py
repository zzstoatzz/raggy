from functools import partial
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

import anyio
from anyio import create_task_group, to_thread

from raggy import settings

P = ParamSpec("P")
T = TypeVar("T")

_raggy_thread_limiter: anyio.CapacityLimiter | None = None


def get_thread_limiter():
    global _raggy_thread_limiter

    if _raggy_thread_limiter is None:
        _raggy_thread_limiter = anyio.CapacityLimiter(250)

    return _raggy_thread_limiter


async def run_sync_in_worker_thread(
    __fn: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    """Runs a sync function in a new worker thread so that the main thread's event loop
    is not blocked."""
    call = partial(__fn, *args, **kwargs)
    return await to_thread.run_sync(
        call, cancellable=True, limiter=get_thread_limiter()
    )


async def run_concurrent_tasks(
    tasks: list[Awaitable[T]],
    max_concurrent: int = settings.max_concurrent_tasks,
) -> list[T]:
    """Run multiple tasks concurrently with a limit on concurrent execution.

    Args:
        tasks: List of awaitables to execute
        max_concurrent: Maximum number of tasks to run concurrently
    """
    semaphore = anyio.Semaphore(max_concurrent)
    results: list[T] = []

    async def _run_task(task: Awaitable[T]):
        async with semaphore:
            result = await task
            results.append(result)

    async with create_task_group() as tg:
        for task in tasks:
            tg.start_soon(_run_task, task)

    return results
