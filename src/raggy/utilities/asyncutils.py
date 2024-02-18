from functools import partial
from typing import Any, Callable, TypeVar

import anyio

T = TypeVar("T")

RAGGY_THREAD_LIMITER: anyio.CapacityLimiter | None = None


def get_thread_limiter():
    global RAGGY_THREAD_LIMITER

    if RAGGY_THREAD_LIMITER is None:
        RAGGY_THREAD_LIMITER = anyio.CapacityLimiter(250)

    return RAGGY_THREAD_LIMITER


async def run_sync_in_worker_thread(
    __fn: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    """Runs a sync function in a new worker thread so that the main thread's event loop
    is not blocked

    Unlike the anyio function, this defaults to a cancellable thread and does not allow
    passing arguments to the anyio function so users can pass kwargs to their function.

    Note that cancellation of threads will not result in interrupted computation, the
    thread may continue running — the outcome will just be ignored.

    Args:
        __fn: The function to run in a worker thread
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function
    """
    call = partial(__fn, *args, **kwargs)
    return await anyio.to_thread.run_sync(
        call, cancellable=True, limiter=get_thread_limiter()
    )
