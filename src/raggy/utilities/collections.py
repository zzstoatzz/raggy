import itertools
from typing import Any, Callable, Generator, Iterable, Set, TypeVar

T = TypeVar("T")


def distinct(
    iterable: Iterable[T],
    key: Callable[[T], Any] = (lambda i: i),
) -> Generator[T, None, None]:
    seen: Set = set()
    for item in iterable:
        if key(item) in seen:
            continue
        seen.add(key(item))
        yield item


def batched(
    iterable: Iterable[T], size: int, size_fn: Callable[[T], int] | None = None
) -> Generator[tuple[T, ...], None, None]:
    """
    If size_fn is not provided, then the batch size will be determined by the
    number of items in the batch.

    If size_fn is provided, then it will be used
    to compute the batch size. Note that if a single item is larger than the
    batch size, it will be returned as a batch of its own.
    """
    if size_fn is None:
        it = iter(iterable)
        while True:
            batch_tuple = tuple(itertools.islice(it, size))
            if not batch_tuple:
                break
            yield batch_tuple
    else:
        batch_list: list[T] = []
        batch_size = 0
        for item in iterable:
            item_size = size_fn(item)
            if batch_size + item_size > size and batch_list:
                yield tuple(batch_list)
                batch_list = []
                batch_size = 0
            batch_list.append(item)
            batch_size += item_size
        if batch_list:
            yield tuple(batch_list)
