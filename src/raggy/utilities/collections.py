import itertools
from typing import Any, Callable, Generator, Iterable, TypeVar

T = TypeVar("T")


def distinct(
    iterable: Iterable[T],
    key: Callable[[T], Any] = (lambda i: i),
) -> Generator[T, None, None]:
    """Yield distinct items from an iterable.

    Args:
        iterable: The iterable to filter
        key: A function to compute a key for each item

    Yields:
        Distinct items from the iterable

    Example:
        Dedupe a list of Pydantic models by a key:
        ```python
        from pydantic import BaseModel
        from raggy.utilities.collections import distinct

        class MyModel(BaseModel):
            id: int
            name: str

        items = [
            MyModel(id=1, name="foo"),
            MyModel(id=2, name="bar"),
            MyModel(id=1, name="baz"),
        ]

        deduped = list(distinct(items, key=lambda i: i.id))

        assert deduped == [
            MyModel(id=1, name="foo"),
            MyModel(id=2, name="bar"),
        ]
        ```
    """
    seen: set = set()
    for item in iterable:
        if key(item) in seen:
            continue
        seen.add(key(item))
        yield item


def batched(
    iterable: Iterable[T], size: int, size_fn: Callable[[T], int] | None = None
) -> Generator[tuple[T, ...], None, None]:
    """Yield batches of items from an iterable.

    If size_fn is not provided, then the batch size will be determined by the
    number of items in the batch.

    If size_fn is provided, then it will be used
    to compute the batch size. Note that if a single item is larger than the
    batch size, it will be returned as a batch of its own.

    Args:
        iterable: The iterable to batch
        size: The size of the batch
        size_fn: A function to compute the size of an item in the iterable

    Yields:
        A batch of items from the iterable

    Example:
        Batch a list of strings by the number of characters:
        ```python
        from raggy.utilities.collections import batched

        items = [
            "foo",
            "bar",
            "baz",
            "qux",
            "quux",
            "corge",
            "grault",
            "garply",
            "waldo",
            "fred",
            "plugh",
            "xyzzy",
            "thud",
        ]

        batches = list(batched(items, size=10, size_fn=len))

        assert batches == [
            ('foo', 'bar', 'baz'),
            ('qux', 'quux'),
            ('corge',),
            ('grault',),
            ('garply',),
            ('waldo', 'fred'),
            ('plugh', 'xyzzy'),
            ('thud',)
        ]
        ```
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
