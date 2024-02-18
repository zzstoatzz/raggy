import asyncio
import math
import os
from pathlib import Path


def multi_glob(
    directory: str | None = None,
    keep_globs: list[str] | None = None,
    drop_globs: list[str] | None = None,
) -> list[Path]:
    """
    Return a list of all files in the given directory that match the
    patterns in keep_globs and do not match the patterns in drop_globs.
    The patterns are defined using glob syntax.

    Args:
        directory: The directory to search in. If not provided, the current
            working directory is used.
        keep_globs: A list of glob patterns to keep.
        drop_globs: A list of glob patterns to drop.

    Returns:
        A list of Path objects representing the files that match the given
        patterns.

    Raises:
        ValueError: If the directory does not exist.

    Example:
        Get all files (except those in the .git directory) in the current directory
        ```python
        from raggy.utilities.filesystem import multi_glob

        files = multi_glob() # .git files are excluded by default (unless drop_globs is set)
        ```

        Get all python files in the current directory
        ```python
        from raggy.utilities.filesystem import multi_glob

        files = multi_glob(keep_globs=["**/*.py"])
        ```

        Get all files except those in any `__pycache__` directories
        ```python
        from raggy.utilities.filesystem import multi_glob

        files = multi_glob(drop_globs=["**/__pycache__/**/*"])
        ```
    """
    keep_globs = keep_globs or ["**/*"]
    drop_globs = drop_globs or [".git/**/*"]
    directory_path = Path(directory) if directory else Path.cwd()

    if not directory_path.is_dir():
        raise ValueError(f"'{directory}' is not a directory.")

    def files_from_globs(globs: list[str]) -> set[Path]:
        return {
            file
            for pattern in globs
            for file in directory_path.glob(pattern)
            if file.is_file()
        }

    matching_files = files_from_globs(keep_globs) - files_from_globs(drop_globs)
    return [file.relative_to(directory_path) for file in matching_files]


def get_open_file_limit() -> int:
    """Get the maximum number of open files allowed for the current process.

    Returns:
        The maximum number of open files allowed for the current process.
    """
    try:
        if os.name == "nt":
            import ctypes

            return ctypes.cdll.ucrtbase._getmaxstdio()
        else:
            import resource

            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            return soft_limit
    except Exception:
        return 200


OPEN_FILE_CONCURRENCY = asyncio.Semaphore(math.floor(get_open_file_limit() / 2))
