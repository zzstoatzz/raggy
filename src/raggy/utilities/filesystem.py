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
