"""Loaders for GitHub."""

import asyncio
import asyncio.subprocess
import functools
import os
import re
from pathlib import Path

import aiofiles
import chardet
import httpx
from gh_util.types import GitHubComment, GitHubIssue
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from raggy.documents import Document, document_to_excerpts
from raggy.loaders import Loader
from raggy.utilities.filesystem import OPEN_FILE_CONCURRENCY, multi_glob
from raggy.utilities.text import rm_html_comments, rm_text_after


async def read_file_with_chardet(file_path: str | Path, errors: str = "replace") -> str:
    """Read a file with chardet to detect encoding."""
    async with aiofiles.open(file_path, "rb") as f:
        content = await f.read()
        encoding = chardet.detect(content)["encoding"]

    async with aiofiles.open(file_path, "r", encoding=encoding, errors=errors) as f:
        text = await f.read()
    return text


class GitHubIssueLoader(Loader):
    """Loader for GitHub issues in a given repository.

    **Beware** the [GitHub API rate limit](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api).

    Attributes:
        repo: The GitHub repository in the format 'owner/repo'.
        n_issues: The number of issues to load.
        include_comments: Whether to include comments in the issues.
        ignore_body_after: The text to ignore in the issue body.
        ignore_users: A list of users to ignore.
        use_GH_token: Whether to use the `GITHUB_TOKEN` environment variable for authentication (recommended).
    """

    source_type: str = "github issue"

    repo: str = Field(...)
    n_issues: int = Field(default=50)

    include_comments: bool = Field(default=False)
    ignore_body_after: str = Field(default="### Checklist")
    ignore_users: list[str] = Field(default_factory=list)
    use_GH_token: bool = Field(default=False)

    request_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def auth_headers(self) -> Self:
        self.request_headers.update({"Accept": "application/vnd.github.v3+json"})
        if self.use_GH_token and (token := os.getenv("GITHUB_TOKEN")):
            self.request_headers["Authorization"] = f"Bearer {token}"
        return self

    @staticmethod
    @functools.lru_cache(maxsize=2048)
    async def _get_issue_comments(
        repo: str,
        request_header_items: tuple[tuple[str, str]],
        issue_number: int,
        per_page: int = 100,
    ) -> list[GitHubComment]:
        """
        Get a list of all comments for the given issue.

        Returns:
            A list of dictionaries, each representing a comment.
        """
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
        comments: list[GitHubComment] = []
        page = 1
        async with httpx.AsyncClient() as client:
            while True:
                response = await client.get(
                    url=url,
                    headers=dict(request_header_items),
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                if not (new_comments := response.json()):
                    break
                comments.extend(
                    [
                        GitHubComment(
                            **{k: v for k, v in comment.items() if v is not None}
                        )  # type: ignore
                        for comment in new_comments
                    ]
                )
                page += 1
            return comments

    async def _get_issues(self, per_page: int = 100) -> list[GitHubIssue]:
        """
        Get a list of all issues for the given repository.

        per_page: The number of issues to request per page.

        Returns:
            A list of `GitHubIssue` objects, each representing an issue.
        """
        url = f"https://api.github.com/repos/{self.repo}/issues"
        issues: list[GitHubIssue] = []
        page = 1
        async with httpx.AsyncClient() as client:
            while len(issues) < self.n_issues:
                remaining = self.n_issues - len(issues)
                response = await client.get(
                    url=url,
                    headers=self.request_headers,
                    params={
                        "per_page": min(remaining, per_page),
                        "page": page,
                        "include": "comments",
                    },
                )
                response.raise_for_status()
                if not (new_issues := response.json()):
                    break
                issues.extend(
                    [
                        GitHubIssue(**{k: v for k, v in issue.items() if v is not None})  # type: ignore
                        for issue in new_issues
                    ]
                )
                page += 1
            return issues

    async def load(self) -> list[Document]:
        """
        Load all issues for the given repository.

        Returns:
            A list of `Document` objects, each representing an issue.
        """
        documents: list[Document] = []
        for issue in await self._get_issues():
            self.logger.debug(f"Found {issue.title!r}")
            clean_issue_body = rm_text_after(
                rm_html_comments(issue.body or ""), self.ignore_body_after
            )
            text = f"\n\n##**{issue.title}:**\n{clean_issue_body}\n\n"
            if self.include_comments:
                for (
                    comment
                ) in await self._get_issue_comments(  # hashable headers for lru_cache
                    self.repo, tuple(self.request_headers.items()), issue.number
                ):
                    if comment.user.login not in self.ignore_users:
                        text += f"**[{comment.user.login}]**: {comment.body}\n\n"
            metadata = dict(
                source=self.source_type,
                link=getattr(issue, "html_url", None),
                title=issue.title,
                labels=", ".join([label.name for label in issue.labels]),
                created_at=issue.created_at.timestamp() if issue.created_at else None,
            )
            documents.extend(
                await document_to_excerpts(
                    Document(
                        text=text,
                        metadata=metadata,
                    )
                )
            )
        return documents


class GitHubRepoLoader(Loader):
    """Loader for files on GitHub that match a glob pattern.


    Attributes:
        repo: The GitHub repository in the format 'owner/repo'.
        include_globs: A list of glob patterns to include.
        exclude_globs: A list of glob patterns to exclude.

    Raises:
        ValueError: If the repository is not in the format 'owner/repo'.

    Example:
        Load all files from the `prefecthq/prefect`
        ```python
        from raggy.loaders.github import GitHubRepoLoader

        loader = GitHubRepoLoader(repo="prefecthq/prefect")

        documents = await loader.load()
        print(documents)
        ```
    """

    source_type: str = "github source code"

    repo: str = Field(...)
    include_globs: list[str] | None = Field(default=None)
    exclude_globs: list[str] | None = Field(default=None)
    chunk_size: int = Field(default=500)
    overlap: float = Field(default=0.1)

    @field_validator("repo")
    def validate_repo(cls, v: str) -> str:
        if not re.match(r"^[^/\s]+/[^/\s]+$", v):
            raise ValueError(
                "Must provide a GitHub repository in the format 'owner/repo'"
            )
        return f"https://github.com/{v}.git"

    async def load(self) -> list[Document]:
        """Load files from GitHub that match the glob pattern."""
        async with OPEN_FILE_CONCURRENCY:
            async with aiofiles.tempfile.TemporaryDirectory(suffix="_raggy") as tmp_dir:
                process = await asyncio.create_subprocess_exec(
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    self.repo,
                    tmp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                if (await process.wait()) != 0:
                    stderr = await process.stderr.read() if process.stderr else b""
                    raise OSError(f"Failed to clone repository:\n {stderr.decode()}")

                stdout = await process.stdout.read() if process.stdout else b""
                self.logger.debug(stdout.decode())

                # Read the contents of each file that matches the glob pattern
                documents: list[Document] = []

                for file in multi_glob(tmp_dir, self.include_globs, self.exclude_globs):
                    self.logger.info(f"Loading file: {file!r}")

                    metadata = dict(
                        source=self.source_type,
                        link="/".join(
                            [
                                self.repo.replace(".git", ""),
                                "tree/main",
                                str(file),
                            ]
                        ),
                        title=file.name,
                        filename=file.name,
                    )
                    documents.extend(
                        await document_to_excerpts(
                            Document(
                                text=await read_file_with_chardet(Path(tmp_dir) / file),
                                metadata=metadata,
                            ),
                            chunk_size=self.chunk_size,
                            overlap=self.overlap,
                        )
                    )
                return documents
