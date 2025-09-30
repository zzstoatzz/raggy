import asyncio
from abc import ABC, abstractmethod

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field

from raggy.documents import Document
from raggy.utilities.collections import batched
from raggy.utilities.logging import get_logger


class Loader(BaseModel, ABC):
    """A base class for loaders."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    excerpt_template: Template | None = Field(
        default=None,
        description="Jinja2 template for rendering document excerpts. If None, uses the default template.",
    )
    chunk_tokens: int = Field(
        default=300,
        description="Number of tokens per excerpt chunk.",
    )

    @abstractmethod
    async def load(self) -> list[Document]:
        pass

    @property
    def logger(self):
        return get_logger(self.__class__.__name__)


class MultiLoader(Loader):
    """A loader that loads from multiple loaders.

    Attributes:
        loaders: The loaders to load from.

    Examples:
        Basic Usage of `MultiLoader`
        ```python
        from raggy.loaders.base import MultiLoader
        from raggy.loaders.github import GitHubRepoLoader

        loader = MultiLoader(
            loaders=[
                GitHubRepoLoader(repo="prefecthq/prefect"),
                GitHubRepoLoader(repo="prefecthq/marvin"),
            ]
        )

        documents = await loader.load() # all (chunked) files from both repos
        print(documents)
        ```

    """

    loaders: list[Loader]

    async def load(self, batch_size: int = 5) -> list[Document]:
        return [
            doc
            for batch in batched(self.loaders, batch_size)
            for docs in await asyncio.gather(*(loader.load() for loader in batch))
            for doc in docs
        ]
