import asyncio
from typing import Iterable

import turbopuffer as tpuf
from pydantic import (
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from turbopuffer.vectors import VectorResult

from raggy.documents import Document
from raggy.utilities.asyncutils import run_sync_in_worker_thread
from raggy.utilities.embeddings import create_openai_embeddings
from raggy.utilities.text import slice_tokens
from raggy.vectorstores.base import Vectorstore


class TurboPufferSettings(BaseSettings):
    """Settings for the TurboPuffer vectorstore."""

    model_config = SettingsConfigDict(
        env_prefix="TURBOPUFFER_",
        env_file=("~/.raggy/.env", ".env"),
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    api_key: SecretStr
    default_namespace: str = "raggy"

    @model_validator(mode="after")
    def set_api_key(self):
        if not tpuf.api_key and self.api_key:
            tpuf.api_key = self.api_key.get_secret_value()
        return self


tpuf_settings = TurboPufferSettings()


class TurboPuffer(Vectorstore):
    """Wrapper for turbopuffer.Namespace as an async context manager.

    Attributes:
        namespace: The namespace to use for the TurboPuffer instance.

    Examples:
        Upsert documents to a namespace:
        ```python
        from raggy.documents import Document
        from raggy.vectorstores.tpuf import TurboPuffer

        async with TurboPuffer() as tpuf: # default namespace is "raggy"
            await tpuf.upsert(documents=[Document(id="1", text="Hello, world!")])
        ```

        Query a namespace:
        ```python
        from raggy.vectorstores.tpuf import TurboPuffer

        async with TurboPuffer() as tpuf:
            result = await tpuf.query(text="Hello, world!")
            print(result)
        ```
    """

    namespace: str = Field(default_factory=lambda: tpuf_settings.default_namespace)

    @property
    def ns(self):
        return tpuf.Namespace(self.namespace)

    async def upsert(
        self,
        documents: Iterable[Document] | None = None,
        ids: list[str] | list[int] | None = None,
        vectors: list[list[float]] | None = None,
        attributes: dict | None = None,
    ):
        attributes = attributes or {}

        if documents is None and vectors is None:
            raise ValueError("Either `documents` or `vectors` must be provided.")

        if documents:
            ids = [document.id for document in documents]
            vectors = await create_openai_embeddings(
                [document.text for document in documents]
            )  # type: ignore
            if not isinstance(vectors[0], list):  # type: ignore
                vectors = [vectors]  # type: ignore
            if attributes.get("text"):
                raise ValueError(
                    "The `text` attribute is reserved and cannot be used as a custom attribute."
                )
            attributes |= {"text": [document.text for document in documents]}
        await run_sync_in_worker_thread(
            self.ns.upsert, ids=ids, vectors=vectors, attributes=attributes
        )

    async def query(
        self,
        text: str | None = None,
        vector: list[float] | None = None,
        top_k: int = 10,
        distance_metric: str = "cosine_distance",
        filters: dict | None = None,
        include_attributes: list[str] | None = None,
        include_vectors: bool = False,
    ) -> VectorResult:
        if text:
            vector = await create_openai_embeddings(text)
        else:
            if vector is None:
                raise ValueError("Either `text` or `vector` must be provided.")

        return await run_sync_in_worker_thread(
            self.ns.query,
            vector=vector,
            top_k=top_k,
            distance_metric=distance_metric,
            filters=filters,
            include_attributes=include_attributes or ["text"],
            include_vectors=include_vectors,
        )

    async def delete(self, ids: str | int | list[str] | list[int]):
        await run_sync_in_worker_thread(self.ns.delete, ids=ids)

    async def reset(self):
        try:
            await run_sync_in_worker_thread(self.ns.delete_all)
        except tpuf.APIError as e:
            if e.status_code == 404:
                self.logger.debug_kv("404", "Namespace already empty.")
            else:
                raise

    async def ok(self) -> bool:
        try:
            return await run_sync_in_worker_thread(self.ns.exists)
        except tpuf.APIError as e:
            if e.status_code == 404:
                self.logger.debug_kv("404", "Namespace does not exist.")
                return False
            raise


async def query_namespace(
    query_text: str,
    filters: dict | None = None,
    namespace: str = "raggy",
    top_k: int = 10,
    max_tokens: int = 500,
) -> str:
    """Query a TurboPuffer namespace.

    Args:
        query_text: The text to query for.
        filters: Filters to apply to the query.
        namespace: The namespace to query.
        top_k: The number of results to return.

    Examples:
        Basic Usage of `query_namespace`
        ```python
        from raggy.vectorstores.tpuf import query_namespace

        print(await query_namespace("How to create a flow in Prefect?"))
        ```

        Using `filters` with `query_namespace`
        ```python
        from raggy.vectorstores.tpuf import query_namespace

        filters={
            'id': [['In', [1, 2, 3]]],
            'key1': [['Eq', 'one']],
            'filename': [['Or', [['Glob', '**.md'], ['Glob', '**.py']]], ['NotGlob', '/migrations/**']]
        }

        print(await query_namespace("How to create a flow in Prefect?", filters=filters))
        ```
    """
    async with TurboPuffer(namespace=namespace) as tpuf:
        vector_result = await tpuf.query(
            text=query_text,
            filters=filters,
            top_k=top_k,
        )

        concatenated_result = "\n".join(
            row.attributes["text"] for row in vector_result.data
        )

        return slice_tokens(concatenated_result, max_tokens)


async def multi_query_tpuf(
    queries: list[str], n_results: int = 3, namespace: str = "raggy"
) -> str:
    """searches a Turbopuffer namespace for the given queries"""
    results = await asyncio.gather(
        *[
            query_namespace(
                query,
                namespace=namespace,
                top_k=n_results,
                max_tokens=800 // len(queries),
            )
            for query in queries
        ]
    )

    return "\n\n".join(results)
