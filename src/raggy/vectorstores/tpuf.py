import asyncio
from typing import Literal, Sequence

import turbopuffer as tpuf
from prefect.utilities.asyncutils import run_coro_as_sync
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from turbopuffer.query import Filters
from turbopuffer.vectors import VectorResult

from raggy.documents import Document
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

    api_key: SecretStr = Field(
        default=..., description="The API key for the TurboPuffer instance."
    )
    default_namespace: str = Field(default="raggy")

    @model_validator(mode="after")
    def set_api_key(self):
        if not tpuf.api_key and self.api_key:
            tpuf.api_key = self.api_key.get_secret_value()
        return self


tpuf_settings = TurboPufferSettings()


class TurboPuffer(Vectorstore):
    """Wrapper for turbopuffer.Namespace as a context manager.

    Attributes:
        namespace: The namespace to use for the TurboPuffer instance.

    Examples:
        Upsert documents to a namespace:
        ```python
        from raggy.documents import Document
        from raggy.vectorstores.tpuf import TurboPuffer

        with TurboPuffer() as tpuf: # default namespace is "raggy"
            tpuf.upsert(documents=[Document(id="1", text="Hello, world!")])
        ```

        Query a namespace:
        ```python
        from raggy.vectorstores.tpuf import TurboPuffer

        with TurboPuffer() as tpuf:
            result = tpuf.query(text="Hello, world!")
            print(result)
        ```
    """

    namespace: str = Field(default_factory=lambda: tpuf_settings.default_namespace)

    @property
    def ns(self):
        if not tpuf.api_key:
            raise ValueError(
                "TurboPuffer API key not found. Please set TURBOPUFFER_API_KEY environment variable."
            )
        return tpuf.Namespace(self.namespace)

    def upsert(
        self,
        documents: Sequence[Document] | None = None,
        ids: list[str] | list[int] | None = None,
        vectors: list[list[float]] | None = None,
        attributes: dict[str, list[str | int | None]] | None = None,
        distance_metric: Literal[
            "cosine_distance", "euclidean_squared"
        ] = "cosine_distance",
    ):
        attributes = attributes or {}
        upsert_cols = {}

        if documents is None and vectors is None:
            raise ValueError("Either `documents` or `vectors` must be provided.")

        if documents:
            ids = [document.id for document in documents]
            texts = [document.text for document in documents]
            embeddings = run_coro_as_sync(create_openai_embeddings(texts))
            assert embeddings is not None and isinstance(embeddings, list), (
                "embeddings cannot be none"
            )
            vectors = (
                [embeddings] if not isinstance(embeddings[0], list) else embeddings
            )
            if attributes.get("text"):
                raise ValueError(
                    "The `text` attribute is reserved and cannot be used as a custom attribute."
                )
            attributes["text"] = [str(t) for t in texts]

        assert ids is not None, "ids cannot be none"
        assert vectors is not None, "vectors cannot be none"

        upsert_cols["id"] = ids
        upsert_cols["vector"] = vectors  # type: ignore
        upsert_cols.update(attributes)  # type: ignore

        self.ns.write(
            upsert_columns=upsert_cols,
            distance_metric=distance_metric,
        )

    def query(
        self,
        text: str | None = None,
        vector: list[float] | None = None,
        top_k: int = 10,
        distance_metric: str = "cosine_distance",
        filters: Filters | None = None,
        include_attributes: list[str] | None = None,
        include_vectors: bool = False,
    ) -> VectorResult:
        if text:
            vector = run_coro_as_sync(create_openai_embeddings(text))
        elif vector is None:
            raise ValueError("Either `text` or `vector` must be provided.")

        return self.ns.query(  # type: ignore
            vector=vector,
            top_k=top_k,
            distance_metric=distance_metric,
            filters=filters,
            include_attributes=include_attributes or ["text"],
            include_vectors=include_vectors,
        )

    def delete(self, ids: str | int | list[str] | list[int]):
        ids_list = ids if isinstance(ids, list) else [ids]
        self.ns.write(deletes=ids_list)  # type: ignore

    def reset(self):
        try:
            self.ns.delete_all()
        except tpuf.APIError as e:
            if e.status_code == 404:
                self.logger.debug_kv("404", "Namespace already empty.")
            else:
                raise

    def ok(self) -> bool:
        try:
            return self.ns.exists()
        except tpuf.APIError as e:
            if e.status_code == 404:
                self.logger.debug_kv("404", "Namespace does not exist.")
                return False
            raise

    async def upsert_batched(
        self,
        documents: Sequence[Document],
        batch_size: int = 100,
        max_concurrent: int = 8,
        distance_metric: Literal[
            "cosine_distance", "euclidean_squared"
        ] = "cosine_distance",
    ) -> None:
        """Upsert documents in batches concurrently.

        Args:
            documents: Sequence of documents to upsert
            batch_size: Maximum number of documents per batch
            max_concurrent: Maximum number of concurrent upsert operations
        """
        document_list = list(documents)
        batches = [
            document_list[i : i + batch_size]
            for i in range(0, len(document_list), batch_size)
        ]

        async def _upsert(batch: list[Document], batch_num: int) -> None:
            texts = [doc.text for doc in batch]
            batch_ids = [doc.id for doc in batch]
            batch_vectors = await create_openai_embeddings(texts)
            self.ns.write(
                upsert_columns={
                    "id": batch_ids,
                    "vector": batch_vectors,
                    "text": texts,
                },
                distance_metric=distance_metric,
            )
            self.logger.debug_kv(
                "Upserted",
                f"Batch {batch_num + 1}/{len(batches)} ({len(batch)} documents)",
            )

        for i in range(0, len(batches), max_concurrent):
            batch_group = batches[i : i + max_concurrent]
            await asyncio.gather(
                *[_upsert(batch, i + j) for j, batch in enumerate(batch_group)]
            )


def query_namespace(
    query_text: str,
    filters: Filters | None = None,
    namespace: str = "raggy",
    top_k: int = 10,
    max_tokens: int = 500,
) -> str:
    """Query a TurboPuffer namespace."""
    with TurboPuffer(namespace=namespace) as tpuf:
        vector_result = tpuf.query(
            text=query_text,
            filters=filters,
            top_k=top_k,
        )
        assert vector_result.data is not None, "No data found"

        concatenated_result = "\n".join(
            str(row.attributes["text"]) if row.attributes else ""
            for row in vector_result.data
        )

        return slice_tokens(concatenated_result, max_tokens)


def multi_query_tpuf(
    queries: list[str], n_results: int = 3, namespace: str = "raggy"
) -> str:
    """searches a Turbopuffer namespace for the given queries"""
    results = [
        query_namespace(
            query,
            namespace=namespace,
            top_k=n_results,
            max_tokens=800 // len(queries),
        )
        for query in queries
    ]

    return "\n\n".join(results)
