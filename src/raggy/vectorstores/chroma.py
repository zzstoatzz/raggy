import asyncio
import re
from typing import Literal

from raggy.utilities.collections import distinct

try:
    from chromadb import Client, CloudClient, HttpClient
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.api.models.Collection import Document as ChromaDocument
    from chromadb.api.types import QueryResult
    from chromadb.utils.batch_utils import create_batches
except ImportError:
    raise ImportError(
        "You must have `chromadb` installed to use the Chroma vector store. "
        "Install it with `pip install 'raggy[chroma]'`."
    )

from raggy.documents import Document as RaggyDocument
from raggy.settings import settings
from raggy.utilities.asyncutils import run_sync_in_worker_thread
from raggy.utilities.embeddings import create_openai_embeddings
from raggy.utilities.text import slice_tokens
from raggy.vectorstores.base import Vectorstore

ChromaClientType = Literal["base", "http", "cloud"]


def get_client(client_type: ChromaClientType) -> ClientAPI:
    if client_type == "base":
        return Client()
    elif client_type == "http":
        return HttpClient()
    elif client_type == "cloud":
        return CloudClient(
            tenant=settings.chroma.cloud_tenant,
            database=settings.chroma.cloud_database,
            api_key=settings.chroma.cloud_api_key.get_secret_value(),
        )
    else:
        raise ValueError(f"Unknown client type: {client_type}")


class Chroma(Vectorstore):
    """A wrapper for chromadb.Client - used as an async context manager.

    Attributes:
        client_type: The type of client to use. Must be one of "base" or "http".
        collection_name: The name of the collection to use.

    Example:
        Query a collection:
        ```python
        from raggy.vectorstores.chroma import Chroma

        async with Chroma(collection_name="my-collection") as chroma:
            result = await chroma.query(query_texts=["Hello, world!"])
            print(result)
        ```

    """

    client_type: ChromaClientType = "base"
    collection_name: str = "raggy"

    @property
    def collection(self) -> Collection:
        return get_client(self.client_type).get_or_create_collection(
            name=self.collection_name
        )

    async def delete(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        where_document: ChromaDocument | None = None,
    ):
        await run_sync_in_worker_thread(
            self.collection.delete,
            ids=ids,
            where=where,
            where_document=where_document,
        )

    async def add(self, documents: list[RaggyDocument]) -> list[ChromaDocument]:
        unique_documents = list(distinct(documents, key=lambda doc: doc.text))

        ids = [doc.id for doc in unique_documents]
        texts = [doc.text for doc in unique_documents]
        metadatas = [
            doc.metadata.model_dump(exclude_none=True) for doc in unique_documents
        ]

        embeddings = await create_openai_embeddings(texts)

        data = {
            "ids": ids,
            "documents": texts,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }

        batched_data: list[tuple] = create_batches(
            get_client(self.client_type),
            **data,
        )

        await asyncio.gather(
            *(asyncio.to_thread(self.collection.add, *batch) for batch in batched_data)
        )

        get_result = await asyncio.to_thread(self.collection.get, ids=ids)

        return get_result.get("documents") or []

    async def query(
        self,
        query_embeddings: list[list[float]] | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
        include: list[str] = ["metadatas"],
        **kwargs,
    ) -> "QueryResult":
        return await run_sync_in_worker_thread(
            self.collection.query,
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
            **kwargs,
        )

    async def count(self) -> int:
        return await run_sync_in_worker_thread(self.collection.count)

    async def upsert(self, documents: list[RaggyDocument]) -> list[ChromaDocument]:
        documents = list(distinct(documents, key=lambda doc: hash(doc.text)))
        kwargs = dict(
            ids=[document.id for document in documents],
            documents=[document.text for document in documents],
            metadatas=[
                document.metadata.model_dump(exclude_none=True) or None
                for document in documents
            ],
            embeddings=await create_openai_embeddings(
                [document.text for document in documents]
            ),
        )
        await run_sync_in_worker_thread(self.collection.upsert, **kwargs)

        get_result = await run_sync_in_worker_thread(
            self.collection.get, ids=kwargs["ids"]
        )

        return get_result.get("documents") or []

    async def reset_collection(self):
        client = get_client(self.client_type)
        try:
            await run_sync_in_worker_thread(
                client.delete_collection, self.collection_name
            )
        except Exception:
            self.logger.warning_kv(
                "Collection not found",
                f"Creating a new collection {self.collection_name!r}",
            )
        await run_sync_in_worker_thread(client.create_collection, self.collection_name)

    def ok(self) -> bool:
        try:
            version = get_client(self.client_type).get_version()
        except Exception as e:
            self.logger.error_kv("Connection error", f"Cannot connect to Chroma: {e}")
        if re.match(r"^\d+\.\d+\.\d+$", version):
            self.logger.debug_kv("OK", f"Connected to Chroma {version!r}")
            return True
        return False


async def query_collection(
    query_text: str,
    query_embedding: list[float] | None = None,
    collection_name: str = "raggy",
    top_k: int = 10,
    where: dict | None = None,
    where_document: dict | None = None,
    max_tokens: int = 500,
    client_type: ChromaClientType = "base",
) -> str:
    """Query a Chroma collection.

    Args:
        query_text: The text to query for.
        filters: Filters to apply to the query.
        collection: The collection to query.
        top_k: The number of results to return.

    Example:
        Basic query of a collection:
        ```python
        from raggy.vectorstores.chroma import query_collection

        print(await query_collection("How to create a flow in Prefect?"))
        ```
    """
    async with Chroma(
        collection_name=collection_name, client_type=client_type
    ) as chroma:
        query_embedding = query_embedding or await create_openai_embeddings(query_text)

        query_result = await chroma.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents"],
        )

        assert (
            result := query_result.get("documents")
        ) is not None, "No documents found"

        return slice_tokens("\n".join(result[0]), max_tokens)
