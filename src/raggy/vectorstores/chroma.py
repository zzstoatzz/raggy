import re
from typing import Iterable, Literal

try:
    from chromadb import Client, HttpClient
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import Include, QueryResult
except ImportError:
    raise ImportError(
        "You must have `chromadb` installed to use the Chroma vector store. "
        "Install it with `pip install 'raggy[chroma]'`."
    )

from raggy.documents import Document, get_distinct_documents
from raggy.utilities.asyncutils import run_sync_in_worker_thread
from raggy.utilities.embeddings import create_openai_embeddings
from raggy.vectorstores.base import Vectorstore


def get_client(client_type: Literal["base", "http"]) -> HttpClient:
    if client_type == "base":
        return Client()
    elif client_type == "http":
        return HttpClient()
    else:
        raise ValueError(f"Unknown client type: {client_type}")


class Chroma(Vectorstore):
    """A wrapper for chromadb.Client - used as an async context manager"""

    client_type: Literal["base", "http"] = "base"
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
        where_document: Document | None = None,
    ):
        await run_sync_in_worker_thread(
            self.collection.delete,
            ids=ids,
            where=where,
            where_document=where_document,
        )

    async def add(self, documents: list[Document]) -> Iterable[Document]:
        documents = list(get_distinct_documents(documents))
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

        await run_sync_in_worker_thread(self.collection.add, **kwargs)

        get_result = await run_sync_in_worker_thread(
            self.collection.get, ids=kwargs["ids"]
        )

        return get_result.get("documents")

    async def query(
        self,
        query_embeddings: list[list[float]] | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
        include: "Include" = ["metadatas"],
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

    async def upsert(self, documents: list[Document]):
        documents = list(get_distinct_documents(documents))
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

        return get_result.get("documents")

    async def reset_collection(self):
        client = get_client(self.client_type)
        try:
            await run_sync_in_worker_thread(
                client.delete_collection, self.collection_name
            )
        except ValueError:
            self.logger.warning_kv(
                "Collection not found",
                f"Creating a new collection {self.collection_name!r}",
            )
        await run_sync_in_worker_thread(client.create_collection, self.collection_name)

    def ok(self) -> bool:
        try:
            version = self.client.get_version()
        except Exception as e:
            self.logger.error_kv("Connection error", f"Cannot connect to Chroma: {e}")
        if re.match(r"^\d+\.\d+\.\d+$", version):
            self.logger.debug_kv("OK", f"Connected to Chroma {version!r}")
            return True
        return False
