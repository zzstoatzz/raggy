from typing import Literal, Sequence

from chromadb import Client, CloudClient, HttpClient, Include
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.models.Collection import Document as ChromaDocument
from chromadb.api.types import Embedding, OneOrMany, PyEmbedding, QueryResult
from chromadb.utils.batch_utils import create_batches
from prefect.utilities.asyncutils import run_coro_as_sync

from raggy.documents import Document as RaggyDocument
from raggy.documents import DocumentMetadata
from raggy.settings import settings
from raggy.utilities.asyncutils import run_concurrent_tasks
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
        assert settings.chroma.cloud_api_key is not None
        return CloudClient(
            tenant=settings.chroma.cloud_tenant,
            database=settings.chroma.cloud_database,
            api_key=settings.chroma.cloud_api_key.get_secret_value(),
        )
    else:
        raise ValueError(f"Unknown client type: {client_type}")


class Chroma(Vectorstore):
    """A wrapper for chromadb.Client."""

    client_type: ChromaClientType = "base"
    collection_name: str = "raggy"

    @property
    def collection(self) -> Collection:
        return get_client(self.client_type).get_or_create_collection(
            name=self.collection_name
        )

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        where_document: ChromaDocument | None = None,
    ):
        self.collection.delete(
            ids=ids,
            where=where,
            where_document=where_document,  # type: ignore
        )

    def add(self, documents: Sequence[RaggyDocument]) -> list[ChromaDocument]:
        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [
            doc.metadata.model_dump(exclude_none=True)
            if isinstance(doc.metadata, DocumentMetadata)
            else None
            for doc in documents
        ]

        embeddings = run_coro_as_sync(create_openai_embeddings(texts))

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

        for batch in batched_data:
            self.collection.add(*batch)

        get_result = self.collection.get(ids=ids)
        return get_result.get("documents") or []

    def query(
        self,
        query_embeddings: OneOrMany[Embedding] | OneOrMany[PyEmbedding] | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
        include: Include = ["metadatas"],  # type: ignore
        **kwargs,
    ) -> QueryResult:
        return self.collection.query(
            query_embeddings=(
                run_coro_as_sync(create_openai_embeddings(query_texts))
                if query_texts and not query_embeddings
                else query_embeddings
            ),
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
            **kwargs,
        )

    def count(self) -> int:
        return self.collection.count()

    def upsert(self, documents: Sequence[RaggyDocument]) -> list[ChromaDocument]:
        kwargs = dict(
            ids=[document.id for document in documents],
            documents=[document.text for document in documents],
            metadatas=[
                document.metadata.model_dump(exclude_none=True)
                if isinstance(document.metadata, DocumentMetadata)
                else None
                for document in documents
            ],
            embeddings=run_coro_as_sync(
                create_openai_embeddings([document.text for document in documents])
            ),
        )
        self.collection.upsert(**kwargs)  # type: ignore

        get_result = self.collection.get(ids=kwargs["ids"])
        return get_result.get("documents") or []

    def reset_collection(self):
        client = get_client(self.client_type)
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            self.logger.warning_kv(
                "Collection not found",
                f"Creating a new collection {self.collection_name!r}",
            )
        client.create_collection(self.collection_name)

    def ok(self) -> bool:
        try:
            version = get_client(self.client_type).get_version()
            self.logger.debug_kv("OK", f"Connected to Chroma {version!r}")
            return True
        except Exception as e:
            self.logger.error_kv("Connection error", f"Cannot connect to Chroma: {e}")
            return False

    async def upsert_batched(
        self,
        documents: Sequence[RaggyDocument],
        batch_size: int = 100,
        max_concurrent: int = 8,
    ):
        """Upsert documents in batches concurrently."""
        document_list = list(documents)
        batches = [
            document_list[i : i + batch_size]
            for i in range(0, len(document_list), batch_size)
        ]

        # Create tasks that will run concurrently
        tasks = []
        for i, batch in enumerate(batches):

            async def _upsert(b=batch, n=i):
                # Get embeddings for the entire batch at once
                texts = [doc.text for doc in b]
                embeddings = await create_openai_embeddings(texts)

                # Prepare the batch data
                kwargs = dict(
                    ids=[doc.id for doc in b],
                    documents=texts,
                    metadatas=[
                        doc.metadata.model_dump(exclude_none=True)
                        if isinstance(doc.metadata, DocumentMetadata)
                        else None
                        for doc in b
                    ],
                    embeddings=embeddings,
                )

                # Do the upsert
                self.collection.upsert(**kwargs)
                self.logger.debug_kv(
                    "Upserted",
                    f"Batch {n + 1}/{len(batches)} ({len(b)} documents)",
                )

            tasks.append(_upsert)

        await run_concurrent_tasks(tasks, max_concurrent=max_concurrent)


def query_collection(
    query_text: str,
    collection_name: str = "raggy",
    top_k: int = 10,
    where: dict | None = None,
    where_document: dict | None = None,
    max_tokens: int = 500,
    client_type: ChromaClientType = "base",
) -> str:
    """Query a Chroma collection."""
    with Chroma(collection_name=collection_name, client_type=client_type) as chroma:
        query_result = chroma.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents"],  # type: ignore
        )

        assert (
            result := query_result.get("documents")
        ) is not None, "No documents found"
        return slice_tokens("\n".join(result[0]), max_tokens)
