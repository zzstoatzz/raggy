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
from marvin.utilities.asyncio import run_async
from pydantic import BaseModel, model_validator

import raggy
from raggy.documents import Document
from raggy.utils import create_openai_embeddings, get_distinct_documents


def get_client(client_type: Literal["base", "http"]) -> HttpClient:
    if client_type == "base":
        return Client()
    elif client_type == "http":
        return HttpClient()
    else:
        raise ValueError(f"Unknown client type: {client_type}")


class Chroma(BaseModel):
    """A wrapper for chromadb.Client - used as an async context manager"""

    client_type: Literal["base", "http"] = "base"
    collection: Collection | None = None

    _in_context: bool = False

    @model_validator(mode="after")
    def validate_collection(self):
        if not self.collection:
            self.collection = get_client(self.client_type).get_or_create_collection(
                name="raggy"
            )
        return self

    async def delete(
        self,
        ids: list[str] = None,
        where: dict = None,
        where_document: Document = None,
    ):
        await run_async(
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

        await run_async(self.collection.add, **kwargs)

        get_result = await run_async(self.collection.get, ids=kwargs["ids"])

        return get_result.get("documents")

    async def query(
        self,
        query_embeddings: list[list[float]] = None,
        query_texts: list[str] = None,
        n_results: int = 10,
        where: dict = None,
        where_document: dict = None,
        include: "Include" = ["metadatas"],
        **kwargs,
    ) -> "QueryResult":
        return await run_async(
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
        return await run_async(self.collection.count)

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
        await run_async(self.collection.upsert, **kwargs)

        get_result = await run_async(self.collection.get, ids=kwargs["ids"])

        return get_result.get("documents")

    async def reset_collection(self):
        """Delete and recreate the collection."""
        client = get_client(self.client_type)
        await run_async(client.delete_collection, self.collection.name)
        self.collection = await run_async(
            client.create_collection,
            name=self.collection.name,
        )

    def ok(self) -> bool:
        logger = raggy.utilities.logging.get_logger()
        try:
            version = self.client.get_version()
        except Exception as e:
            logger.error(f"Cannot connect to Chroma: {e}")
        if re.match(r"^\d+\.\d+\.\d+$", version):
            logger.debug(f"Connected to Chroma v{version}")
            return True
        return False

    async def __aenter__(self):
        self._in_context = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self._in_context = False
