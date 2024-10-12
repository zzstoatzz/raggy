from datetime import timedelta
from typing import Literal

from bs4 import BeautifulSoup
from chromadb.api.models.Collection import Document as ChromaDocument
from prefect import flow, task
from prefect.tasks import task_input_hash

import raggy
from raggy.documents import Document
from raggy.loaders.base import Loader
from raggy.loaders.github import GitHubRepoLoader
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.chroma import Chroma, ChromaClientType


def html_parser(html: str) -> str:
    import trafilatura

    return trafilatura.extract(html) or BeautifulSoup(html, "html.parser").get_text()


raggy.settings.html_parser = html_parser


prefect_loaders = [
    SitemapLoader(
        urls=[
            "https://docs-3.prefect.io/sitemap.xml",
            "https://prefect.io/sitemap.xml",
        ],
        exclude=["api-ref", "www.prefect.io/events"],
    ),
    GitHubRepoLoader(
        repo="PrefectHQ/prefect",
        include_globs=["README.md"],
    ),
]


@task(
    retries=2,
    retry_delay_seconds=[3, 60],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    task_run_name="Run {loader.__class__.__name__}",
    persist_result=True,
    refresh_cache=True,
)
async def run_loader(loader: Loader) -> list[Document]:
    return await loader.load()


@task
async def add_documents(
    chroma: Chroma, documents: list[Document], mode: Literal["upsert", "reset"]
) -> list[ChromaDocument]:
    if mode == "reset":
        await chroma.reset_collection()
        docs = await chroma.add(documents)
    elif mode == "upsert":
        docs = await chroma.upsert(documents)
    return docs


@flow(name="Update Knowledge", log_prints=True)
async def refresh_chroma(
    collection_name: str = "default",
    chroma_client_type: ChromaClientType = "base",
    mode: Literal["upsert", "reset"] = "upsert",
):
    """Flow updating vectorstore with info from the Prefect community."""
    documents = [
        doc
        for future in run_loader.map(prefect_loaders)  # type: ignore
        for doc in future.result()
    ]

    print(f"Loaded {len(documents)} documents from the Prefect community.")

    async with Chroma(
        collection_name=collection_name, client_type=chroma_client_type
    ) as chroma:
        docs = await add_documents(chroma, documents, mode)

        print(f"Added {len(docs)} documents to the {collection_name} collection.")  # type: ignore


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        refresh_chroma(collection_name="test", chroma_client_type="cloud", mode="reset")  # type: ignore
    )
