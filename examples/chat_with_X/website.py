# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marvin",
#   "prefect",
#   "raggy[tpuf]",
# ]
# ///

import asyncio
import re
import warnings
from datetime import timedelta
from typing import Any

import httpx
import marvin
from prefect import flow, task
from prefect.context import TaskRunContext
from rich.status import Status

from raggy.documents import Document
from raggy.loaders.web import SitemapLoader, URLLoader
from raggy.vectorstores.tpuf import TurboPuffer, multi_query_tpuf

TPUF_NS = "demo"


def get_last_modified(
    context: TaskRunContext, parameters: dict[str, Any]
) -> str | None:
    """Cache based on Last-Modified header of the first URL."""
    try:
        with httpx.Client() as client:
            response = client.head(parameters["urls"][0])
            return response.headers.get("Last-Modified", "")
    except Exception:
        return None


@task(
    task_run_name="load documents from {urls}",
    cache_key_fn=get_last_modified,
    cache_expiration=timedelta(hours=24),
)
async def gather_documents_from_sitemap(
    urls: list[str], exclude: list[str | re.Pattern[str]] | None = None
) -> list[Document]:
    return await SitemapLoader(urls=urls, exclude=exclude or []).load()


@task(task_run_name="load documents from {urls}")
async def gather_documents_from_websites(urls: list[str]) -> list[Document]:
    return await URLLoader(urls=urls).load()


@flow(flow_run_name="{sitemap_urls}")
def ingest_sitemaps(
    sitemap_urls: list[str], exclude: list[str | re.Pattern[str]] | None = None
):
    """Ingest a website into the vector database.

    Args:
        urls: The URLs to ingest (exact or glob patterns).
        exclude: The URLs to exclude (exact or glob patterns).
    """
    documents: list[Document] = gather_documents_from_sitemap.submit(
        sitemap_urls, exclude
    ).result()
    with TurboPuffer(namespace=TPUF_NS) as tpuf:
        print(f"Upserting {len(documents)} documents into {TPUF_NS}")
        task(tpuf.upsert_batched).submit(documents).wait()


@flow(flow_run_name="{urls}")
def ingest_websites(urls: list[str]):
    """Ingest a website into the vector database.

    Args:
        urls: The URLs to ingest (exact or glob patterns).
    """
    documents: list[Document] = gather_documents_from_websites.submit(urls).result()
    with TurboPuffer(namespace=TPUF_NS) as tpuf:
        print(f"Upserting {len(documents)} documents into {TPUF_NS}")
        task(tpuf.upsert_batched).submit(documents).wait()


@task(task_run_name="querying: {query_texts}")
def do_research(query_texts: list[str]):
    """Query the vector database.

    Args:
        query_texts: The queries to search for.

    Examples:
        ```python
        >>> "user says: how to create a flow in Prefect?"
        >>> "assistant: do_research(['create flows', 'prefect overview'])"
        ```
    """
    return multi_query_tpuf(queries=query_texts, namespace=TPUF_NS)


@flow(log_prints=True)
async def chat_with_urls(initial_message: str, clean_up: bool = True):
    agent = marvin.Agent(
        model="gpt-4o",
        name="Website Expert",
        instructions=(
            "Use your tools to ingest and chat about website content. "
            "Let the user choose questions, query as needed to get good answers. "
            "You must find documented content on the website before making claims."
        ),
        tools=[
            ingest_sitemaps,
            ingest_websites,
            do_research,
        ],
    )

    result = await agent.run_async(initial_message)
    print(f"Assistant: {result}")

    while True:
        message = input("You: ")
        if message.lower() in ["exit", "quit"]:
            break
        result = await agent.run_async(message)
        print(f"Assistant: {result}")

    if clean_up:
        with TurboPuffer(namespace=TPUF_NS) as tpuf:
            with Status(f"Cleaning up namespace {TPUF_NS}"):
                task(tpuf.reset)()


if __name__ == "__main__":
    import sys

    warnings.filterwarnings("ignore", category=UserWarning)

    if len(sys.argv) > 1:
        initial_message = sys.argv[1]
    else:
        initial_message = (
            "let's chat about this project - "
            "please ingest the docs @ https://zzstoatzz.github.io/raggy/sitemap.xml "
            "and then tell me how it works"
        )

    with marvin.Thread():
        asyncio.run(chat_with_urls(initial_message))
