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

import httpx
from marvin.beta.assistants import Assistant
from prefect import flow, task
from rich.status import Status

from raggy.documents import Document
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.tpuf import TurboPuffer, multi_query_tpuf

TPUF_NS = "demo"


def get_last_modified(context, parameters):
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
async def gather_documents(
    urls: list[str], exclude: list[str | re.Pattern] | None = None
) -> list[Document]:
    return await SitemapLoader(urls=urls, exclude=exclude or []).load()


@flow(flow_run_name="{urls}")
async def ingest_website(
    urls: list[str], exclude: list[str | re.Pattern] | None = None
):
    """Ingest a website into the vector database.

    Args:
        urls: The URLs to ingest (exact or glob patterns).
        exclude: The URLs to exclude (exact or glob patterns).
    """
    documents = await gather_documents(urls, exclude)
    with TurboPuffer(namespace=TPUF_NS) as tpuf:
        print(f"Upserting {len(documents)} documents into {TPUF_NS}")
        await tpuf.upsert_batched(documents)


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
async def chat_with_website(initial_message: str | None = None, clean_up: bool = True):
    try:
        with Assistant(
            model="gpt-4o",
            name="Website Expert",
            instructions=(
                "Use your tools to ingest and chat about website content. "
                "Let the user choose questions, query as needed to get good answers. "
                "You must find documented content on the website before making claims."
            ),
            tools=[
                ingest_website,
                do_research,
            ],
        ) as assistant:
            assistant.chat(initial_message=initial_message)  # type: ignore

    finally:
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

    asyncio.run(chat_with_website(initial_message))
