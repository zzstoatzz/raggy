# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marvin",
#   "prefect",
#   "raggy[tpuf]",
# ]
# ///

import asyncio
import warnings
from typing import Any

import httpx
import marvin
from prefect import flow, task
from prefect.context import TaskRunContext
from rich.status import Status

from raggy.documents import Document
from raggy.loaders.github import GitHubRepoLoader
from raggy.vectorstores.tpuf import TurboPuffer, multi_query_tpuf

TPUF_NS = "demo"


def get_last_commit_sha(
    context: TaskRunContext, parameters: dict[str, Any]
) -> str | None:
    """Cache based on Last-Modified header of the first URL."""
    try:
        return httpx.get(
            f"https://api.github.com/repos/{parameters['repo']}/commits/main"
        ).json()["sha"]
    except Exception:
        return None


@task(
    task_run_name="load documents from {repo}",
    cache_key_fn=get_last_commit_sha,
)
async def gather_documents(
    repo: str,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
) -> list[Document]:
    return await GitHubRepoLoader(
        repo=repo, include_globs=include_globs, exclude_globs=exclude_globs
    ).load()


@flow(flow_run_name="{repo}")
def ingest_repo(repo: str):
    """Ingest a GitHub repository into the vector database.

    Args:
        repo: The repository to ingest (format: "owner/repo").
    """
    documents: list[Document] = gather_documents.submit(repo).result()
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
        >>> "user says: what does this repo use for packaging?"
        >>> "assistant: do_research(['packaging', 'dependencies', 'setuptools'])"
        ```
    """
    return multi_query_tpuf(queries=query_texts, namespace=TPUF_NS)


@flow(log_prints=True)
async def chat_with_repo(initial_message: str, clean_up: bool = True):
    try:
        agent = marvin.Agent(
            model="gpt-4o",
            name="Repo Researcher",
            instructions=(
                "Use your tools to ingest and chat about a GitHub repo. "
                "Let the user choose questions, query as needed to get good answers. "
                "You must find documented syntax in the repo of question before suggesting it."
            ),
            tools=[
                ingest_repo,
                do_research,
            ],
        )

        result = await agent.run_async(initial_message)
        print(result)

        while True:
            user_input = input("User: ")
            if user_input == "exit":
                break
            result = await agent.run_async(user_input)
            print(result)

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
        initial_message = "lets chat about zzstoatzz/raggy - please ingest it and tell me how to contribute"

    with marvin.Thread():
        asyncio.run(chat_with_repo(initial_message))
