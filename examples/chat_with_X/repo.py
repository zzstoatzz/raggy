import warnings

import httpx
import prefect.runtime.flow_run as run
from marvin.beta.assistants import Assistant
from marvin.utilities.tools import custom_partial
from prefect import flow, task
from prefect.utilities.asyncutils import run_coro_as_sync
from rich.status import Status

from raggy.documents import Document
from raggy.loaders.github import GitHubRepoLoader
from raggy.vectorstores.tpuf import TurboPuffer, query_namespace

TPUF_NS = "demo"


@task(
    task_run_name="load documents from {repo}",
    cache_key_fn=lambda *_, **__: httpx.get(  # update embeddings on changes to the repo
        f"https://api.github.com/repos/{run.parameters['repo']}/commits/main"
    ).json()["sha"],
)
async def gather_documents(repo: str) -> list[Document]:
    return await GitHubRepoLoader(repo=repo).load()


@task
async def upsert_documents(documents: list[Document]):
    async with TurboPuffer(namespace=TPUF_NS) as tpuf:
        print(f"Upserting {len(documents)} documents into {TPUF_NS}")
        await tpuf.upsert(documents)


@flow(flow_run_name="{repo}")
async def ingest_repo(repo: str):
    documents = await gather_documents(repo)
    await upsert_documents(documents)


@flow(log_prints=True)
async def chat_with_repo(initial_message: str | None = None, clean_up: bool = True):
    try:
        with Assistant(
            model="gpt-4o",
            name="Raggy Expert",
            instructions=(
                "Use your tools to ingest and chat about a GitHub repo. "
                "Let the user choose questions, query as needed to get good answers. "
                "You must find documented syntax in the repo of question before suggesting it."
            ),
            tools=[
                ingest_repo,
                custom_partial(
                    task(task_run_name="Q: {query_text}")(query_namespace),
                    namespace=TPUF_NS,
                ),
            ],
        ) as assistant:
            assistant.chat(initial_message=initial_message)  # type: ignore

    finally:
        if clean_up:
            async with TurboPuffer(namespace=TPUF_NS) as tpuf:
                with Status(f"Cleaning up namespace {TPUF_NS}"):
                    await task(tpuf.reset)()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run_coro_as_sync(
        chat_with_repo("lets chat about zzstoatzz/prefect-bot - please ingest it")
    )
