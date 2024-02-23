import asyncio

from marvin.beta.assistants import Assistant
from marvin.utilities.tools import custom_partial
from rich.status import Status

from raggy.loaders.github import GitHubRepoLoader
from raggy.vectorstores.tpuf import TurboPuffer, query_namespace

TPUF_NS = "demo"


async def ingest_repo(repo: str):
    loader = GitHubRepoLoader(repo=repo)

    with Status(f"Loading {repo}"):
        documents = await loader.load()

    async with TurboPuffer(namespace=TPUF_NS) as tpuf, Status(f"Ingesting {repo}"):
        await tpuf.upsert(documents)


async def chat_with_repo(repo: str, clean_up: bool = True):
    await ingest_repo(repo)

    try:
        with Assistant(
            name="Raggy Expert",
            instructions=(
                "You use `query_namespace` to answer questions about a github"
                f" repo called {repo}!. You MUST use this tool to answer questions."
            ),
            tools=[custom_partial(query_namespace, namespace=TPUF_NS)],
        ) as assistant:
            assistant.chat()

    finally:
        if clean_up:
            async with TurboPuffer(namespace=TPUF_NS) as tpuf, Status(
                f"Cleaning up namespace {TPUF_NS}"
            ):
                await tpuf.reset()


if __name__ == "__main__":
    asyncio.run(chat_with_repo("zzstoatzz/raggy"))
