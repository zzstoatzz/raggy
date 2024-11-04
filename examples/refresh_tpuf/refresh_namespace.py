# /// script
# dependencies = [
#     "prefect",
#     "raggy[tpuf]@git+https://github.com/zzstoatzz/raggy@improve-ingest",
#     "trafilatura",
# ]
# ///

from datetime import timedelta

from bs4 import BeautifulSoup
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.utilities.annotations import quote

import raggy
from raggy.documents import Document
from raggy.loaders.base import Loader
from raggy.loaders.github import GitHubRepoLoader
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.tpuf import TurboPuffer


def html_parser(html: str) -> str:
    import trafilatura

    trafilatura_config = trafilatura.settings.use_config()  # type: ignore
    # disable signal, so it can run in a worker thread
    # https://github.com/adbar/trafilatura/issues/202
    trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    return (
        trafilatura.extract(html, config=trafilatura_config)
        or BeautifulSoup(html, "html.parser").get_text()
    )


raggy.settings.html_parser = html_parser


loaders = {
    "prefect-2": [
        SitemapLoader(
            url_processor=lambda x: x.replace("docs.", "docs-2."),
            urls=[
                "https://docs-2.prefect.io/sitemap.xml",
                "https://prefect.io/sitemap.xml",
            ],
            exclude=["api-ref", "www.prefect.io/events"],
        ),
        GitHubRepoLoader(
            repo="PrefectHQ/prefect",
            include_globs=["flows/"],
        ),
    ],
    "prefect-3": [
        SitemapLoader(
            urls=[
                "https://docs.prefect.io/sitemap.xml",
                "https://prefect.io/sitemap.xml",
            ],
            exclude=["api-ref", "www.prefect.io/events"],
        ),
        GitHubRepoLoader(
            repo="PrefectHQ/prefect",
            include_globs=["flows/"],
        ),
    ],
    "controlflow": [
        SitemapLoader(
            urls=["https://controlflow.ai/sitemap.xml"],
        ),
        GitHubRepoLoader(
            repo="PrefectHQ/controlflow",
            include_globs=["examples/"],
        ),
    ],
}


@task(
    retries=2,
    retry_delay_seconds=[3, 60],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    task_run_name="Run {loader.__class__.__name__}",
    persist_result=True,
    # refresh_cache=True,
)
async def run_loader(loader: Loader) -> list[Document]:
    return await loader.load()


@flow(
    name="Update Namespace",
    flow_run_name="Refreshing {namespace}",
    log_prints=True,
)
async def refresh_tpuf_namespace(
    namespace: str,
    namespace_loaders: list[Loader],
    reset: bool = False,
    batch_size: int = 100,
):
    """Flow updating vectorstore with info from the Prefect community."""
    documents: list[Document] = [
        doc
        for future in run_loader.map(quote(namespace_loaders))  # type: ignore
        for doc in future.result()
    ]

    print(f"Loaded {len(documents)} documents from the Prefect community.")

    async with TurboPuffer(namespace=namespace) as tpuf:
        if reset:
            await task(tpuf.reset)()
            print(f"RESETTING: Deleted all documents from tpuf ns {namespace!r}.")

        await task(tpuf.upsert_batched)(documents=documents, batch_size=batch_size)

    print(f"Updated tpuf ns {namespace!r} with {len(documents)} documents.")


@flow(name="Refresh Namespaces", log_prints=True)
async def refresh_tpuf(reset: bool = False, batch_size: int = 100):
    for namespace, namespace_loaders in loaders.items():
        await refresh_tpuf_namespace(
            namespace, namespace_loaders, reset=reset, batch_size=batch_size
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(refresh_tpuf(reset=True))
