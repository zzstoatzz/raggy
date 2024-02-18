from datetime import timedelta

from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.utilities.annotations import quote

import raggy
from raggy.documents import Document
from raggy.loaders.base import Loader
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.tpuf import TurboPuffer


def html_parser(html: str) -> str:
    import trafilatura

    trafilatura_config = trafilatura.settings.use_config()
    # disable signal, so it can run in a worker thread
    # https://github.com/adbar/trafilatura/issues/202
    trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    return trafilatura.extract(html, config=trafilatura_config)


raggy.settings.html_parser = html_parser

prefect_website_loaders = [
    SitemapLoader(
        urls=["https://docs.prefect.io/sitemap.xml", "https://prefect.io/sitemap.xml"],
        exclude=["api-ref", "/events/"],
    )
]


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


@flow(name="Update Knowledge", log_prints=True)
async def refresh_tpuf_namespace(namespace: str = "testing", reset: bool = False):
    """Flow updating vectorstore with info from the Prefect community."""
    documents = [
        doc
        for future in await run_loader.map(quote(prefect_website_loaders))
        for doc in await future.result()
    ]

    print(f"Loaded {len(documents)} documents from the Prefect community.")

    async with TurboPuffer(namespace=namespace) as tpuf:
        if reset:
            await tpuf.reset()
            print(f"Deleted all documents from tpuf ns {namespace!r}.")

        await tpuf.upsert(documents=documents)

    print(f"Updated tpuf ns {namespace!r} with {len(documents)} documents.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(refresh_tpuf_namespace(reset=True))
