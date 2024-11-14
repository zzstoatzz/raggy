# /// script
# dependencies = [
#     "prefect",
#     "raggy[tpuf]",
# ]
# ///

from datetime import timedelta

from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.utilities.annotations import quote

from raggy.documents import Document
from raggy.loaders.base import Loader
from raggy.loaders.github import GitHubRepoLoader
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.tpuf import TurboPuffer

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
def refresh_tpuf_namespace(
    namespace: str,
    namespace_loaders: list[Loader],
    reset: bool = False,
    batch_size: int = 100,
    max_concurrent: int = 8,
):
    """Flow updating vectorstore with info from the Prefect community."""
    documents: list[Document] = [
        doc
        for future in run_loader.map(quote(namespace_loaders))
        for doc in future.result()
    ]

    print(f"Loaded {len(documents)} documents from the Prefect community.")

    with TurboPuffer(namespace=namespace) as tpuf:
        if reset:
            task(tpuf.reset)()
            print(f"RESETTING: Deleted all documents from tpuf ns {namespace!r}.")

        task(tpuf.upsert_batched).submit(
            documents=documents, batch_size=batch_size, max_concurrent=max_concurrent
        ).wait()

    print(f"Updated tpuf ns {namespace!r} with {len(documents)} documents.")


@flow(name="Refresh Namespaces", log_prints=True)
def refresh_tpuf(reset: bool = False, batch_size: int = 100, test_mode: bool = False):
    for namespace, namespace_loaders in loaders.items():
        if test_mode:
            namespace = f"TESTING-{namespace}"
        refresh_tpuf_namespace(
            namespace, namespace_loaders, reset=reset, batch_size=batch_size
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_mode = sys.argv[1] != "prod"
    else:
        test_mode = True

    refresh_tpuf(reset=True, test_mode=test_mode)
