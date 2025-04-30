# /// script
# dependencies = [
#     "prefect",
#     "raggy[tpuf]",
# ]
# ///

import os
from datetime import timedelta
from typing import Any, Sequence

from prefect import flow, task
from prefect.context import TaskRunContext
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
            exclude=[
                "api-ref",
                "www.prefect.io/events",
                "www.prefect.io/blog/prefect-global-coordination-plane",
            ],
        ),
        GitHubRepoLoader(
            repo="PrefectHQ/prefect",
            include_globs=["flows/"],
        ),
        GitHubRepoLoader(
            repo="PrefectHQ/prefect",
            include_globs=["src/prefect/*.py"],
        ),
        GitHubRepoLoader(
            repo="PrefectHQ/prefect-background-task-examples",
            include_globs=["**/*.py", "**/*.md"],
        ),
        GitHubRepoLoader(
            repo="zzstoatzz/prefect-pack",
            include_globs=["**/*.py", "**/*.md"],
        ),
        GitHubRepoLoader(
            repo="zzstoatzz/prefect-monorepo",
            include_globs=["**/*.py", "**/*.md", "**/*.yaml"],
        ),
    ],
    "marvin": [
        GitHubRepoLoader(
            repo="PrefectHQ/marvin",
            include_globs=["examples/", "docs/"],
        ),
    ],
}


def _cache_key_with_invalidation(
    context: TaskRunContext, parameters: dict[str, Any]
) -> str:
    return f"{task_input_hash(context, parameters)}:{os.getenv('RAGGY_CACHE_VERSION', '0')}"


@task(
    retries=1,
    retry_delay_seconds=3,
    cache_key_fn=_cache_key_with_invalidation,
    cache_expiration=timedelta(days=1),
    task_run_name="Run {loader.__class__.__name__}",
    persist_result=True,
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
    namespace_loaders: Sequence[Loader],
    reset: bool = False,
    batch_size: int = 100,
    max_concurrent: int = 8,
):
    """Flow updating vectorstore with info from the Prefect community."""
    documents: list[Document] = [
        doc
        for future in run_loader.map(quote(namespace_loaders))  # type: ignore
        for doc in future.result()  # type: ignore
    ]

    print(f"Gathered {len(documents)} documents from 🌎")

    with TurboPuffer(namespace=namespace) as tpuf:
        if reset:
            task(tpuf.reset)()
            print(f"RESETTING: Deleted all documents from tpuf ns {namespace!r}.")

        task(tpuf.upsert_batched).submit(  # type: ignore
            documents=documents,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
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
