# Ingest Strategy

When building RAG applications, you often need to load and refresh content from multiple sources. This can involve:

- Expensive API calls
- Large document processing
- Concurrent embedding operations

We use [Prefect](https://docs.prefect.io) to handle these challenges, giving us:

- Automatic caching of expensive operations
- Concurrent processing
- Observability and retries

Let's look at a real example that demonstrates these concepts.

## Building a Knowledge Base

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "prefect",
#     "raggy[tpuf]",
# ]
# ///

from itertools import chain
from datetime import timedelta
from prefect import flow, task, unmapped
from prefect.cache_policies import INPUTS

from raggy.documents import Document
from raggy.loaders.base import Loader
from raggy.loaders.github import GitHubRepoLoader
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.tpuf import TurboPuffer

@task(
    cache_policy=INPUTS,
    cache_expiration=timedelta(hours=24),
    task_run_name="gather documents using {loader.__class__.__name__}",
    retries=2,
)
async def gather_documents(loader: Loader) -> list[Document]:
    return await loader.load()

@flow(flow_run_name="refresh knowledge in {namespace} from urls {urls} and repos {repos}")
def refresh_knowledge(
    urls: list[str] | None = None,
    repos: list[str] | None = None,
    include_globs: list[str] | None = None,
    namespace: str = "knowledge",
):

    # Load from multiple sources
    _2d_list_of_documents = gather_documents.map(
        [
            SitemapLoader(urls=urls),
            *[
                GitHubRepoLoader(repo=repo, include_globs=include_globs or ["README.md"])
                for repo in repos
            ],
        ]
    ).result()

    # batch embedding and upserts to the vector store
    with TurboPuffer(namespace=namespace) as tpuf:
        task(tpuf.upsert_batched).submit(
            documents=list(chain.from_iterable(_2d_list_of_documents)),
            batch_size=unmapped(100),  # tune based on document size
            max_concurrent=unmapped(8),  # tune based on rate limits
        ).wait()

if __name__ == "__main__":
    refresh_knowledge(
        urls=["https://docs.prefect.io/sitemap.xml"],
        repos=["PrefectHQ/prefect"],
        include_globs=["README.md"],
        namespace="test-knowledge",
    )
```

This example shows key patterns:

1. Automatic retries for resilience
2. Concurrent processing
3. Efficient batching of embedding operations

See the [refresh examples](https://github.com/zzstoatzz/raggy/tree/main/examples/refresh_vectorstore) for complete implementations using both Chroma and TurboPuffer.

## Performance Tips

For production workloads:
```python
@task(
    retries=2,
    retry_delay_seconds=[3, 60],  # exponential backoff
    cache_expiration=timedelta(days=1),
    cache_policy=INPUTS,  # for example, hash based on provided parameters
    persist_result=True,  # save results to storage
)
async def gather_documents(loader: Loader) -> list[Document]:
    return await loader.load()
```

See [Prefect's documentation](https://docs.prefect.io/latest/concepts/tasks/) for more on task configuration and caching strategies.