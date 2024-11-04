# Ingest Strategy

When building RAG applications, you often need to load and refresh content from multiple sources. This can involve:
- Expensive API calls
- Large document processing
- Concurrent embedding operations

We use [Prefect](https://docs.prefect.io) to handle these challenges, giving us:

- Automatic caching of expensive operations
- Concurrent processing with backpressure
- Observability and retries

Let's look at a real example that demonstrates these concepts.

## Building a Knowledge Base

```python
from datetime import timedelta
import httpx
from prefect import flow, task
from prefect.tasks import task_input_hash

from raggy.loaders.github import GitHubRepoLoader
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.tpuf import TurboPuffer

# Cache based on content changes
def get_last_modified(context, parameters):
    """Only reload if the content has changed."""
    try:
        return httpx.head(parameters["urls"][0]).headers.get("Last-Modified", "")
    except Exception:
        return None

@task(
    cache_key_fn=get_last_modified,
    cache_expiration=timedelta(hours=24),
    retries=2,
)
async def gather_documents(urls: list[str]):
    return await SitemapLoader(urls=urls).load()

@flow
async def refresh_knowledge():
    # Load from multiple sources
    documents = []
    for loader in [
        SitemapLoader(urls=["https://docs.prefect.io/sitemap.xml"]),
        GitHubRepoLoader(repo="PrefectHQ/prefect", include_globs=["README.md"]),
    ]:
        documents.extend(await gather_documents(loader))

    # Store efficiently with concurrent embedding
    with TurboPuffer(namespace="knowledge") as tpuf:
        await tpuf.upsert_batched(
            documents,
            batch_size=100,  # tune based on document size
            max_concurrent=8  # tune based on rate limits
        )
```

This example shows key patterns:

1. Content-aware caching (`Last-Modified` headers, commit SHAs, etc)
2. Automatic retries for resilience
3. Concurrent processing with backpressure
4. Efficient batching of embedding operations

See the [refresh examples](https://github.com/zzstoatzz/raggy/tree/main/examples/refresh_vectorstore) for complete implementations using both Chroma and TurboPuffer.

## Performance Tips

For production workloads:
```python
@task(
    retries=2,
    retry_delay_seconds=[3, 60],  # exponential backoff
    cache_expiration=timedelta(days=1),
    persist_result=True,  # save results to storage
)
async def gather_documents(loader):
    return await loader.load()
```

See [Prefect's documentation](https://docs.prefect.io/latest/concepts/tasks/) for more on task configuration and caching strategies.