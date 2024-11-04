# Tutorial

## Loading documents

```python
from raggy.loaders.web import SitemapLoader

raggy_documentation_loader = SitemapLoader(
    urls=["https://zzstoatzz.github.io/raggy/sitemap.xml"],
    exclude=["api-ref", "/events/"],
)
documents = await raggy_documentation_loader.load()

print(documents[0])
```

## Adding documents to a vectorstore

!!! note "New in 0.2.0"
    Vectorstore operations are now synchronous by default, with async batching available via `upsert_batched`.

```python
from raggy.vectorstores.tpuf import TurboPuffer

with TurboPuffer(namespace="my_documents") as vectorstore:
    # Synchronous operation
    vectorstore.upsert(documents)

    # Async batched usage for large document sets
    await vectorstore.upsert_batched(
        documents,
        batch_size=100,
        max_concurrent=8
    )
```

## Querying the vectorstore

```python
from raggy.vectorstores.tpuf import query_namespace, multi_query_tpuf

# Single query
result = query_namespace("how do I get started with raggy?")
print(result)

# Multiple related queries for better coverage
result = multi_query_tpuf([
    "how to install raggy",
    "basic raggy usage",
    "raggy getting started"
])
print(result)
```

## Real-world examples

- [Chat with a GitHub repo](https://github.com/zzstoatzz/raggy/blob/main/examples/chat_with_X/repo.py)
- [Chat with a website](https://github.com/zzstoatzz/raggy/blob/main/examples/chat_with_X/website.py)
- [Refresh a vectorstore](https://github.com/zzstoatzz/raggy/blob/main/examples/refresh_vectorstore/tpuf_namespace.py)
