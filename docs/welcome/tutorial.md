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

```python
from raggy.vectorstores.tpuf import Turbopuffer

async with Turbopuffer() as vectorstore: # uses default `raggy` namespace
    await vectorstore.upsert(documents)
```

## Querying the vectorstore

```python
from raggy.vectorstores.tpuf import query_namespace

print(await query_namespace("how do I get started with raggy?"))
```

## Real-world example

See [this example](https://github.com/zzstoatzz/raggy/blob/main/examples/refresh_chroma/refresh_collection.py) I use to refresh a chatbot that knows about `prefect`.

