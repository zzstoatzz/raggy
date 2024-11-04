# Example Gallery

Here are some practical examples of using `raggy` in real-world scenarios.

## Chat with Content

Ye old "chat your data" examples.

#### Chat with a Website

```bash
uv run examples/chat_with_X/website.py "let's chat about docs.astral.sh/uv"
```

#### Chat with a GitHub Repo

```bash
uv run examples/chat_with_X/repo.py "let's chat about astral-sh/uv"
```

## Refresh Vectorstores

A `prefect` flow to gather documents from sources of knowledge, embed them and put them in a vectorstore.

#### Refresh TurboPuffer

```bash
uv run examples/refresh_vectorstore/tpuf_namespace.py
```

#### Refresh Chroma

```bash
uv run examples/refresh_vectorstore/chroma_collection.py
```
