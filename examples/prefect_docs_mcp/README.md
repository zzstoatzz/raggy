# Prefect Docs MCP Server

This example provides a [FastMCP](https://github.com/jlowin/fastmcp) server that exposes
Prefect documentation through a single semantic search tool powered by
[Raggy](https://github.com/zzstoatzz/raggy) and a TurboPuffer vector store.

## Features

- `SearchPrefect` MCP tool returns high-signal Prefect documentation excerpts
- Powered by the same extraction pipeline as `examples/refresh_vectorstore/tpuf_namespace.py`
- Configurable namespace and result sizes via environment variables

## Prerequisites

1. A TurboPuffer API key with access to the Prefect documentation namespace
2. Python 3.10+

Set the following environment variables (for example in `.env` or `~/.raggy/.env`):

```bash
export TURBOPUFFER_API_KEY="sk-..."
export TURBOPUFFER_REGION="api"            # optional, defaults to `api`
export PREFECT_DOCS_MCP_NAMESPACE="prefect-docs-for-mcp"  # optional override
```

Additional tuning knobs:

- `PREFECT_DOCS_MCP_TOP_K`: default number of results (defaults to `5`)
- `PREFECT_DOCS_MCP_MAX_TOKENS`: aggregate context budget (defaults to `900`)

## Install & Run

```bash
cd examples/prefect_docs_mcp
uv pip install -e .
uv run prefect-docs-mcp
```

Alternatively, use the provided `prefect_docs.fastmcp.json` with the FastMCP CLI or
an MCP-compatible client.

### Quick Test

Test the search functionality using the FastMCP client:

```python
from fastmcp import Client
from prefect_docs_mcp.server import prefect_docs_mcp

async with Client(prefect_docs_mcp) as client:
    response = await client.call_tool("SearchPrefect", {"query": "how to build flows"})
    print(response)
```

## Tool Reference

### `SearchPrefect`

Search the Prefect knowledge base for relevant passages. Accepts a natural-language
`query` and optional `top_k` override. Responses include the namespace, query, and a
list of snippet results with scores, titles, and links when available.

## Minimal Client Example

```python
from fastmcp import Client
from prefect_docs_mcp.server import prefect_docs_mcp

async def demo():
    async with Client(prefect_docs_mcp) as client:
        response = await client.call_tool(
            "SearchPrefect", {"query": "how to build prefect flows"}
        )
        print(response)
```

## Notes

- Populate the default namespace with `examples/prefect_docs_mcp/refresh_namespace.py`
  (see `--help` for options) before pointing the MCP server at it
- The legacy flows under `examples/refresh_vectorstore/` remain handy for broader
  refreshes or scheduled runs
- If the namespace is empty or unavailable the tool returns an empty result list
  rather than raising, enabling graceful fallbacks in clients

## Performance

Server-side metrics show ~99% of query time is spent in TurboPuffer vector search
(~1s mean latency), with response formatting taking <10ms. The bottleneck is the
vector database, not the MCP server code.
