"""Prefect documentation MCP server backed by Raggy and TurboPuffer."""

import json
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field
from turbopuffer import NotFoundError

from prefect_docs_mcp.settings import settings
from raggy.utilities.text import slice_tokens
from raggy.vectorstores.tpuf import TurboPuffer

prefect_docs_mcp = FastMCP(
    "Prefect Docs MCP",
    instructions=(
        "Expose the Prefect documentation via a single semantic search tool backed by Raggy."
    ),
    dependencies=[
        "prefect_docs_mcp@git+https://github.com/zzstoatzz/raggy.git#subdirectory=examples/prefect_docs_mcp"
    ],
)


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a TurboPuffer row response into a plain dict."""

    if isinstance(row, dict):
        data = dict(row)
    elif hasattr(row, "model_dump"):
        data = row.model_dump()  # type: ignore[call-arg]
    elif hasattr(row, "__dict__"):
        data = dict(row.__dict__)
    else:
        data = {"text": str(row)}

    # Standardize known fields and coerce JSON metadata
    metadata = data.get("metadata")
    if isinstance(metadata, str):
        try:
            data["metadata"] = json.loads(metadata)
        except json.JSONDecodeError:
            pass

    return data


def _normalize_score(value: Any) -> float | None:
    """Coerce TurboPuffer scores into floats when present."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@prefect_docs_mcp.tool(name="SearchPrefect")
def search_prefect(
    query: Annotated[
        str,
        Field(
            description=(
                "A natural language query to search against Prefect documentation and examples."
            )
        ),
    ],
    top_k: Annotated[
        int | None,
        Field(
            ge=1,
            le=20,
            description="Override for the number of search results to return.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Search the Prefect documentation vector store for relevant passages."""

    if not query.strip():
        raise ValueError("Query must not be empty.")

    result_limit = top_k or settings.top_k
    snippet_token_budget = max(120, settings.max_tokens // max(result_limit, 1))

    try:
        with TurboPuffer(namespace=settings.namespace) as tpuf:
            response = tpuf.query(
                text=query,
                top_k=result_limit,
                include_attributes=list(dict.fromkeys(settings.include_attributes)),
            )
    except NotFoundError:
        rows: list[Any] = []
    except ValueError as exc:  # Configuration issues like missing API key
        raise RuntimeError(
            "TurboPuffer configuration error. Ensure API key and namespace are set."
        ) from exc
    else:
        rows = list(response.rows or [])

    results: list[dict[str, Any]] = []
    for row in rows:
        data = _row_to_dict(row)
        snippet = data.get("text") or data.get("content")
        snippet_text = (
            slice_tokens(str(snippet), snippet_token_budget) if snippet else ""
        )

        result_payload: dict[str, Any] = {
            "snippet": snippet_text,
            "id": data.get("id"),
            "score": _normalize_score(
                data.get("score") or data.get("distance") or data.get("similarity")
            ),
        }

        title = data.get("title") or (
            data.get("metadata", {}).get("title")
            if isinstance(data.get("metadata"), dict)
            else None
        )
        link = data.get("link") or (
            data.get("metadata", {}).get("link")
            if isinstance(data.get("metadata"), dict)
            else None
        )
        if title:
            result_payload["title"] = title
        if link:
            result_payload["link"] = link
        if metadata := data.get("metadata"):
            result_payload["metadata"] = metadata

        results.append(result_payload)

    return {
        "query": query,
        "namespace": settings.namespace,
        "results": results,
    }
