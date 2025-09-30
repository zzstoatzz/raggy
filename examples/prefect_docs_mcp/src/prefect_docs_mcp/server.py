"""Prefect documentation MCP server backed by Raggy and TurboPuffer."""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastmcp import FastMCP
from openai import OpenAIError
from pydantic import Field
from turbopuffer import (
    APIError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)
from turbopuffer.types.namespace_query_response import Row as TurboPufferRow

from prefect_docs_mcp.metrics import metrics
from prefect_docs_mcp.settings import settings
from raggy.vectorstores.tpuf import TurboPuffer

prefect_docs_mcp = FastMCP(
    "Prefect Docs MCP",
    instructions=(
        "Expose the Prefect documentation via a single semantic search tool backed by Raggy."
    ),
)


def _row_to_dict(row: TurboPufferRow | dict[str, Any]) -> dict[str, Any]:
    """Normalize TurboPuffer row objects to plain dictionaries."""

    if isinstance(row, dict):
        data = dict(row)
    else:
        data = row.model_dump(mode="python")
        if row.model_extra:
            data.update(row.model_extra)

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


def _build_response(
    query: str, results: list[dict[str, Any]], error: str | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "namespace": settings.namespace,
        "results": results,
    }
    if error:
        payload["error"] = error
    return payload


def _run_query(
    tpuf: TurboPuffer,
    *,
    query: str,
    top_k: int,
    include_attributes: list[str],
):
    include_arg = include_attributes or None
    try:
        return tpuf.query(
            text=query,
            top_k=top_k,
            include_attributes=include_arg,
        )
    except BadRequestError as exc:
        if include_arg is not None and "include_attributes" in str(exc).lower():
            # Retry without extra attribute hints; schema likely lacks one of them.
            return tpuf.query(text=query, top_k=top_k)
        raise


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

    metrics.increment("search_calls_total")

    result_limit = top_k or settings.top_k
    include_attributes = list(dict.fromkeys(settings.include_attributes))

    with metrics.timer("search_total"):
        try:
            with TurboPuffer(namespace=settings.namespace) as tpuf:
                with metrics.timer("search_vector_query"):
                    response = _run_query(
                        tpuf,
                        query=query,
                        top_k=result_limit,
                        include_attributes=include_attributes,
                    )
        except NotFoundError:
            rows: list[Any] = []
        except (AuthenticationError, PermissionDeniedError) as exc:
            metrics.increment("search_calls_error")
            return _build_response(
                query, [], f"TurboPuffer authentication error: {exc}"
            )
        except APIError as exc:
            metrics.increment("search_calls_error")
            return _build_response(query, [], f"TurboPuffer API error: {exc}")
        except OpenAIError as exc:
            metrics.increment("search_calls_error")
            return _build_response(query, [], f"OpenAI error: {exc}")
        except BadRequestError as exc:
            metrics.increment("search_calls_error")
            return _build_response(query, [], f"TurboPuffer query error: {exc}")
        except ValueError:
            metrics.increment("search_calls_error")
            return _build_response(
                query,
                [],
                "TurboPuffer configuration error. Ensure API key and namespace are set.",
            )
        else:
            rows = list(response.rows or [])

        results: list[dict[str, Any]] = []
        with metrics.timer("search_response_format"):
            for row in rows:
                data = _row_to_dict(row)
                snippet = data.get("text") or data.get("content")
                # no slicing - content is already chunked at ingestion time
                snippet_text = str(snippet) if snippet else ""

                result_payload: dict[str, Any] = {
                    "snippet": snippet_text,
                    "id": data.get("id"),
                    "score": _normalize_score(
                        data.get("score")
                        or data.get("distance")
                        or data.get("similarity")
                    ),
                }

                metadata = (
                    data.get("metadata")
                    if isinstance(data.get("metadata"), dict)
                    else None
                )
                title = data.get("title") or (metadata or {}).get("title")
                link = data.get("link") or (metadata or {}).get("link")
                if title:
                    result_payload["title"] = title
                if link:
                    result_payload["link"] = link
                if metadata:
                    result_payload["metadata"] = metadata

                results.append(result_payload)

        metrics.increment("search_calls_success")

        return _build_response(query, results)
