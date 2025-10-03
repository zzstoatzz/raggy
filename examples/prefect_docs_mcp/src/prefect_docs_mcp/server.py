"""Prefect documentation MCP server backed by Raggy and TurboPuffer."""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp import FastMCP
from openai import OpenAIError
from pydantic import Field
from turbopuffer import (
    APIError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
)

from prefect_docs_mcp._internal.metrics import metrics
from prefect_docs_mcp._internal.tpuf import normalize_score, row_to_dict, run_query
from prefect_docs_mcp.settings import settings
from raggy.vectorstores.tpuf import TurboPuffer

prefect_docs_mcp = FastMCP("Prefect Docs MCP", version="0.1.0")


def _build_response(
    query: str, results: list[dict[str, Any]], error: str | None = None
) -> dict[str, Any]:
    """build a response payload for the search tool."""
    payload: dict[str, Any] = {
        "query": query,
        "namespace": settings.namespace,
        "results": results,
    }
    if error:
        payload["error"] = error
    return payload


@prefect_docs_mcp.tool
def search_prefect(
    query: Annotated[
        str,
        Field(
            description=(
                "a search query to find relevant document excerpts from Prefect's knowledgebase"
            )
        ),
    ],
    top_k: Annotated[
        int | None,
        Field(
            ge=1,
            le=20,
            description="How many document excerpts to return.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Search the Prefect knowledgebase for documentation on concepts, usage, and best practices."""

    if not query.strip():
        raise ValueError("Query must not be empty.")

    metrics.increment("search_calls_total")

    result_limit = top_k or settings.top_k
    include_attributes = list(dict.fromkeys(settings.include_attributes))

    with metrics.timer("search_total"):
        try:
            with TurboPuffer(namespace=settings.namespace) as tpuf:
                with metrics.timer("search_vector_query"):
                    response = run_query(
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
                data = row_to_dict(row)
                snippet = data.get("text") or data.get("content")
                # no slicing - content is already chunked at ingestion time
                snippet_text = str(snippet) if snippet else ""

                result_payload: dict[str, Any] = {
                    "snippet": snippet_text,
                    "id": data.get("id"),
                    "score": normalize_score(
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
