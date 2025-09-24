#!/usr/bin/env python
"""Manual test harness for the Prefect Docs MCP server.

This script exercises the SearchPrefect tool using FastMCP's in-process client.
It patches the TurboPuffer vector store with a test double so we can verify
query/response handling (including BadRequest fallbacks) without relying on
external services.

Usage:
    uv run python examples/prefect_docs_mcp/run_client_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from turbopuffer import BadRequestError

from fastmcp import Client

# Ensure the example package is importable without installation
EXAMPLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_ROOT / "src"))

from prefect_docs_mcp.server import prefect_docs_mcp  # noqa: E402


class FakeTurboPuffer:
    """Simplistic TurboPuffer stand-in for local testing."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._attempt = 0

    def __enter__(self) -> "FakeTurboPuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def query(self, *, text: str, top_k: int, include_attributes: list[str] | None = None):
        self._attempt += 1
        if self._attempt == 1 and include_attributes:
            raise BadRequestError(
                "attribute \"link\" not found in schema",
                response=httpx.Response(400, json={"error": "missing attribute"}),
                body={"error": "missing attribute"},
            )

        rows = [
            SimpleNamespace(
                id="doc-1",
                text="Prefect flows let you orchestrate tasks.",
                metadata={"title": "Prefect Flows", "link": "https://docs.prefect.io/flows"},
                score=0.12,
                title="Prefect Flows",
                link="https://docs.prefect.io/flows",
            )
        ]
        return SimpleNamespace(rows=rows)


async def main(mock: bool, query: str, top_k: int) -> None:
    print("Running Prefect Docs MCP test client...")
    call_args = {"query": query, "top_k": top_k}

    if mock:
        from unittest.mock import patch

        with patch("prefect_docs_mcp.server.TurboPuffer", FakeTurboPuffer):
            async with Client(prefect_docs_mcp) as client:
                result = await client.call_tool("SearchPrefect", call_args)
    else:
        async with Client(prefect_docs_mcp) as client:
            result = await client.call_tool("SearchPrefect", call_args)

    print("Tool call output:\n", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise the Prefect Docs MCP tool")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a fake TurboPuffer backend instead of the live service",
    )
    parser.add_argument("--query", default="build a flow", help="Query text to send to the tool")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to request")
    args = parser.parse_args()

    asyncio.run(main(mock=args.mock, query=args.query, top_k=args.top_k))
