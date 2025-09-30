#!/usr/bin/env -S uv run -s -q --with-editable .
"""Populate the Prefect docs TurboPuffer namespace used by the MCP server."""

import argparse
import asyncio
import inspect
from typing import Iterable

from jinja2 import Environment

from raggy.documents import Document
from raggy.loaders.web import SitemapLoader
from raggy.vectorstores.tpuf import TurboPuffer

# clean template without fluff
jinja_env = Environment(enable_async=True)
CLEAN_TEMPLATE = jinja_env.from_string(inspect.cleandoc("""{{ excerpt_text }}"""))

DEFAULT_NAMESPACE = "prefect-docs-for-mcp"
DEFAULT_SITEMAPS = (
    "https://docs.prefect.io/sitemap.xml",
    # "https://prefect.io/sitemap.xml",
)
DEFAULT_EXCLUDES = ("www.prefect.io/events",)


async def load_documents(
    urls: Iterable[str], exclude: Iterable[str] | None
) -> list[Document]:
    loader = SitemapLoader(
        urls=list(urls),
        exclude=list(exclude or ()),
        excerpt_template=CLEAN_TEMPLATE,
        chunk_tokens=600,  # larger chunks, cleaner content
        use_ai_agent_headers=True,  # request markdown via Accept: text/plain
    )
    return await loader.load()


async def populate_namespace(
    namespace: str,
    reset: bool,
    batch_size: int,
    max_concurrent: int,
    urls: Iterable[str],
    exclude: Iterable[str] | None,
) -> None:
    url_list = list(urls)
    exclude_list = list(exclude or ())
    if not (documents := await load_documents(url_list, exclude_list)):
        raise SystemExit("No documents retrieved from sitemap sources.")

    print(f"Fetched {len(documents)} documents from {len(url_list)} sitemap sources.")

    with TurboPuffer(namespace=namespace) as tpuf:
        if reset:
            tpuf.reset()
            print(f"Reset namespace {namespace!r} before upserting.")

        if not tpuf.ok():
            print(
                f"Namespace {namespace!r} does not exist yet; it will be created on write."
            )

        await tpuf.upsert_batched(
            documents=documents,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            skip_errors=True,  # continue on oversized documents
        )

    print(f"Populated namespace {namespace!r} with {len(documents)} documents.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate the Prefect docs TurboPuffer namespace for MCP usage.",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help="TurboPuffer namespace to populate.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the namespace before inserting new documents.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents per batch when writing to TurboPuffer.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum concurrent TurboPuffer write batches.",
    )
    parser.add_argument(
        "--sitemap",
        action="append",
        dest="sitemaps",
        help="Additional sitemap URL to include. Can be provided multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        dest="excludes",
        help="Substring to exclude from sitemap URLs. Can be provided multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = list(DEFAULT_SITEMAPS)
    if args.sitemaps:
        urls.extend(args.sitemaps)

    excludes = list(DEFAULT_EXCLUDES)
    if args.excludes:
        excludes.extend(args.excludes)

    asyncio.run(
        populate_namespace(
            namespace=args.namespace,
            reset=args.reset,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            urls=urls,
            exclude=excludes,
        )
    )


if __name__ == "__main__":
    main()
