# /// script
# dependencies = [
#     "raggy",
#     "rich",
# ]
# ///

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from raggy.documents import Document, DocumentMetadata
from raggy.loaders.web import SitemapLoader

console = Console()


async def main(urls: list[str]) -> list[Document]:
    loader = SitemapLoader(urls=urls, create_excerpts=False)
    docs = await loader.load()
    console.print(f"\n[bold green]✓[/] Scraped {len(docs)} documents\n")

    if docs:
        doc = docs[0]

        assert isinstance(doc.metadata, DocumentMetadata)
        title = Text(f"Document from: {doc.metadata.link}", style="bold blue")

        preview = doc.text[:500] + "..." if len(doc.text) > 500 else doc.text

        console.print(
            Panel(
                Text.from_markup(preview),
                title=title,
                border_style="green",
                padding=(1, 2),
            )
        )

    return docs


if __name__ == "__main__":
    WEBSITE_URLS = ["https://prefect.io/blog/sitemap.xml"]
    documents = asyncio.run(main(WEBSITE_URLS))
