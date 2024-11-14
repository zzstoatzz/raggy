# /// script
# dependencies = [
#     "raggy",
#     "trafilatura",
#     "rich",
# ]
# ///

import asyncio

from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import raggy
from raggy.documents import Document, DocumentMetadata
from raggy.loaders.web import SitemapLoader

console = Console()


def html_parser(html: str) -> str:
    import trafilatura

    trafilatura_config = trafilatura.settings.use_config()  # type: ignore
    trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    return (
        trafilatura.extract(html, config=trafilatura_config)
        or BeautifulSoup(html, "html.parser").get_text()
    )


async def main(urls: list[str]) -> list[Document]:
    raggy.settings.html_parser = html_parser

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
