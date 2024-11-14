# /// script
# dependencies = [
#     "raggy",
#     "trafilatura",
# ]
# ///

import asyncio

from bs4 import BeautifulSoup

import raggy
from raggy.loaders.web import SitemapLoader


def html_parser(html: str) -> str:
    import trafilatura

    trafilatura_config = trafilatura.settings.use_config()  # type: ignore
    trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    return (
        trafilatura.extract(html, config=trafilatura_config)
        or BeautifulSoup(html, "html.parser").get_text()
    )


async def main(urls: list[str]):
    raggy.settings.html_parser = html_parser
    loader = SitemapLoader(urls=urls, create_excerpts=False)
    docs = await loader.load()
    print(f"scraped {len(docs)} documents")


if __name__ == "__main__":
    asyncio.run(main(["https://prefect.io/blog/sitemap.xml"]))
