import re
from typing import Callable, Self
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from httpx import AsyncClient, Response
from pydantic import Field

import raggy
from raggy.documents import Document, document_to_excerpts
from raggy.loaders.base import Loader, MultiLoader
from raggy.utilities.asyncutils import run_concurrent_tasks
from raggy.utilities.collections import batched

user_agent = UserAgent()


def ensure_http(url):
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url


async def sitemap_search(sitemap_url) -> list[str]:
    async with AsyncClient() as client:
        response = await client.get(sitemap_url, follow_redirects=True)
        response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    return [loc.text for loc in soup.find_all("loc")]


class WebLoader(Loader):
    document_type: str = "web page"
    headers: dict = Field(default_factory=dict, repr=False)

    async def get_headers(self) -> dict:
        return {"User-Agent": user_agent.random, **self.headers}


class URLLoader(WebLoader):
    """
    Given a list of URLs, loads whatever it finds there.

    Attributes:
        urls: The URLs to load from.
        create_excerpts: Whether to split documents into excerpts. Defaults to True.
    """

    source_type: str = "url"
    urls: list[str] = Field(default_factory=list)
    create_excerpts: bool = Field(default=True)

    async def load(self) -> list[Document]:
        headers = await self.get_headers()
        async with AsyncClient(
            headers=headers, timeout=30, follow_redirects=True
        ) as client:

            async def load_url_task(url):
                try:
                    return await self.load_url(url, client)
                except Exception as e:
                    self.logger.error(e)
                    return None

            documents = await run_concurrent_tasks(
                [lambda u=url: load_url_task(u) for url in self.urls], max_concurrent=30
            )

        final_documents = []
        for d in documents:
            if d is not None:
                if self.create_excerpts:
                    final_documents.extend(await document_to_excerpts(d))
                else:
                    final_documents.append(d)
        return final_documents

    async def load_url(self, url, client) -> Document | None:
        response = await client.get(url, follow_redirects=True)

        if not response.status_code == 200:
            self.logger.warning_style(
                f"Received status {response.status_code} from {url}", "red"
            )

        # check for a meta refresh redirect in the response content
        soup = BeautifulSoup(response.text, "html.parser")
        meta_refresh = soup.find(
            "meta", attrs={"http-equiv": re.compile(r"refresh", re.I)}
        )
        if meta_refresh and isinstance(meta_refresh, BeautifulSoup.Tag):
            content = meta_refresh.get("content", "")
            if isinstance(content, str):
                redirect_url_match = re.search(r"url=([\S]+)", content, re.I)
                if redirect_url_match:
                    redirect_url = redirect_url_match.group(1)
                    # join base url with relative url
                    redirect_url = urljoin(str(response.url), redirect_url)
                    # Now ensure the URL includes the protocol
                    redirect_url = ensure_http(redirect_url)
                    response = await client.get(redirect_url, follow_redirects=True)

        document = await self.response_to_document(response)
        if document:
            self.logger.info(f"Loaded document from {url}")
        else:
            self.logger.warning_style(f"Could not load document from {url}", "red")
        return document

    async def response_to_document(self, response: Response) -> Document:
        """Convert an HTTP response to a Document."""
        return Document(
            text=await self.get_document_text(response),
            metadata=dict(
                link=str(response.url),
                source=self.source_type,
                document_type=self.document_type,
            ),
        )

    async def get_document_text(self, response: Response) -> str:
        return response.text


class HTMLLoader(URLLoader):
    """
    A loader that loads HTML, optionally converting it to markdown or stripping tags
    """

    async def get_document_text(self, response: Response) -> str:
        text = await super().get_document_text(response)
        return raggy.settings.html_parser(text)


class SitemapLoader(URLLoader):
    """A loader that loads URLs from a sitemap.
    Attributes:
        include: A list of strings or regular expressions. Only URLs that match one of these will be included.
        exclude: A list of strings or regular expressions. URLs that match one of these will be excluded.
        url_loader: The loader to use for loading the URLs.
        create_excerpts: Whether to split documents into excerpts. Defaults to True.
    Examples:
        Load all URLs from a sitemap:
        ```python
        from raggy.loaders.web import SitemapLoader
        loader = SitemapLoader(urls=["https://controlflow.ai/sitemap.xml"])
        documents = await loader.load()
        print(documents)
        ```
    """

    include: list[str | re.Pattern] = Field(default_factory=list)
    exclude: list[str | re.Pattern] = Field(default_factory=list)
    url_loader: URLLoader = Field(default_factory=HTMLLoader)
    url_processor: Callable[[str], str] = lambda x: x  # noqa: E731
    create_excerpts: bool = Field(default=True)

    async def _get_loader(self: Self) -> MultiLoader:
        urls = await run_concurrent_tasks(
            [lambda u=url: self.load_sitemap(u) for url in self.urls], max_concurrent=5
        )
        return MultiLoader(
            loaders=[
                type(self.url_loader)(
                    urls=url_batch,
                    headers=await self.get_headers(),
                    create_excerpts=self.create_excerpts,
                )  # type: ignore
                for url_batch in batched(
                    [self.url_processor(u) for url_list in urls for u in url_list],  # type: ignore
                    10,
                )
            ]
        )

    async def load(self) -> list[Document]:
        loader = await self._get_loader()
        return await loader.load()

    async def load_sitemap(self, url: str) -> list[str]:
        def is_included(url: str) -> bool:
            if not self.include:
                return True
            return any(
                (isinstance(i, str) and i in url)
                or (isinstance(i, re.Pattern) and re.search(i, url))
                for i in self.include
            )

        def is_excluded(url: str) -> bool:
            return any(
                (isinstance(e, str) and e in url)
                or (isinstance(e, re.Pattern) and re.search(e, url))
                for e in self.exclude
            )

        return [
            url
            for url in await sitemap_search(url)
            if is_included(url) and not is_excluded(url)
        ]
