"""test custom excerpt templates in loaders."""

import pytest
from jinja2 import Environment

from raggy.documents import Document
from raggy.loaders.web import URLLoader


@pytest.fixture
def simple_template():
    """a template without fluff."""
    env = Environment(enable_async=True)
    return env.from_string("{{ excerpt_text }}")


async def test_loader_custom_excerpt_template(simple_template):
    """loaders should accept custom excerpt templates."""

    # mock a simple url loader that doesn't actually fetch
    class MockURLLoader(URLLoader):
        async def load_url(self, url, client):
            return Document(
                text="short test content",
                metadata={"title": "test", "link": url},
            )

    loader = MockURLLoader(
        urls=["http://example.com"],
        create_excerpts=True,
        excerpt_template=simple_template,
        chunk_tokens=10,
    )

    docs = await loader.load()

    assert len(docs) > 0
    # with simple template, text should just be the content, no fluff
    assert "This is an excerpt from a document" not in docs[0].text
    assert "short test content" in docs[0].text


async def test_loader_default_template_unchanged():
    """loaders without custom template should use default behavior."""

    class MockURLLoader(URLLoader):
        async def load_url(self, url, client):
            return Document(
                text="test content here",
                metadata={"title": "test", "link": url},
            )

    loader = MockURLLoader(
        urls=["http://example.com"],
        create_excerpts=True,
    )

    docs = await loader.load()

    assert len(docs) > 0
    # default template should still add the excerpt header
    assert "This is an excerpt from a document" in docs[0].text


async def test_sitemap_loader_propagates_template(simple_template, monkeypatch):
    """SitemapLoader should pass excerpt_template to child loaders."""
    from raggy.loaders.web import SitemapLoader

    # mock sitemap_search to return fake urls
    async def mock_sitemap_search(url):
        return ["http://example.com/page1", "http://example.com/page2"]

    monkeypatch.setattr("raggy.loaders.web.sitemap_search", mock_sitemap_search)

    # mock URLLoader.load_url to return test docs
    from raggy.loaders.web import URLLoader

    async def mock_load_url(self, url, client):
        return Document(
            text="sitemap test content",
            metadata={"title": "test", "link": url},
        )

    monkeypatch.setattr(URLLoader, "load_url", mock_load_url)

    loader = SitemapLoader(
        urls=["http://example.com/sitemap.xml"],
        create_excerpts=True,
        excerpt_template=simple_template,
        chunk_tokens=20,
    )

    docs = await loader.load()

    assert len(docs) > 0
    # should use clean template, not default
    assert "This is an excerpt from a document" not in docs[0].text
    assert "sitemap test content" in docs[0].text


async def test_html_loader_formatting(monkeypatch):
    """HTMLLoader should optionally preserve formatting."""
    from raggy.loaders.web import HTMLLoader

    mock_html = "<html><body><h1>Test Title</h1><p>content here</p></body></html>"

    # mock html_parser to verify include_formatting is passed
    calls = []

    def mock_parser(html, include_formatting=False):
        calls.append(include_formatting)
        return "## Test Title\n\ncontent here" if include_formatting else "Test Title content here"

    import raggy.settings
    original_parser = raggy.settings.settings.html_parser
    raggy.settings.settings.html_parser = mock_parser

    async def mock_load_url(self, url, client):
        from httpx import Request, Response
        request = Request("GET", url)
        response = Response(200, content=mock_html, request=request)
        response._request = request
        return await self.response_to_document(response)

    monkeypatch.setattr(HTMLLoader, "load_url", mock_load_url)

    # test with formatting disabled
    loader = HTMLLoader(urls=["http://example.com"], create_excerpts=False, include_formatting=False)
    docs = await loader.load()
    assert calls[-1] is False
    assert "##" not in docs[0].text

    # test with formatting enabled
    loader = HTMLLoader(urls=["http://example.com"], create_excerpts=False, include_formatting=True)
    docs = await loader.load()
    assert calls[-1] is True
    assert "##" in docs[0].text

    raggy.settings.settings.html_parser = original_parser
