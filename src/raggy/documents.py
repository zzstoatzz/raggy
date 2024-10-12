import asyncio
import inspect
from functools import partial
from typing import Annotated

from jinja2 import Environment, Template
from pydantic import BaseModel, ConfigDict, Field, model_validator

from raggy.utilities.ids import generate_prefixed_uuid
from raggy.utilities.text import count_tokens, extract_keywords, hash_text, split_text

jinja_env = Environment(enable_async=True)


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    title: str | None = Field(default=None)
    link: str | None = Field(default=None)


class Document(BaseModel):
    """A source of information that is storable & searchable.

    Anything that can be represented as text can be stored as a document:
    web pages, git repos / issues, PDFs, and or just plain text files.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: str = Field(default_factory=partial(generate_prefixed_uuid, "doc"))
    text: str = Field(..., description="Document text content.")

    embedding: list[float] | None = Field(default=None)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    tokens: int | None = Field(default=None)
    keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_tokens(self):
        if self.tokens is None:
            self.tokens = count_tokens(self.text)
        return self

    def __hash__(self) -> int:
        """thanks claude shannon"""
        return int(hash_text(self.text), 16)


EXCERPT_TEMPLATE = jinja_env.from_string(
    inspect.cleandoc(
        """This is an excerpt from a document
        {% if document.metadata %}\n\n# Document metadata
        {{ document.metadata }}
        {% endif %}
        {% if document.keywords %}
        # Document keywords
        {{ document.keywords }}
        {% endif %}
        # Excerpt content: {{ excerpt_text }}
        """
    )
)


async def document_to_excerpts(
    document: Document,
    excerpt_template: Template | None = None,
    chunk_tokens: int = 300,
    overlap: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.1,
    **extra_template_kwargs,
) -> list[Document]:
    """
    Create document excerpts by chunking the document text into regularly-sized
    chunks and adding a "minimap" directory to the top (if document is markdown).

    Args:
        excerpt_template: A jinja2 template to use for rendering the excerpt.
        chunk_tokens: The number of tokens to include in each excerpt.
        overlap: The fraction of overlap between each excerpt.

    """
    if not excerpt_template:
        excerpt_template = EXCERPT_TEMPLATE

    text_chunks: list[str] = split_text(
        text=document.text,
        chunk_size=chunk_tokens,
        chunk_overlap=overlap,
    )

    return await asyncio.gather(
        *[
            _create_excerpt(
                document=document,
                text=text,
                index=i,
                excerpt_template=excerpt_template,
                **extra_template_kwargs,
            )
            for i, text in enumerate(text_chunks)
        ]
    )


async def _create_excerpt(
    document: Document,
    text: str,
    excerpt_template: Template,
    **extra_template_kwargs,
) -> Document:
    keywords = extract_keywords(text)

    excerpt_text = await excerpt_template.render_async(
        document=document,
        excerpt_text=text,
        keywords=", ".join(keywords),
        **extra_template_kwargs,
    )
    return Document(
        parent_document_id=document.id,
        text=excerpt_text,
        keywords=keywords,
        metadata=document.metadata if document.metadata else {},
        tokens=count_tokens(excerpt_text),
    )
