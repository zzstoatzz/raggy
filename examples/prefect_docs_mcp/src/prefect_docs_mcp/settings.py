"""Settings for the Prefect docs MCP server."""

from typing import Sequence

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrefectDocsSettings(BaseSettings):
    """Configuration options for the Prefect docs MCP server."""

    model_config = SettingsConfigDict(
        env_prefix="PREFECT_DOCS_MCP_",
        env_file=("~/.raggy/.env", ".env"),
        extra="ignore",
    )

    namespace: str = Field(
        default="prefect-3",
        description="TurboPuffer namespace containing Prefect documentation embeddings.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of results to return from the vector store.",
    )
    max_tokens: int = Field(
        default=900,
        ge=100,
        le=2000,
        description="Maximum number of tokens to include when concatenating excerpts.",
    )
    include_attributes: Sequence[str] = Field(
        default_factory=lambda: ["text", "title", "link"],
        description="Attributes to request from TurboPuffer rows when searching.",
    )


settings = PrefectDocsSettings()

__all__ = ["settings", "PrefectDocsSettings"]
