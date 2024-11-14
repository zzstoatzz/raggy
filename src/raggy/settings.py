from typing import Callable

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_html_parser(html: str) -> str:
    """The default HTML parser using trafilatura or bs4 as a fallback.
    Args:
        html: The HTML to parse.

    Returns:
        The parsed HTML.
    """
    import trafilatura
    from bs4 import BeautifulSoup

    trafilatura_config = trafilatura.settings.use_config()  # type: ignore
    # disable signal, so it can run in a worker thread
    # https://github.com/adbar/trafilatura/issues/202
    trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    return (
        trafilatura.extract(html, config=trafilatura_config)
        or BeautifulSoup(html, "html.parser").get_text()
    )


class ChromaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHROMA_", env_file=".env", extra="ignore"
    )

    cloud_tenant: str = Field(
        default="default",
        description="The tenant to use for the Chroma Cloud client.",
    )
    cloud_database: str = Field(
        default="default",
        description="The database to use for the Chroma Cloud client.",
    )
    cloud_api_key: SecretStr | None = Field(
        default=None,
        description="The API key to use for the Chroma Cloud client.",
    )


class Settings(BaseSettings):
    """The settings for Raggy.

    Attributes:
        html_parser: The function to use for parsing HTML.
        log_level: The log level to use.
        log_verbose: Whether to log verbose messages.
        openai_chat_completions_model: The OpenAI model to use for chat completions.
        openai_embeddings_model: The OpenAI model to use for creating embeddings.

    """

    model_config = SettingsConfigDict(
        env_prefix="RAGGY_",
        env_file=("~/.raggy/.env", ".env"),
        extra="allow",
        validate_assignment=True,
    )
    max_concurrent_tasks: int = Field(
        default=50, gt=3, description="The maximum number of concurrent tasks to run."
    )
    html_parser: Callable[[str], str] = default_html_parser

    log_level: str = Field(
        default="INFO",
        description="The log level to use.",
    )

    log_verbose: bool = Field(
        default=False,
        description=(
            "Whether to log verbose messages, such as full API requests and responses."
        ),
    )
    openai_chat_completions_model: str = Field(
        default="gpt-3.5-turbo",
        description="The OpenAI model to use for chat completions.",
    )

    openai_embeddings_model: str = Field(
        default="text-embedding-3-small",
        description="The OpenAI model to use for creating embeddings.",
    )

    chroma: ChromaSettings = Field(default_factory=ChromaSettings)  # type: ignore

    @field_validator("log_level", mode="after")
    @classmethod
    def set_log_level(cls, v):
        from raggy.utilities.logging import setup_logging

        setup_logging(level=v)
        return v


settings = Settings()
