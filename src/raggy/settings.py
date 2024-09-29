from typing import Callable

from bs4 import BeautifulSoup
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_html_parser(html: str) -> str:
    """The default HTML parser. This uses `bs4`'s `html.parser`, which is not very good.
    Like, at all.

    In fact it's really bad. You should definitely set `raggy.settings.html_parser` to a
    `Callable[[str], str]` that parses HTML well.

    Args:
        html: The HTML to parse.

    Returns:
        The parsed HTML.
    """
    from raggy.utilities.logging import get_logger

    get_logger().warning_kv(
        "USING DEFAULT HTML PARSER",
        (
            "BeautifulSoup's html.parser is the default parser and is not very good. "
            "Consider setting `raggy.settings.html_parser` to a `Callable[[str], str]` that parses HTML well."
        ),
        "red",
    )
    return BeautifulSoup(html, "html.parser").get_text()


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
    cloud_api_key: SecretStr = Field(
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
