from typing import overload

from openai import APIConnectionError, AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

import raggy


@overload
async def create_openai_embeddings(
    input_: str,
    timeout: int = 60,
    model: str = raggy.settings.openai_embeddings_model,
) -> list[float]:
    ...


@overload
async def create_openai_embeddings(
    input_: list[str],
    timeout: int = 60,
    model: str = raggy.settings.openai_embeddings_model,
) -> list[list[float]]:
    ...


@retry(
    retry=retry_if_exception_type(APIConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
)
async def create_openai_embeddings(
    input_: str | list[str],
    timeout: int = 60,
    model: str = raggy.settings.openai_embeddings_model,
) -> list[float] | list[list[float]]:
    """Create OpenAI embeddings for a list of texts.

    Args:
        input_: The input text or list of texts to embed
        timeout: The maximum time to wait for the request to complete
        model: The model to use for the embeddings. Defaults to the value
            of `raggy.settings.openai_embeddings_model`, which is "text-embedding-3-small"
            by default

    Returns:
        The embeddings for the input text or list of texts

    Raises:
        TypeError: If input_ is not a str or a list of str

    Examples:
        Create an embedding for a single text:
        ```python
        from raggy.utilities.embeddings import create_openai_embeddings

        embedding = await create_openai_embeddings("Hello, world!")
        ```

        Create embeddings for a list of texts:
        ```python
        from raggy.utilities.embeddings import create_openai_embeddings

        embeddings = await create_openai_embeddings([
            "Hello, world!",
            "Goodbye, world!",
        ])
        ```
    """

    if isinstance(input_, str):
        input_ = [input_]
    elif not isinstance(input_, list):
        raise TypeError(
            f"Expected input to be a str or a list of str, got {type(input_).__name__}."
        )

    embedding: CreateEmbeddingResponse = await AsyncOpenAI().embeddings.create(
        input=input_, model=model, timeout=timeout
    )

    if len(embedding.data) == 1:
        return embedding.data[0].embedding

    return [data.embedding for data in embedding.data]
