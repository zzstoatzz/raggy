from openai import APIConnectionError, AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

import raggy


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
    """Create OpenAI embeddings for a list of texts."""

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
