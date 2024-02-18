import re
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    TypeVar,
)

import tiktoken
import xxhash

import raggy

T = TypeVar("T")

if TYPE_CHECKING:
    pass


def rm_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def rm_text_after(text: str, substring: str) -> str:
    return (
        text[: start + len(substring)]
        if (start := text.find(substring)) != -1
        else text
    )


def extract_keywords(text: str) -> list[str]:
    """Extract keywords from the given text using the yake library.

    Args:
        text: The text to extract keywords from.

    Returns:
        list[str]: The keywords extracted from the text.

    Raises:
        ImportError: If yake is not installed.

    Example:
        Extract keywords from a text:
        ```python
        from raggy.utilities.text import extract_keywords

        text = "This is a sample text from which we will extract keywords."
        keywords = extract_keywords(text)
        print(keywords) # ['keywords', 'sample', 'text', 'extract']
        ```
    """
    try:
        import yake
    except ImportError:
        raise ImportError(
            "yake is required for keyword extraction. Please install it with"
            " `pip install `raggy[rag]` or `pip install yake`."
        )

    kw = yake.KeywordExtractor(
        lan="en",
        n=1,
        dedupLim=0.9,
        dedupFunc="seqm",
        windowsSize=1,
        top=10,
        features=None,
    )

    return [k[0] for k in kw.extract_keywords(text)]


@lru_cache(maxsize=2048)
def hash_text(*text: str) -> str:
    """Hash the given text using the xxhash algorithm.

    Args:
        text: The text to hash.

    Returns:
        str: The hash of the text.

    Example:
        Hash a single text:
        ```python
        from raggy.utilities.text import hash_text

        text = "This is a sample text."
        hash_ = hash_text(text)
        print(hash_) # 4a2db845d20188ce069196726a065a09
        ```
    """
    bs = [t.encode() if not isinstance(t, bytes) else t for t in text]
    return xxhash.xxh3_128_hexdigest(b"".join(bs))


def get_encoding_for_model(model: str | None = None) -> tiktoken.Encoding:
    """Get the `tiktoken` encoding for the specified model.

    Args:
        model: The model to get the encoding for. If not provided, the default
            chat completions model is used (as specified in `raggy.settings`).
            If an invalid model is provided, 'gpt-3.5-turbo' is used.

    Returns:
        tiktoken.Encoding: The encoding for the specified model.

    Example:
        Get the encoding for the default chat completions model:
        ```python
        from raggy.utilities.text import get_encoding_for_model

        encoding = get_encoding_for_model() # 'gpt-3.5-turbo' by default
        ```
    """
    if model is None:
        model = raggy.settings.openai_chat_completions_model
    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        return tiktoken.encoding_for_model("gpt-3.5-turbo")


def tokenize(text: str, model: str | None = None) -> list[int]:
    """
    Tokenizes the given text using the specified model.

    Args:
        text: The text to tokenize.
        model: The model to use for tokenization. If not provided,
            the default model is used.

    Returns:
        list[int]: The tokenized text as a list of integers.
    """
    return get_encoding_for_model(model).encode(text)


def detokenize(tokens: list[int], model: str | None = None) -> str:
    """
    Detokenizes the given tokens using the specified model.

    Args:
        tokens: The tokens to detokenize.
        model: The model to use for detokenization. If not provided,
            the default model is used.

    Returns:
        str: The detokenized text.
    """
    return get_encoding_for_model(model).decode(tokens)


def count_tokens(text: str, model: str | None = None) -> int:
    """
    Counts the number of tokens in the given text using the specified model.

    Args:
        text: The text to count tokens in.
        model: The model to use for token counting. If not provided,
            the default model is used.

    Returns:
        int: The number of tokens in the text.
    """
    return len(tokenize(text, model=model))


def slice_tokens(text: str, n_tokens: int) -> str:
    """Slices the given text to the specified number of tokens.

    Args:
        text: The text to slice.
        n_tokens: The number of tokens to slice the text to.

    Returns:
        str: The sliced text.

    Example:
        Slice a text to the first 50 tokens:
        ```python
        from raggy.utilities.text import slice_tokens

        text = "This is a sample text."*100
        sliced_text = slice_tokens(text, 5)
        print(sliced_text) # 'This is a sample text.'
        ```
    """
    return detokenize(tokenize(text)[:n_tokens])


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: float | None = None,
    last_chunk_threshold: float | None = None,
) -> list[str]:
    """
    Split a text into a list of strings. Chunks are split by tokens.

    Args:
        text: The text to split.
        chunk_size: The number of tokens in each chunk.
        chunk_overlap: The fraction of overlap between chunks.
        last_chunk_threshold: If the last chunk is less than this fraction of
            the chunk_size, it will be added to the prior chunk

    Returns:
        list[str]: The list of chunks.

    Example:
        Split a text into chunks of 5 tokens with 10% overlap:
        ```python
        from raggy.utilities.text import split_text

        text = "This is a sample text."*3
        chunks = split_text(text, 5, 0.1)
        print(chunks) # ['This is a sample text', '.This is a sample text', '.This is a sample text.']
        ```
    """
    if chunk_overlap is None:
        chunk_overlap = 0.1
    if chunk_overlap < 0 or chunk_overlap > 1:
        raise ValueError("chunk_overlap must be between 0 and 1")
    if last_chunk_threshold is None:
        last_chunk_threshold = 0.25

    tokens = tokenize(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - int(chunk_overlap * chunk_size)):
        chunks.append((tokens[i : i + chunk_size], len(detokenize(tokens[:i]))))

    # if the last chunk is too small, merge it with the previous chunk
    if len(chunks) > 1 and len(chunks[-1][0]) < chunk_size * last_chunk_threshold:
        chunks[-2][0].extend(chunks.pop(-1)[0])

    return [detokenize(chunk) for chunk, _ in chunks]
