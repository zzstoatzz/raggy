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
    bs = [t.encode() if not isinstance(t, bytes) else t for t in text]
    return xxhash.xxh3_128_hexdigest(b"".join(bs))


def get_encoding_for_model(model: str | None = None) -> tiktoken.Encoding:
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
