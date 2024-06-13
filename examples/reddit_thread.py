from functools import lru_cache

import marvin  # pip install marvin
import praw  # pip install praw
from marvin.utilities.logging import get_logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from raggy.documents import Document, document_to_excerpts
from raggy.vectorstores.tpuf import TurboPuffer, query_namespace

logger = get_logger("reddit_thread_example")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()


def create_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=getattr(settings, "reddit_client_id"),
        client_secret=getattr(settings, "reddit_client_secret"),
        user_agent="testscript by /u/_n80n8",
    )


@lru_cache
def read_thread(submission_id: str):
    logger.info(f"Reading thread {submission_id}")
    submission = create_reddit_client().submission(submission_id)

    text_buffer = ""
    text_buffer += f"Title: {submission.title}\n"
    text_buffer += f"Selftext: {submission.selftext}\n"

    submission.comments.replace_more(limit=None)  # Retrieve all comments
    for comment in submission.comments.list():
        text_buffer += "\n---\n"
        text_buffer += f"Comment Text: {comment.body}\n"

    return text_buffer


async def save_thread(thread_text: str):
    logger.info("Saving thread")
    chunked_documents = await document_to_excerpts(Document(text=thread_text))

    async with TurboPuffer(namespace="reddit_thread") as tpuf:
        await tpuf.upsert(chunked_documents)

    return "Thread saved!"


@marvin.fn
def summarize_results(relevant_excerpts: str) -> str:  # type: ignore[empty-body]
    """give a summary of the relevant excerpts"""


async def main():
    logger.info("Starting Reddit thread example")
    thread_text = read_thread("1bpf4lr")  # r/Chicago thread
    await save_thread(thread_text)

    query = "how do people feel about the return of the water taxis?"
    results = await query_namespace(query, namespace="reddit_thread")
    print(summarize_results(results))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

"""
The consensus among several comments is a positive reaction to the Chicago Water Taxi resuming 7-day service, which hadn't occurred since 2019. People express that this marks the city's recovery and share their enthusiasm for the convenient transportation it provides. Some commenters also discuss potential improvements and expansions, such as increased service locations and eco-friendly options like electric hydrofoils.
"""
