# /// script
# dependencies = [
#   "marvin",
#   "praw",
#   "raggy[tpuf]",
# ]
# ///

from functools import lru_cache

import marvin  # type: ignore
import praw  # type: ignore
from marvin.utilities.logging import get_logger  # type: ignore
from pydantic_settings import BaseSettings, SettingsConfigDict

from raggy.documents import Document, document_to_excerpts
from raggy.vectorstores.tpuf import TurboPuffer, query_namespace

logger = get_logger("reddit_thread_example")  # type: ignore


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()


def create_reddit_client() -> praw.Reddit:  # type: ignore
    return praw.Reddit(  # type: ignore
        client_id=getattr(settings, "reddit_client_id"),
        client_secret=getattr(settings, "reddit_client_secret"),
        user_agent="testscript by /u/_n80n8",
    )


@lru_cache
def read_thread(submission_id: str) -> str:
    logger.info(f"Reading thread {submission_id}")  # type: ignore
    submission: praw.models.Submission = create_reddit_client().submission(  # type: ignore
        submission_id
    )

    text_buffer = ""
    text_buffer += f"Title: {submission.title}\n"  # type: ignore
    text_buffer += f"Selftext: {submission.selftext}\n"  # type: ignore

    submission.comments.replace_more(limit=None)  # type: ignore
    for comment in submission.comments.list():  # type: ignore
        text_buffer += "\n---\n"
        text_buffer += f"Comment Text: {comment.body}\n"  # type: ignore

    return text_buffer


@marvin.fn  # type: ignore
def summarize_results(relevant_excerpts: str) -> str:  # type: ignore[empty-body]
    """give a summary of the relevant excerpts"""


async def main(thread_id: str):
    logger.info("Starting Reddit thread example")  # type: ignore
    thread_text = read_thread(thread_id)
    chunked_documents = await document_to_excerpts(Document(text=thread_text))

    with TurboPuffer(namespace="reddit_thread") as tpuf:
        tpuf.upsert(chunked_documents)

    logger.info("Thread saved!")  # type: ignore

    query = "how do people feel about the return of the water taxis?"
    results = query_namespace(query, namespace="reddit_thread")
    print(summarize_results(results))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(thread_id="1bpf4lr"))  # r/Chicago thread

"""
The consensus among several comments is a positive reaction to the Chicago Water Taxi resuming 7-day service, which hadn't occurred since 2019. People express that this marks the city's recovery and share their enthusiasm for the convenient transportation it provides. Some commenters also discuss potential improvements and expansions, such as increased service locations and eco-friendly options like electric hydrofoils.
"""
