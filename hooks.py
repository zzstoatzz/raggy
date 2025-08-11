import logging
from typing import Any

log = logging.getLogger("mkdocs")


def on_pre_build(config: Any, **kwargs: Any) -> None:
    """Add any pre-build hooks here."""
    pass
