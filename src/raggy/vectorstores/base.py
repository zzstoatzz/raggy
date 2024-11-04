from pydantic import BaseModel, ConfigDict, PrivateAttr

from raggy.utilities.logging import RaggyLogger, get_logger


class Vectorstore(BaseModel):
    """Base class for vectorstores.

    Allows for easy logging and context management.

    Attributes:
        _in_context: Whether the vectorstore is currently in a context.

    Example:
        Basic Usage of `Vectorstore`
        ```python
        from raggy.vectorstores.base import Vectorstore

        class MyVectorstore(Vectorstore):
            pass

        with MyVectorstore() as vectorstore:
            ...
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _in_context: bool = PrivateAttr(False)

    @property
    def logger(self) -> RaggyLogger:
        return get_logger(self.__class__.__name__)

    def __enter__(self):
        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._in_context = False
