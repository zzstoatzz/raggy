from pydantic import BaseModel, ConfigDict, PrivateAttr

from raggy.utilities.logging import RaggyLogger, get_logger


class Vectorstore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _in_context: bool = PrivateAttr(False)

    @property
    def logger(self) -> RaggyLogger:
        return get_logger(self.__class__.__name__)

    async def __aenter__(self):
        self._in_context = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self._in_context = False
