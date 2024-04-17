from logging import getLogger
from typing import Any, Protocol, runtime_checkable

from lightning.pytorch.loggers import Logger

log = getLogger(__name__)


@runtime_checkable
class Finalizable(Protocol):
    def finish(self) -> Any: ...


def finalize_loggers(loggers: list[Logger]):
    for logger in loggers:
        if not isinstance(logger, Finalizable):
            continue
        logger.finish()
