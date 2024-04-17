from logging import getLogger
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lightning.pytorch.loggers import Logger

from ..model.config import BaseConfig

log = getLogger(__name__)


def default_root_dir(
    config: BaseConfig,
    *,
    logs_dirname: str = "lightning_logs",
):
    if (base_dir := config.trainer.default_root_dir) is None:
        base_dir = Path.cwd()
    base_path = (base_dir / logs_dirname).resolve().absolute()
    path = base_path / config.id
    path.mkdir(parents=True, exist_ok=True)
    return path


@runtime_checkable
class Finalizable(Protocol):
    def finish(self) -> Any: ...


def finalize_loggers(loggers: list[Logger]):
    for logger in loggers:
        if not isinstance(logger, Finalizable):
            continue
        logger.finish()
