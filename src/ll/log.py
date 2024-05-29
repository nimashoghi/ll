import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _default_log_handlers(log_save_dir: Path):
    # Capture the logs to `dirpath`/log.log
    log_file = log_save_dir / "log.log"
    log_file.touch(exist_ok=True)
    return logging.FileHandler(log_file)


def init_logging(
    *,
    lovely_tensors: bool = False,
    lovely_numpy: bool = False,
    rich: bool = False,
    log_level: int | str | None = logging.INFO,
    log_save_dir: Path | None = None,
):
    if lovely_tensors:
        try:
            import lovely_tensors as _lovely_tensors

            _lovely_tensors.monkey_patch()
        except ImportError:
            log.warning(
                "Failed to import `lovely_tensors`. Ignoring pretty PyTorch tensor formatting"
            )

    if lovely_numpy:
        try:
            import lovely_numpy as _lovely_numpy

            _lovely_numpy.set_config(repr=_lovely_numpy.lovely)
        except ImportError:
            log.warning(
                "Failed to import `lovely_numpy`. Ignoring pretty numpy array formatting"
            )

    log_handlers: list[logging.Handler] = []
    if log_save_dir:
        log_handlers.append(_default_log_handlers(log_save_dir))

    if rich:
        try:
            from rich.logging import RichHandler

            log_handlers.append(RichHandler())
        except ImportError:
            log.warning(
                "Failed to import rich. Falling back to default Python logging."
            )

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=log_handlers,
    )