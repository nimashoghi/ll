import logging
from pathlib import Path

log = logging.getLogger(__name__)


def init_python_logging(
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
        log_file = log_save_dir / "log.log"
        log_file.touch(exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file))

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


def pretty(
    *,
    lovely_tensors: bool = True,
    lovely_numpy: bool = True,
    log_level: int | str | None = logging.INFO,
    log_save_dir: Path | None = None,
    rich_log_handler: bool = True,
):
    init_python_logging(
        lovely_tensors=lovely_tensors,
        lovely_numpy=lovely_numpy,
        rich=rich_log_handler,
        log_level=log_level,
        log_save_dir=log_save_dir,
    )


def lovely(
    *,
    lovely_tensors: bool = True,
    lovely_numpy: bool = True,
    log_level: int | str | None = logging.INFO,
    log_save_dir: Path | None = None,
    rich_log_handler: bool = True,
):
    pretty(
        lovely_tensors=lovely_tensors,
        lovely_numpy=lovely_numpy,
        log_level=log_level,
        log_save_dir=log_save_dir,
        rich_log_handler=rich_log_handler,
    )
