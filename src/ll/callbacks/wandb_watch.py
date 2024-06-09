from logging import getLogger
from typing import Literal, Protocol, cast, runtime_checkable

import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override

from .base import CallbackConfigBase

log = getLogger(__name__)


@runtime_checkable
class _HasWandbLogModuleProtocol(Protocol):
    def wandb_log_module(self) -> nn.Module | None: ...


class WandbWatchCallback(Callback):
    def __init__(self, config: "WandbWatchConfig"):
        super().__init__()

        self.config = config

    @override
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_start(trainer, pl_module)

    @override
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_start(trainer, pl_module)

    @override
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_start(trainer, pl_module)

    @override
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._on_start(trainer, pl_module)

    def _on_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # If not enabled, return
        if not self.config:
            return

        # If we're in fast_dev_run, don't watch the model
        if getattr(trainer, "fast_dev_run", False):
            return

        if (
            logger := next(
                (
                    logger
                    for logger in trainer.loggers
                    if isinstance(logger, WandbLogger)
                ),
                None,
            )
        ) is None:
            log.warning("Could not find wandb logger or module to log")
            return

        if getattr(pl_module, "_model_watched", False):
            return

        # Get which module to log
        if (
            not isinstance(pl_module, _HasWandbLogModuleProtocol)
            or (module := pl_module.wandb_log_module()) is None
        ):
            module = cast(nn.Module, pl_module)

        logger.watch(
            module,
            log=cast(str, self.config.log),
            log_freq=self.config.log_freq,
            log_graph=self.config.log_graph,
        )
        setattr(pl_module, "_model_watched", True)


class WandbWatchConfig(CallbackConfigBase):
    name: Literal["finite_checks"] = "finite_checks"

    enabled: bool = True
    """Enable watching the model for wandb."""

    log: str | None = None
    """Log type for wandb."""

    log_graph: bool = True
    """Whether to log the graph for wandb."""

    log_freq: int = 100
    """Log frequency for wandb."""

    def __bool__(self):
        return self.enabled

    @override
    def construct_callbacks(self, root_config):
        yield WandbWatchCallback(self)
