from logging import getLogger
from typing import Protocol, cast, runtime_checkable

import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin

log = getLogger(__name__)


@runtime_checkable
class _HasWandbLogModuleProtocol(Protocol):
    def wandb_log_module(self) -> nn.Module | None: ...


class WandbWatchCallback(Callback):
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
        config = cast(BaseConfig, pl_module.hparams)
        if (
            not config.trainer.logging.enabled
            or (wandb_config := config.trainer.logging.wandb) is None
            or not wandb_config.watch
        ):
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
            log=cast(str, wandb_config.watch.log),
            log_freq=wandb_config.watch.log_freq,
            log_graph=wandb_config.watch.log_graph,
        )
        setattr(pl_module, "_model_watched", True)


class WandbWrapperMixin(mixin_base_type(CallbackModuleMixin)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_callback(lambda: WandbWatchCallback())
