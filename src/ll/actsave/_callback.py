import contextlib
from typing import TYPE_CHECKING, Literal, cast

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from typing_extensions import TypeAlias, override

from ._saver import ActSave

if TYPE_CHECKING:
    from ..model.config import BaseConfig

Stage: TypeAlias = Literal["train", "validation", "test", "predict"]


class ActSaveCallback(Callback):
    def __init__(self):
        super().__init__()

        self._active_contexts: dict[Stage, contextlib._GeneratorContextManager] = {}

    def _on_start(self, stage: Stage, trainer: Trainer, pl_module: LightningModule):
        hparams = cast("BaseConfig", pl_module.hparams)
        if not hparams.trainer.actsave:
            return

        # If we have an active context manager for this stage, exit it
        if active_contexts := self._active_contexts.get(stage):
            active_contexts.__exit__(None, None, None)

        # Enter a new context manager for this stage
        context = ActSave.context(stage)
        context.__enter__()
        self._active_contexts[stage] = context

    def _on_end(self, stage: Stage, trainer: Trainer, pl_module: LightningModule):
        hparams = cast("BaseConfig", pl_module.hparams)
        if not hparams.trainer.actsave:
            return

        # If we have an active context manager for this stage, exit it
        if active_contexts := self._active_contexts.get(stage):
            active_contexts.__exit__(None, None, None)

    @override
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        return self._on_start("train", trainer, pl_module)

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        return self._on_end("train", trainer, pl_module)

    @override
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        return self._on_start("validation", trainer, pl_module)

    @override
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        return self._on_end("validation", trainer, pl_module)

    @override
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        return self._on_start("test", trainer, pl_module)

    @override
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        return self._on_end("test", trainer, pl_module)

    @override
    def on_predict_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        return self._on_start("predict", trainer, pl_module)

    @override
    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        return self._on_end("predict", trainer, pl_module)
