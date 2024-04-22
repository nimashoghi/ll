import contextlib
from typing import Any, Literal, cast

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import TypeAlias, override

from ..model.config import BaseConfig
from ._saver import ActSave

Stage: TypeAlias = Literal["train", "validation", "test", "predict"]


class ActSaveCallback(Callback):
    def __init__(self):
        super().__init__()

        self._active_contexts: dict[Stage, contextlib._GeneratorContextManager] = {}

    def _on_start(
        self,
        stage: Stage,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        hparams = cast(BaseConfig, pl_module.hparams)
        if not hparams.trainer.actsave:
            return

        # If we have an active context manager for this stage, exit it
        if active_contexts := self._active_contexts.get(stage):
            active_contexts.__exit__(None, None, None)

        # Signal to ActSave that we're starting a new stage
        ActSave._start_stage(stage)

        # Enter a new context manager for this stage
        self._active_contexts[stage] = ActSave.context(stage)

    def _on_end(
        self,
        stage: Stage,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        hparams = cast(BaseConfig, pl_module.hparams)
        if not hparams.trainer.actsave:
            return

        # If we have an active context manager for this stage, exit it
        if active_contexts := self._active_contexts.get(stage):
            active_contexts.__exit__(None, None, None)

    @override
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        return self._on_start("train", trainer, pl_module)

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        return self._on_end("train", trainer, pl_module)

    @override
    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._on_start("validation", trainer, pl_module)

    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._on_end("validation", trainer, pl_module)

    @override
    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._on_start("test", trainer, pl_module)

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._on_end("test", trainer, pl_module)

    @override
    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._on_start("predict", trainer, pl_module)

    @override
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self._on_end("predict", trainer, pl_module)
