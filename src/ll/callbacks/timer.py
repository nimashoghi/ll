import logging
import time
from typing import Any, Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

from .base import CallbackConfigBase

log = logging.getLogger(__name__)


class EpochTimer(Callback):
    def __init__(self):
        super().__init__()

        self._start_time: dict[str, float] = {}
        self._elapsed_time: dict[str, float] = {}
        self._total_batches: dict[str, int] = {}

    @override
    def on_train_epoch_start(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self._start_time["train"] = time.monotonic()
        self._total_batches["train"] = 0

    @override
    def on_train_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._total_batches["train"] += 1

    @override
    def on_train_epoch_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self._elapsed_time["train"] = time.monotonic() - self._start_time["train"]
        if trainer.is_global_zero:
            self._log_epoch_info("train")

    @override
    def on_validation_epoch_start(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self._start_time["val"] = time.monotonic()
        self._total_batches["val"] = 0

    @override
    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._total_batches["val"] += 1

    @override
    def on_validation_epoch_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self._elapsed_time["val"] = time.monotonic() - self._start_time["val"]
        if trainer.is_global_zero:
            self._log_epoch_info("val")

    @override
    def on_test_epoch_start(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self._start_time["test"] = time.monotonic()
        self._total_batches["test"] = 0

    @override
    def on_test_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._total_batches["test"] += 1

    @override
    def on_test_epoch_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self._elapsed_time["test"] = time.monotonic() - self._start_time["test"]
        if trainer.is_global_zero:
            self._log_epoch_info("test")

    @override
    def on_predict_epoch_start(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self._start_time["predict"] = time.monotonic()
        self._total_batches["predict"] = 0

    @override
    def on_predict_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._total_batches["predict"] += 1

    @override
    def on_predict_epoch_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        self._elapsed_time["predict"] = time.monotonic() - self._start_time["predict"]
        if trainer.is_global_zero:
            self._log_epoch_info("predict")

    def _log_epoch_info(self, stage: str) -> None:
        if (elapsed_time := self._elapsed_time.get(stage)) is None:
            return
        total_batches = self._total_batches[stage]
        log.critical(
            f"Epoch {stage.capitalize()} Summary: Elapsed Time: {elapsed_time:.2f} seconds | "
            f"Total Batches: {total_batches}"
        )

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "elapsed_time": self._elapsed_time,
            "total_batches": self._total_batches,
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._elapsed_time = state_dict["elapsed_time"]
        self._total_batches = state_dict["total_batches"]


class EpochTimerConfig(CallbackConfigBase):
    name: Literal["epoch_timer"] = "epoch_timer"

    @override
    def construct_callbacks(self, root_config):
        yield EpochTimer()
