import logging
import math
from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from typing_extensions import override

log = logging.getLogger(__name__)


class LogEpochCallback(Callback):
    def __init__(self, metric_name: str = "computed_epoch"):
        super().__init__()

        self.metric_name = metric_name

    @override
    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ):
        if trainer.logger is None:
            return

        # If trainer.num_training_batches is not set or is nan/inf, we cannot calculate the epoch
        if (
            not trainer.num_training_batches
            or math.isnan(trainer.num_training_batches)
            or math.isinf(trainer.num_training_batches)
        ):
            log.warning("Trainer has no valid num_training_batches. Cannot log epoch.")
            return

        epoch = pl_module.global_step / trainer.num_training_batches
        pl_module.log(self.metric_name, epoch, on_step=True, on_epoch=False)
