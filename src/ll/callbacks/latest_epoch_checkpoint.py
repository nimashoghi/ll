import logging
from pathlib import Path

from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Checkpoint
from typing_extensions import override

log = logging.getLogger(__name__)


class LatestEpochCheckpoint(Checkpoint):
    DEFAULT_FILENAME = "latest_epoch{epoch:02d}_step{step:04d}.ckpt"

    def __init__(
        self,
        dirpath: _PATH,
        filename: str | None = None,
        save_weights_only: bool = False,
    ):
        super().__init__()

        self._dirpath = Path(dirpath)
        self._filename = filename or self.DEFAULT_FILENAME
        self._save_weights_only = save_weights_only

        # Also, we hold a reference to the last checkpoint path
        # to be able to remove it when a new checkpoint is saved.
        self._last_ckpt_path: Path | None = None

    def _ckpt_path(self, trainer: Trainer):
        return self._dirpath / self._filename.format(
            epoch=trainer.current_epoch, step=trainer.global_step
        )

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # Remove the last checkpoint if it exists
        if self._last_ckpt_path is not None:
            trainer.strategy.remove_checkpoint(self._last_ckpt_path)

        # Save the new checkpoint
        filepath = self._ckpt_path(trainer)
        trainer.save_checkpoint(filepath, self._save_weights_only)
        self._last_ckpt_path = filepath
