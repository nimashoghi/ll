import datetime
import logging
import os
from typing import Any

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import OnExceptionCheckpoint as _OnExceptionCheckpoint
from typing_extensions import override

log = logging.getLogger(__name__)


class OnExceptionCheckpoint(_OnExceptionCheckpoint):
    @property
    @override
    def ckpt_path(self) -> str:
        ckpt_path = super().ckpt_path

        # Remve the extension and add the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ckpt_path, ext = os.path.splitext(ckpt_path)
        return f"{ckpt_path}_{timestamp}{ext}"

    @override
    def on_exception(self, trainer: Trainer, *_: Any, **__: Any) -> None:
        # We override this to checkpoint the model manually,
        # without calling the dist barrier.

        # trainer.save_checkpoint(self.ckpt_path)

        if trainer.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached to the Trainer. Did you call"
                " `Trainer.save_checkpoint()` before calling `Trainer.{fit,validate,test,predict}`?"
            )
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        trainer.strategy.save_checkpoint(
            checkpoint, self.ckpt_path, storage_options=None
        )
        # self.strategy.barrier("Trainer.save_checkpoint") # <-- This is disabled

    @override
    def teardown(self, trainer: Trainer, *_: Any, **__: Any) -> None:
        trainer.strategy.remove_checkpoint(self.ckpt_path)
