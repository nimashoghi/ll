import datetime
import logging
import os

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
