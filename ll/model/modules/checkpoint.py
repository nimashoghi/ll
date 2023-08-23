from typing import cast

from lightning.pytorch.callbacks import OnExceptionCheckpoint
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin


class CheckpointMixin(mixin_base_type(CallbackModuleMixin)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def _cb():
            nonlocal self

            config = cast(BaseConfig, self.config)
            if config.trainer.on_exception_checkpoint:
                yield OnExceptionCheckpoint(".")

        self.register_callback(_cb)
