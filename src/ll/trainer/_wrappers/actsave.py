import contextlib
from logging import getLogger
from typing import Any, cast

from lightning.pytorch import LightningModule
from typing_extensions import override

from ...actsave import ActSave
from ...model.config import BaseConfig
from ._base import LightningModuleWrapper, MethodName, StepFunction

log = getLogger(__name__)


class ActSaveWrapper(LightningModuleWrapper):
    @override
    def name(self):
        return "act_save"

    @override
    def wrapped_step(
        self,
        module: LightningModule,
        fn: StepFunction,
        fn_name: MethodName,
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> Any:
        with contextlib.ExitStack() as stack:
            # Ensure this is enabled in the `trainer` config
            hparams = cast(BaseConfig, module.hparams)
            if (
                actsave_config := hparams.trainer.actsave
            ) is not None and actsave_config.enabled:
                stack.enter_context(ActSave.context(fn_name))

            return fn(batch, batch_idx, *args, **kwargs)
