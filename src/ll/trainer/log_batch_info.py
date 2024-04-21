from logging import getLogger
from typing import Any, cast

from lightning.pytorch import LightningModule
from typing_extensions import override

from ..exception import SkipBatch, TrainingError
from ..model.config import BaseConfig
from ..util.lightning_module_wrapper import (
    LightningModuleWrapper,
    MethodName,
    StepFunction,
)

log = getLogger(__name__)


class LogBatchInfoWrapper(LightningModuleWrapper):
    @override
    def name(self):
        return "log_batch_info"

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
        # Make sure `log_batch_info_on_error` is enabled in the `trainer` config
        hparams = cast(BaseConfig, module.hparams)
        if not hparams.trainer.log_batch_info_on_error:
            return fn(batch, batch_idx, *args, **kwargs)

        try:
            return fn(batch, batch_idx, *args, **kwargs)
        except BaseException as e:
            if isinstance(e, SkipBatch):
                # We don't need to handle this case.
                raise e

            # We need to re-raise the exception with more information
            raise TrainingError(
                e,
                batch_idx=batch_idx,
                batch=batch,
                epoch=module.current_epoch,
                global_step=module.global_step,
                training_fn=fn_name,
            ) from e
