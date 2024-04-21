from logging import getLogger
from typing import Any, cast

from lightning.pytorch import LightningModule
from typing_extensions import override

from ..exception import SkipBatch
from ..model.config import BaseConfig
from ..util.lightning_module_wrapper import (
    LightningModuleWrapper,
    MethodName,
    StepFunction,
)

log = getLogger(__name__)


class SkipBatchWrapper(LightningModuleWrapper):
    @override
    def name(self):
        return "skip_batch"

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
        # Ensure this is enabled in the `trainer` config.
        hparams = cast(BaseConfig, module.hparams)
        if not hparams.trainer.supports_skip_batch_exception:
            return fn(batch, batch_idx, *args, **kwargs)

        try:
            return fn(batch, batch_idx, *args, **kwargs)
        except SkipBatch as e:
            log.info(
                f"[{fn_name}] @ [step={module.global_step}, batch={batch_idx}]: Skipping batch due to SkipBatch exception: {e}"
            )

            if (skip_batch_fn := getattr(module, f"skipped_{fn_name}", None)) is None:
                log.critical(
                    f"SkipBatch exception was raised, but no method `skipped_{fn_name}` was found in the LightningModule. "
                    f"Please implement this method to handle the SkipBatch exception. "
                    f"Otherwise, nothing will be returned from the `{fn_name}` method."
                )
                return None

            return skip_batch_fn(batch, batch_idx, *args, **kwargs)
