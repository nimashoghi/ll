from logging import getLogger
from typing import Literal

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from typing_extensions import override

from .base import CallbackConfigBase

log = getLogger(__name__)


def finite_checks(
    module: LightningModule,
    nonfinite_grads: bool = True,
    none_grads: bool = False,
):
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            if none_grads:
                log.critical(f"Parameter {name} ({param.shape}) has None gradients")
            continue

        if not nonfinite_grads or torch.isfinite(param.grad.float()).all():
            continue

        has_nan = torch.isnan(param.grad.float()).any()
        has_inf = torch.isinf(param.grad.float()).any()
        kinds = [
            "NaN" if has_nan else None,
            "Inf" if has_inf else None,
        ]
        kinds = " and ".join(prop for prop in kinds if prop is not None)
        log.critical(f"{name} ({param.shape}) has {kinds} gradients")


class FiniteChecksCallback(Callback):
    def __init__(
        self,
        *,
        nonfinite_grads: bool = True,
        none_grads: bool = True,
    ):
        super().__init__()

        self._nonfinite_grads = nonfinite_grads
        self._none_grads = none_grads

    @override
    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        finite_checks(
            pl_module,
            nonfinite_grads=self._nonfinite_grads,
            none_grads=self._none_grads,
        )


class FiniteChecksConfig(CallbackConfigBase):
    name: Literal["finite_checks"] = "finite_checks"

    nonfinite_grads: bool = True
    """Whether to check for non-finite (i.e. NaN or Inf) gradients"""

    none_grads: bool = True
    """Whether to check for None gradients"""

    @override
    def construct_callbacks(self, root_config):
        yield FiniteChecksCallback(
            nonfinite_grads=self.nonfinite_grads,
            none_grads=self.none_grads,
        )
