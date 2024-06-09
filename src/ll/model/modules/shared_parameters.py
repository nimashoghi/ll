from collections.abc import Sequence
from logging import getLogger
from typing import cast

import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from typing_extensions import override

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackRegistrarModuleMixin

log = getLogger(__name__)


def _parameters_to_names(parameters: Sequence[nn.Parameter], model: nn.Module):
    mapping = {id(p): n for n, p in model.named_parameters()}
    return [mapping[id(p)] for p in parameters]


class SharedParametersModuleMixin(mixin_base_type(CallbackRegistrarModuleMixin)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shared_parameters: list[tuple[nn.Parameter, int | float]] = []
        self._warned_shared_parameters = False

        def on_after_backward(_trainer: Trainer, pl_module: LightningModule):
            nonlocal self

            config = cast(BaseConfig, pl_module.hparams)
            if not config.trainer.supports_shared_parameters:
                return

            log.debug(f"Scaling {len(self.shared_parameters)} shared parameters...")
            no_grad_parameters: list[nn.Parameter] = []
            for p, factor in self.shared_parameters:
                if not hasattr(p, "grad") or p.grad is None:
                    no_grad_parameters.append(p)
                    continue

                _ = p.grad.data.div_(factor)

            if no_grad_parameters and not self._warned_shared_parameters:
                no_grad_parameters_str = ", ".join(
                    _parameters_to_names(no_grad_parameters, pl_module)
                )
                log.warning(
                    "The following parameters were marked as shared, but had no gradients: "
                    f"{no_grad_parameters_str}"
                )
                self._warned_shared_parameters = True

            log.debug(
                f"Done scaling shared parameters. (len={len(self.shared_parameters)})"
            )

        self.register_callback(on_after_backward=on_after_backward)

    def register_shared_parameters(
        self, parameters: list[tuple[nn.Parameter, int | float]]
    ):
        for parameter, factor in parameters:
            if not isinstance(parameter, nn.Parameter):
                raise ValueError("Shared parameters must be PyTorch parameters")
            if not isinstance(factor, (int, float)):
                raise ValueError("Factor must be an integer or float")

            self.shared_parameters.append((parameter, factor))

        log.info(f"Registered {len(parameters)} shared parameters")
