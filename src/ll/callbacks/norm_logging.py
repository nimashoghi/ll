from logging import getLogger
from typing import Literal, cast

import torch
import torch.nn as nn
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.optim import Optimizer
from typing_extensions import override

from .base import CallbackConfigBase

log = getLogger(__name__)


def grad_norm(
    module: nn.Module,
    norm_type: float | int | str,
    group_separator: str = "/",
    grad: bool = True,
) -> dict[str, torch.Tensor | float]:
    """Compute each parameter's gradient's norm and their overall norm.

    The overall norm is computed over all gradients together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.
    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(
            f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}"
        )

    if grad:
        norms = {
            f"grad_{norm_type}_norm{group_separator}{name}": p.grad.data.norm(norm_type)
            for name, p in module.named_parameters()
            if p.grad is not None
        }
        if norms:
            total_norm = torch.tensor(list(norms.values())).norm(norm_type)
            norms[f"grad_{norm_type}_norm_total"] = total_norm
    else:
        norms = {
            f"param_{norm_type}_norm{group_separator}{name}": p.data.norm(norm_type)
            for name, p in module.named_parameters()
            if p.grad is not None
        }
        if norms:
            total_norm = torch.tensor(list(norms.values())).norm(norm_type)
            norms[f"param_{norm_type}_norm_total"] = total_norm

    return norms


def _to_norm_type(log_grad_norm_per_param: float | str | Literal[True]):
    norm_type = 2.0
    if log_grad_norm_per_param is not True:
        norm_type = log_grad_norm_per_param
    return norm_type


def compute_norm(
    pl_module: LightningModule,
    optimizer: Optimizer | None = None,
    p: float | str = 2.0,
    *,
    grad: bool,
) -> torch.Tensor:
    if optimizer is not None:
        tensors = [
            cast(torch.Tensor, p.grad if grad else p)
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
    else:
        tensors = [
            p.grad if grad else p for p in pl_module.parameters() if p.grad is not None
        ]

    if not tensors:
        return torch.tensor(0.0, device=pl_module.device)

    return torch.norm(torch.stack([torch.norm(g, p=p) for g in tensors]), p=p)


class NormLoggingCallback(Callback):
    def __init__(self, config: "NormLoggingConfig"):
        super().__init__()

        self.config = config

    def _perform_norm_logging(
        self,
        pl_module: LightningModule,
        optimizer: Optimizer,
        prefix: str,
    ):
        # Gradient norm logging
        if log_grad_norm := self.config.log_grad_norm:
            norm = compute_norm(
                pl_module,
                optimizer,
                _to_norm_type(log_grad_norm),
                grad=True,
            )
            pl_module.log(f"{prefix}grad_norm", norm, on_step=True, on_epoch=False)
        if log_grad_norm_per_param := self.config.log_grad_norm_per_param:
            norm_type = _to_norm_type(log_grad_norm_per_param)
            pl_module.log_dict(
                {
                    f"{prefix}{k}": v
                    for k, v in grad_norm(pl_module, norm_type, grad=True).items()
                }
            )

        # Parameter norm logging
        if log_param_norm := self.config.log_param_norm:
            norm = compute_norm(
                pl_module,
                optimizer,
                _to_norm_type(log_param_norm),
                grad=False,
            )
            pl_module.log(f"{prefix}param_norm", norm, on_step=True, on_epoch=False)
        if log_param_norm_per_param := self.config.log_param_norm_per_param:
            norm_type = _to_norm_type(log_param_norm_per_param)
            pl_module.log_dict(
                {
                    f"{prefix}{k}": v
                    for k, v in grad_norm(pl_module, norm_type, grad=False).items()
                }
            )

    @override
    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        if len(trainer.optimizers) == 1:
            optimizer = trainer.optimizers[0]
            self._perform_norm_logging(pl_module, optimizer, prefix="train/")
        else:
            for i, optimizer in enumerate(trainer.optimizers):
                self._perform_norm_logging(
                    pl_module, optimizer, prefix=f"train/optimizer_{i}/"
                )


class NormLoggingConfig(CallbackConfigBase):
    name: Literal["norm_logging"] = "norm_logging"

    log_grad_norm: bool | str | float = False
    """If enabled, will log the gradient norm (averaged across all model parameters) to the logger."""
    log_grad_norm_per_param: bool | str | float = False
    """If enabled, will log the gradient norm for each model parameter to the logger."""

    log_param_norm: bool | str | float = False
    """If enabled, will log the parameter norm (averaged across all model parameters) to the logger."""
    log_param_norm_per_param: bool | str | float = False
    """If enabled, will log the parameter norm for each model parameter to the logger."""

    def __bool__(self):
        return any(
            v
            for v in (
                self.log_grad_norm,
                self.log_grad_norm_per_param,
                self.log_param_norm,
                self.log_param_norm_per_param,
            )
        )

    @override
    def construct_callbacks(self, root_config):
        if not self:
            return

        yield NormLoggingCallback(self)
