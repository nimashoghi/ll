from logging import getLogger
from typing import Literal, Protocol, runtime_checkable

import torch
import torchmetrics
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.optim import Optimizer
from typing_extensions import override

from .base import CallbackConfigBase
from .norm_logging import compute_norm

log = getLogger(__name__)


@runtime_checkable
class HasGradSkippedSteps(Protocol):
    grad_skipped_steps: torchmetrics.SumMetric


class GradientSkipping(Callback):
    def __init__(self, config: "GradientSkippingConfig"):
        super().__init__()

        self.config = config

    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if not isinstance(pl_module, HasGradSkippedSteps):
            pl_module.grad_skipped_steps = torchmetrics.SumMetric()

    @override
    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ):
        # This should never happen, but just in case
        if not isinstance(pl_module, HasGradSkippedSteps):
            raise TypeError(
                f"Expected HasGradSkippedSteps, got {type(pl_module)} instead"
            )

        # Skip the step if the global step is less than the start_after_n_steps
        # This is because we want to let AMP adjust the loss scale before we start
        if (
            self.config.start_after_n_steps is not None
            and pl_module.global_step < self.config.start_after_n_steps
        ):
            return

        norm = compute_norm(
            pl_module,
            optimizer,
            self.config.norm_type,
            grad=True,
        )

        # If the norm is NaN/Inf, we don't want to skip the step
        # beacuse AMP checks for NaN/Inf grads to adjust the loss scale.
        if self.config.skip_non_finite and not torch.isfinite(norm).all():
            optimizer.zero_grad()
            pl_module.grad_skipped_steps(1)
            log.warning(
                f"Skipping step at global step {pl_module.global_step} with non-finite norm {norm:.2f}"
            )
        elif (norm > self.config.threshold).any():
            optimizer.zero_grad()
            pl_module.grad_skipped_steps(1)
            log.warning(
                f"Skipping step at global step {pl_module.global_step} with norm {norm:.2f} > {self.config.threshold:.2f}"
            )
        else:
            pl_module.grad_skipped_steps(0)

        pl_module.log(
            "train/grad_skipped_steps",
            pl_module.grad_skipped_steps,
            on_step=True,
            on_epoch=False,
        )


class GradientSkippingConfig(CallbackConfigBase):
    name: Literal["gradient_skipping"] = "gradient_skipping"

    threshold: float
    """Threshold to use for gradient skipping."""

    norm_type: str | float = 2.0
    """Norm type to use for gradient skipping."""

    start_after_n_steps: int | None = 100
    """Number of steps to wait before starting gradient skipping."""

    skip_non_finite: bool = False
    """
    If False, it doesn't skip steps with non-finite norms. This is useful when using AMP, as AMP checks for NaN/Inf grads to adjust the loss scale. Otherwise, skips steps with non-finite norms.

    Should almost always be False, especially when using AMP (unless you know what you're doing!).
    """

    @override
    def construct_callbacks(self, root_config):
        yield GradientSkipping(self)
