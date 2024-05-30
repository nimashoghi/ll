import math
import warnings
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import override

from ..config import Field
from ._base import LRSchedulerConfigBase


class LinearWarmupCosineAnnealingLR(LRScheduler):
    _get_lr_called_within_step: bool

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        should_restart: bool = True,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.should_restart = should_restart

        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        if not self.should_restart and self.last_epoch >= self.max_epochs:
            return [self.eta_min] * len(self.base_lrs)

        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


class LinearWarmupCosineDecayLRSchedulerConfig(LRSchedulerConfigBase):
    name: Literal["linear_warmup_cosine_decay"] = "linear_warmup_cosine_decay"

    warmup_epochs: int = Field(ge=0)
    r"""The number of epochs for the linear warmup phase.
    The learning rate is linearly increased from `warmup_start_lr` to the initial learning rate over this number of epochs."""

    max_epochs: int = Field(gt=0)
    r"""The total number of epochs.
    The learning rate is decayed to `min_lr` over this number of epochs."""

    warmup_start_lr_factor: float = 0.0
    r"""The initial learning rate for the linear warmup phase, as a factor of the initial learning rate.
    The learning rate is linearly increased from this value to the initial learning rate over `warmup_epochs` epochs."""

    min_lr_factor: float = 0.0
    r"""The minimum learning rate, as a factor of the initial learning rate.
    The learning rate is decayed to this value over `max_epochs` epochs."""

    annealing: bool = False
    r"""Whether to restart the learning rate schedule after `max_epochs` epochs.
    If `False`, the learning rate will be decayed to `min_lr` over `max_epochs` epochs, and then the learning rate will be set to `min_lr` for all subsequent epochs.
    If `True`, the learning rate will be decayed to `min_lr` over `max_epochs` epochs, and then the learning rate will be increased back to the initial learning rate over `max_epochs` epochs, and so on (this is called a cosine annealing schedule)."""

    @override
    def metadata(self) -> LRSchedulerConfigBase.Metadata:
        return {
            "interval": "step",
        }

    @override
    def create_scheduler_impl(self, optimizer, lightning_module, lr):
        num_steps_per_epoch = self.compute_num_steps_per_epoch(lightning_module)
        warmup_steps = self.warmup_epochs * num_steps_per_epoch
        max_steps = self.max_epochs * num_steps_per_epoch
        warmup_start_lr = self.warmup_start_lr_factor * lr
        min_lr = self.min_lr_factor * lr

        # Create the scheduler
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=warmup_steps,
            max_epochs=max_steps,
            warmup_start_lr=warmup_start_lr,
            eta_min=min_lr,
            should_restart=self.annealing,
        )
        return scheduler
