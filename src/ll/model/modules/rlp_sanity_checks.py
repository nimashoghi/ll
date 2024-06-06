from collections.abc import Mapping
from logging import getLogger
from typing import cast

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    LRSchedulerTypeUnion,
)
from typing_extensions import Protocol, override, runtime_checkable

from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig
from .callback import CallbackModuleMixin

log = getLogger(__name__)


def _on_train_start_callback(trainer: Trainer, pl_module: LightningModule):
    # If we're in PL's "sanity check" mode, we don't need to run this check
    if trainer.sanity_checking:
        return

    config = cast(BaseConfig, pl_module.hparams)
    if config.trainer.sanity_checking.reduce_lr_on_plateau == "disable":
        return

    # if no lr schedulers, return
    if not trainer.lr_scheduler_configs:
        return

    errors: list[str] = []
    disable_message = (
        "Otherwise, set `config.trainer.sanity_checking.reduce_lr_on_plateau='disable'` "
        "to disable this sanity check."
    )

    for lr_scheduler_config in trainer.lr_scheduler_configs:
        if not lr_scheduler_config.reduce_on_plateau:
            continue

        match lr_scheduler_config.interval:
            case "epoch":
                # we need to make sure that the trainer runs val every `frequency` epochs

                # If `trainer.check_val_every_n_epoch` is None, then Lightning
                # will run val every `int(trainer.val_check_interval)` steps.
                # So, first we need to make sure that `trainer.val_check_interval` is not None first.
                if trainer.check_val_every_n_epoch is None:
                    errors.append(
                        "Trainer is not running validation at epoch intervals "
                        "(i.e., `trainer.check_val_every_n_epoch` is None) but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} is used."
                        f"Please set `config.trainer.check_val_every_n_epoch={lr_scheduler_config.frequency}`. "
                        + disable_message
                    )

                # Second, we make sure that the trainer runs val at least every `frequency` epochs
                if (
                    trainer.check_val_every_n_epoch is not None
                    and lr_scheduler_config.frequency % trainer.check_val_every_n_epoch
                    != 0
                ):
                    errors.append(
                        f"Trainer is not running validation every {lr_scheduler_config.frequency} epochs but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} and frequency={lr_scheduler_config.frequency} is used."
                        f"Please set `config.trainer.check_val_every_n_epoch` to a multiple of {lr_scheduler_config.frequency}. "
                        + disable_message
                    )

            case "step":
                # In this case, we need to make sure that the trainer runs val at step intervals
                # that are multiples of `frequency`.

                # First, we make sure that validation is run at step intervals
                if trainer.check_val_every_n_epoch is not None:
                    errors.append(
                        "Trainer is running validation at epoch intervals "
                        "(i.e., `trainer.check_val_every_n_epoch` is not None) but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} is used."
                        "Please set `config.trainer.check_val_every_n_epoch=None` "
                        f"and `config.trainer.val_check_interval={lr_scheduler_config.frequency}`. "
                        + disable_message
                    )

                # Second, we make sure `trainer.val_check_interval` is an integer
                if not isinstance(trainer.val_check_interval, int):
                    errors.append(
                        f"Trainer is not running validation at step intervals "
                        f"(i.e., `trainer.val_check_interval` is not an integer) but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} is used."
                        "Please set `config.trainer.val_check_interval=None` "
                        f"and `config.trainer.val_check_interval={lr_scheduler_config.frequency}`. "
                        + disable_message
                    )

                # Third, we make sure that the trainer runs val at least every `frequency` steps
                if (
                    isinstance(trainer.val_check_interval, int)
                    and trainer.val_check_interval % lr_scheduler_config.frequency != 0
                ):
                    errors.append(
                        f"Trainer is not running validation every {lr_scheduler_config.frequency} steps but "
                        f"a ReduceLRPlateau scheduler with interval={lr_scheduler_config.interval} and frequency={lr_scheduler_config.frequency} is used."
                        "Please set `config.trainer.val_check_interval` "
                        f"to a multiple of {lr_scheduler_config.frequency}. "
                        + disable_message
                    )

            case _:
                pass

    if not errors:
        return

    message = (
        "ReduceLRPlateau sanity checks failed with the following errors:\n"
        + "\n".join(errors)
    )
    match config.trainer.sanity_checking.reduce_lr_on_plateau:
        case "warn":
            log.warning(message)
        case "error":
            raise ValueError(message)
        case _:
            pass


@runtime_checkable
class CustomRLPImplementation(Protocol):
    __reduce_lr_on_plateau__: bool


class RLPSanityCheckModuleMixin(mixin_base_type(CallbackModuleMixin)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global _on_train_start_callback
        self.register_callback(on_train_start=_on_train_start_callback)

    def reduce_lr_on_plateau_config(
        self,
        lr_scheduler: LRSchedulerTypeUnion | LRSchedulerConfigType,
    ) -> LRSchedulerConfigType:
        if (trainer := self._trainer) is None:
            raise RuntimeError(
                "Could not determine the frequency of ReduceLRPlateau scheduler "
                "because `self.trainer` is None."
            )

        # First, resolve the LR scheduler from the provided config.
        lr_scheduler_config: LRSchedulerConfigType
        match lr_scheduler:
            case Mapping():
                lr_scheduler_config = cast(LRSchedulerConfigType, lr_scheduler)
            case _:
                lr_scheduler_config = {"scheduler": lr_scheduler}

        # Make sure the scheduler is a ReduceLRPlateau scheduler. Otherwise, warn the user.
        if (
            not isinstance(
                lr_scheduler_config["scheduler"],
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            )
        ) and (
            not isinstance(lr_scheduler_config["scheduler"], CustomRLPImplementation)
            or not lr_scheduler_config["scheduler"].__reduce_lr_on_plateau__
        ):
            log.warning(
                "`reduce_lr_on_plateau_config` should only be used with a ReduceLRPlateau scheduler. "
                f"The provided scheduler, {lr_scheduler_config['scheduler']}, does not subclass "
                "`torch.optim.lr_scheduler.ReduceLROnPlateau`. "
                "Please ensure that the scheduler is a ReduceLRPlateau scheduler. "
                "If you are using a custom ReduceLRPlateau scheduler implementation, "
                "please either (1) make sure that it subclasses `torch.optim.lr_scheduler.ReduceLROnPlateau`, "
                "or (2) set the scheduler's `__reduce_lr_on_plateau__` attribute to `True`."
            )

        # If trainer.check_val_every_n_epoch is an integer, then we run val at epoch intervals.
        if trainer.check_val_every_n_epoch is not None:
            return {
                "reduce_on_plateau": True,
                "interval": "epoch",
                "frequency": trainer.check_val_every_n_epoch,
                **lr_scheduler_config,
            }

        # Otherwise, we run val at step intervals.
        if not isinstance(trainer.val_check_batch, int):
            raise ValueError(
                "Could not determine the frequency of ReduceLRPlateau scheduler "
                f"because {trainer.val_check_batch=} is not an integer."
            )

        return {
            "reduce_on_plateau": True,
            "interval": "step",
            "frequency": trainer.val_check_batch,
            **lr_scheduler_config,
        }
