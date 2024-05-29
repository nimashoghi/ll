import time
from abc import ABC, abstractmethod
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal

import torch
from lightning.pytorch.callbacks.throughput_monitor import (
    ThroughputMonitor as _ThroughputMonitor,
)
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from typing_extensions import override

from ..config import TypedConfig

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.callbacks.callback import Callback


log = getLogger(__name__)


class CallbackBaseConfig(TypedConfig, ABC):
    @abstractmethod
    def construct_callback(self) -> "Callback": ...


class ThroughputMonitorCallback(_ThroughputMonitor):
    @override
    def _update(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        batch: Any,
        iter_num: int,
    ) -> None:
        stage = trainer.state.stage
        assert stage is not None
        throughput = self._throughputs[stage]

        if trainer.strategy.root_device.type == "cuda":
            # required or else perf_counter() won't be correct
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._t0s[stage]
        if self.length_fn is not None:
            with torch.inference_mode():
                length = self.length_fn(batch)
            self._lengths[stage] += length

        if hasattr(pl_module, "flops_per_batch"):
            flops_per_batch = pl_module.flops_per_batch
        else:
            rank_zero_warn(
                "When using the `ThroughputMonitor`, you need to define a `flops_per_batch` attribute or property"
                f" in {type(pl_module).__name__} to compute the FLOPs."
            )
            flops_per_batch = None

        with torch.inference_mode():
            batch_size = self.batch_size_fn(batch)
        throughput.update(
            time=elapsed,
            batches=iter_num,
            # this assumes that all iterations used the same batch size
            samples=iter_num * batch_size,
            lengths=None if self.length_fn is None else self._lengths[stage],
            flops=flops_per_batch,
        )


class ThroughputMonitorConfig(CallbackBaseConfig):
    name: Literal["throughput_monitor"] = "throughput_monitor"

    batch_size: int
    """Batch size to use for computing throughput."""

    length: int | None = None
    """Length to use for computing throughput."""

    window_size: int = 100
    """Number of batches to use for a rolling average."""

    separator: str = "/"
    """Key separator to use when creating per-device and global metrics."""

    @override
    def construct_callback(self):
        return ThroughputMonitorCallback(
            batch_size_fn=lambda _: self.batch_size,
            length_fn=(lambda _: l) if (l := self.length) is not None else None,
            window_size=self.window_size,
            separator=self.separator,
        )