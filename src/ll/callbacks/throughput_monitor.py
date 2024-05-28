import time
from abc import ABC, abstractmethod
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

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


@runtime_checkable
class ThroughputMonitorableWithBatchSizeFnModule(Protocol):
    def throughput_monitor_compute_batch_size(self, batch: Any) -> int: ...


@runtime_checkable
class ThroughputMonitorableModuleWithLengthFn(Protocol):
    def throughput_monitor_compute_length(self, batch: Any) -> int: ...


class ThroughputMonitorCallback(_ThroughputMonitor):
    @override
    @override
    def setup(
        self, trainer: "Trainer", pl_module: "LightningModule", stage: str
    ) -> None:
        super().setup(trainer, pl_module, stage)

        # If batch_size_fn is not provided, we can use the default one.
        if self.batch_size_fn is None:
            if not isinstance(pl_module, ThroughputMonitorableWithBatchSizeFnModule):
                raise ValueError(
                    f"Module {type(pl_module)} does not implement throughput_monitor_compute_batch_size. "
                    "See the ThroughputMonitorableWithBatchSizeFnModule protocol."
                )

            self.batch_size_fn = pl_module.throughput_monitor_compute_batch_size

        # If length_fn is not provided, we can use the default one.
        if self.length_fn is None:
            if not isinstance(pl_module, ThroughputMonitorableModuleWithLengthFn):
                rank_zero_warn(
                    f"Module {type(pl_module)} does not implement throughput_monitor_compute_length. "
                    "See the ThroughputMonitorableModuleWithLengthFn protocol. "
                    "Length function will not be used."
                )
            else:
                self.length_fn = pl_module.throughput_monitor_compute_length

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

    window_size: int = 100
    """Number of batches to use for a rolling average."""

    separator: str = "/"
    """Key separator to use when creating per-device and global metrics."""

    @override
    def construct_callback(self):
        return ThroughputMonitorCallback(
            batch_size_fn=cast(Any, None),  # None is okay here
            length_fn=None,
            window_size=self.window_size,
            separator=self.separator,
        )

    @classmethod
    def from_pyg_batch(cls):
        try:
            from torch_geometric.data import Batch
        except ImportError:
            raise ImportError(
                f"{cls.__name__}.from_pyg_batch requires torch-geometric to be installed."
            )

        def batch_size_fn(batch: Batch):
            return batch.num_graphs

        def length_fn(batch: Batch):
            if (num_nodes := getattr(batch, "num_nodes", None)) is not None:
                assert isinstance(
                    num_nodes, int
                ), f"num_nodes must be an int, got {type(num_nodes)=}"
                return num_nodes

            num_nodes = 0
            for graph in batch.to_data_list():
                assert (
                    graph.num_nodes is not None
                ), "num_nodes must be set for all graphs"
                num_nodes += graph.num_nodes
            return num_nodes

        return cls(batch_size_fn=batch_size_fn, length_fn=length_fn)
