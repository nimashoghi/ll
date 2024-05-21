from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import getLogger
from typing import Annotated, Any, Literal

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks.throughput_monitor import ThroughputMonitor
from typing_extensions import override

from ...config import Field, TypedConfig

log = getLogger(__name__)


class CallbackBaseConfig(TypedConfig, ABC):
    @abstractmethod
    def construct_callback(self) -> Callback: ...


class ThroughputMonitorCallbackConfig(CallbackBaseConfig):
    name: Literal["throughput_monitor"] = "throughput_monitor"

    batch_size_fn: Callable[[Any], int]
    """Function that takes in a batch and returns the batch size."""

    length_fn: Callable[[Any], int] | None = None
    """Function that takes in a batch and returns the number of items in each sample of the batch."""

    available_flops: float | None = None
    """Number of theoretical flops available for a single device."""

    world_size: int = 1
    """Number of devices available across hosts. Global metrics are not included if the world size is 1."""

    window_size: int = 100
    """Number of batches to use for a rolling average."""

    separator: str = "/"
    """Key separator to use when creating per-device and global metrics."""

    @override
    def construct_callback(self):
        return ThroughputMonitor(
            batch_size_fn=self.batch_size_fn,
            length_fn=self.length_fn,
            available_flops=self.available_flops,
            world_size=self.world_size,
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


CallbackConfig = Annotated[
    ThroughputMonitorCallbackConfig,
    Field(discriminator="name"),
]
