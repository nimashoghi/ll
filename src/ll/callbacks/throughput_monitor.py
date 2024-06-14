from logging import getLogger
from typing import Any, Literal, Protocol, TypedDict, cast, runtime_checkable

from typing_extensions import NotRequired, override

from ._throughput_monitor_callback import ThroughputMonitor as _ThroughputMonitor
from .base import CallbackConfigBase

log = getLogger(__name__)


class ThroughputMonitorBatchStats(TypedDict):
    batch_size: int
    length: NotRequired[int | None]


@runtime_checkable
class SupportsThroughputMonitorModuleProtocol(Protocol):
    def throughput_monitor_batch_stats(
        self, batch: Any
    ) -> ThroughputMonitorBatchStats: ...


class ThroughputMonitor(_ThroughputMonitor):
    def __init__(self, window_size: int = 100) -> None:
        super().__init__(cast(Any, None), cast(Any, None), window_size=window_size)

    @override
    def setup(self, trainer, pl_module, stage):
        if not isinstance(pl_module, SupportsThroughputMonitorModuleProtocol):
            raise RuntimeError(
                "The model does not implement `throughput_monitor_batch_stats`. "
                "Please either implement this method, or do not use the `ThroughputMonitor` callback."
            )

        def batch_size_fn(batch):
            return pl_module.throughput_monitor_batch_stats(batch)["batch_size"]

        def length_fn(batch):
            return pl_module.throughput_monitor_batch_stats(batch).get("length")

        self.batch_size_fn = batch_size_fn
        self.length_fn = length_fn

        return super().setup(trainer, pl_module, stage)


class ThroughputMonitorConfig(CallbackConfigBase):
    name: Literal["throughput_monitor"] = "throughput_monitor"

    window_size: int = 100
    """Number of batches to use for a rolling average."""

    @override
    def construct_callbacks(self, root_config):
        yield ThroughputMonitor(window_size=self.window_size)
