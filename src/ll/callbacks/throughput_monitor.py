from logging import getLogger
from typing import TYPE_CHECKING, Literal

from lightning.pytorch.callbacks.callback import Callback
from typing_extensions import override

from .base import CallbackConfigBase

log = getLogger(__name__)


try:
    from ._throughput_monitor_callback import ThroughputMonitor


except ImportError:
    if TYPE_CHECKING:
        from ._throughput_monitor_callback import ThroughputMonitor
    else:

        class ThroughputMonitor(Callback):
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "To use the `ThroughputMonitor`, please install the `lightning` package."
                )


class ThroughputMonitorConfig(CallbackConfigBase):
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
    def construct_callbacks(self, root_config):
        yield ThroughputMonitor(
            batch_size_fn=lambda _: self.batch_size,
            length_fn=(lambda _: l) if (l := self.length) is not None else None,
            window_size=self.window_size,
            separator=self.separator,
        )
