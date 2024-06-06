from typing import Annotated

from ..config import Field
from .bad_gradients import PrintBadGradientsCallback as PrintBadGradientsCallback
from .early_stopping import EarlyStopping as EarlyStopping
from .ema import EMA as EMA
from .interval import EpochIntervalCallback as EpochIntervalCallback
from .interval import IntervalCallback as IntervalCallback
from .interval import StepIntervalCallback as StepIntervalCallback
from .latest_epoch_checkpoint import LatestEpochCheckpoint as LatestEpochCheckpoint
from .log_epoch import LogEpochCallback as LogEpochCallback
from .on_exception_checkpoint import OnExceptionCheckpoint as OnExceptionCheckpoint
from .print_table import PrintTableMetricsCallback as PrintTableMetricsCallback
from .throughput_monitor import ThroughputMonitorConfig as ThroughputMonitorConfig
from .timer import EpochTimer as EpochTimer
from .timer import EpochTimerConfig as EpochTimerConfig

CallbackConfig = Annotated[
    ThroughputMonitorConfig | EpochTimerConfig,
    Field(discriminator="name"),
]
