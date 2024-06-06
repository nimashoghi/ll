from typing import Annotated

from ..config import Field
from .bad_gradients import PrintBadGradientsCallback as PrintBadGradientsCallback
from .early_stopping import EarlyStopping as EarlyStopping
from .ema import EMA as EMA
from .interval import EpochIntervalCallback as EpochIntervalCallback
from .interval import IntervalCallback as IntervalCallback
from .interval import StepIntervalCallback as StepIntervalCallback
from .latest_epoch_checkpoint import LatestEpochCheckpoint as LatestEpochCheckpoint
from .on_exception_checkpoint import OnExceptionCheckpoint as OnExceptionCheckpoint
from .throughput_monitor import ThroughputMonitorConfig as ThroughputMonitorConfig
from .timer import TimerConfig as TimerConfig

CallbackConfig = Annotated[
    ThroughputMonitorConfig | TimerConfig,
    Field(discriminator="name"),
]
