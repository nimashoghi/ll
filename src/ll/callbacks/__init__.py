from typing import Annotated

from ..config import Field
from .base import CallbackConfigBase as CallbackConfigBase
from .early_stopping import EarlyStopping as EarlyStopping
from .ema import EMA as EMA
from .ema import EMAConfig as EMAConfig
from .finite_checks import FiniteChecksCallback as FiniteChecksCallback
from .finite_checks import FiniteChecksConfig as FiniteChecksConfig
from .gradient_skipping import GradientSkipping as GradientSkipping
from .gradient_skipping import GradientSkippingConfig as GradientSkippingConfig
from .interval import EpochIntervalCallback as EpochIntervalCallback
from .interval import IntervalCallback as IntervalCallback
from .interval import StepIntervalCallback as StepIntervalCallback
from .latest_epoch_checkpoint import LatestEpochCheckpoint as LatestEpochCheckpoint
from .log_epoch import LogEpochCallback as LogEpochCallback
from .norm_logging import NormLoggingCallback as NormLoggingCallback
from .norm_logging import NormLoggingConfig as NormLoggingConfig
from .on_exception_checkpoint import OnExceptionCheckpoint as OnExceptionCheckpoint
from .print_table import PrintTableMetricsCallback as PrintTableMetricsCallback
from .print_table import PrintTableMetricsConfig as PrintTableMetricsConfig
from .throughput_monitor import ThroughputMonitorConfig as ThroughputMonitorConfig
from .timer import EpochTimer as EpochTimer
from .timer import EpochTimerConfig as EpochTimerConfig

CallbackConfig = Annotated[
    ThroughputMonitorConfig
    | EpochTimerConfig
    | PrintTableMetricsConfig
    | FiniteChecksConfig
    | NormLoggingConfig
    | GradientSkippingConfig
    | EMAConfig,
    Field(discriminator="name"),
]
