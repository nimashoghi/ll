from typing_extensions import TypeAlias

from ..actsave._config import ActSaveAsyncSaverConfig as ActSaveAsyncSaverConfig
from ..actsave._config import ActSaveConfig as ActSaveConfig
from ..actsave._config import ActSaveSyncSaverConfig as ActSaveSyncSaverConfig
from ..actsave._config import ActSaveTransformConfig as ActSaveTransformConfig
from .base import Base as Base
from .base import LightningDataModuleBase as LightningDataModuleBase
from .base import LightningModuleBase as LightningModuleBase
from .config import BaseConfig as BaseConfig
from .config import BaseLoggerConfig as BaseLoggerConfig
from .config import BaseProfilerConfig as BaseProfilerConfig
from .config import CheckpointCallbackBaseConfig as CheckpointCallbackBaseConfig
from .config import CheckpointLoadingConfig as CheckpointLoadingConfig
from .config import CheckpointSavingConfig as CheckpointSavingConfig
from .config import DirectoryConfig as DirectoryConfig
from .config import EarlyStoppingConfig as EarlyStoppingConfig
from .config import (
    EnvironmentClassInformationConfig as EnvironmentClassInformationConfig,
)
from .config import EnvironmentConfig as EnvironmentConfig
from .config import (
    EnvironmentLinuxEnvironmentConfig as EnvironmentLinuxEnvironmentConfig,
)
from .config import (
    EnvironmentSLURMInformationConfig as EnvironmentSLURMInformationConfig,
)
from .config import GradientClippingConfig as GradientClippingConfig
from .config import GradientSkippingConfig as GradientSkippingConfig
from .config import (
    LatestEpochCheckpointCallbackConfig as LatestEpochCheckpointCallbackConfig,
)
from .config import LoggingConfig as LoggingConfig
from .config import ModelCheckpointCallbackConfig as ModelCheckpointCallbackConfig
from .config import (
    OnExceptionCheckpointCallbackConfig as OnExceptionCheckpointCallbackConfig,
)
from .config import OptimizationConfig as OptimizationConfig
from .config import PrimaryMetricConfig as PrimaryMetricConfig
from .config import PythonLogging as PythonLogging
from .config import ReproducibilityConfig as ReproducibilityConfig
from .config import RunnerConfig as RunnerConfig
from .config import RunnerOutputSaveConfig as RunnerOutputSaveConfig
from .config import SeedEverythingConfig as SeedEverythingConfig
from .config import TrainerConfig as TrainerConfig
from .config import WandbWatchConfig as WandbWatchConfig

ConfigList: TypeAlias = list[tuple[BaseConfig, type[LightningModuleBase]]]
