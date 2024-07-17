import copy
import os
import re
import signal
import socket
import string
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import numpy as np
import torch
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins import _PLUGIN_INPUT
from lightning.pytorch.plugins.layer_sync import LayerSync
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies.strategy import Strategy
from pydantic import DirectoryPath
from typing_extensions import Self, TypedDict, TypeVar, override

from ..callbacks import CallbackConfig
from ..callbacks.base import CallbackConfigBase
from ..callbacks.wandb_watch import WandbWatchConfig
from ..config import Field, TypedConfig
from ..util.slurm import parse_slurm_node_list

log = getLogger(__name__)


class IdSeedWarning(Warning):
    pass


class BaseProfilerConfig(TypedConfig, ABC):
    dirpath: str | Path | None = None
    """
    Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
        ``trainer.log_dir`` (from :class:`~lightning.pytorch.loggers.tensorboard.TensorBoardLogger`)
        will be used.
    """
    filename: str | None = None
    """
    If present, filename where the profiler results will be saved instead of printing to stdout.
        The ``.txt`` extension will be used automatically.
    """

    @abstractmethod
    def construct_profiler(self, root_config: "BaseConfig") -> Profiler: ...


class SimpleProfilerConfig(BaseProfilerConfig):
    kind: Literal["simple"] = "simple"

    extended: bool = True
    """
    If ``True``, adds extra columns representing number of calls and percentage of
        total time spent onrespective action.
    """

    @override
    def construct_profiler(self, root_config):
        from lightning.pytorch.profilers.simple import SimpleProfiler

        if (dirpath := self.dirpath) is None:
            dirpath = root_config.directory.resolve_subdirectory(
                root_config.id, "profile"
            )

        if (filename := self.filename) is None:
            filename = f"{root_config.id}_profile.txt"

        return SimpleProfiler(
            extended=self.extended,
            dirpath=dirpath,
            filename=filename,
        )


class AdvancedProfilerConfig(BaseProfilerConfig):
    kind: Literal["advanced"] = "advanced"

    line_count_restriction: float = 1.0
    """
    This can be used to limit the number of functions
        reported for each action. either an integer (to select a count of lines),
        or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)
    """

    @override
    def construct_profiler(self, root_config):
        from lightning.pytorch.profilers.advanced import AdvancedProfiler

        if (dirpath := self.dirpath) is None:
            dirpath = root_config.directory.resolve_subdirectory(
                root_config.id, "profile"
            )

        if (filename := self.filename) is None:
            filename = f"{root_config.id}_profile.txt"

        return AdvancedProfiler(
            line_count_restriction=self.line_count_restriction,
            dirpath=dirpath,
            filename=filename,
        )


class PyTorchProfilerConfig(BaseProfilerConfig):
    kind: Literal["pytorch"] = "pytorch"

    group_by_input_shapes: bool = False
    """Include operator input shapes and group calls by shape."""

    emit_nvtx: bool = False
    """
    Context manager that makes every autograd operation emit an NVTX range
        Run::

            nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

        To visualize, you can either use::

            nvvp trace_name.prof
            torch.autograd.profiler.load_nvprof(path)
    """

    export_to_chrome: bool = True
    """
    Whether to export the sequence of profiled operators for Chrome.
        It will generate a ``.json`` file which can be read by Chrome.
    """

    row_limit: int = 20
    """
    Limit the number of rows in a table, ``-1`` is a special value that
        removes the limit completely.
    """

    sort_by_key: str | None = None
    """
    Attribute used to sort entries. By default
        they are printed in the same order as they were registered.
        Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
        ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
        ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.
    """

    record_module_names: bool = True
    """Whether to add module names while recording autograd operation."""

    table_kwargs: dict[str, Any] | None = None
    """Dictionary with keyword arguments for the summary table."""

    additional_profiler_kwargs: dict[str, Any] = {}
    """Keyword arguments for the PyTorch profiler. This depends on your PyTorch version"""

    @override
    def construct_profiler(self, root_config):
        from lightning.pytorch.profilers.pytorch import PyTorchProfiler

        if (dirpath := self.dirpath) is None:
            dirpath = root_config.directory.resolve_subdirectory(
                root_config.id, "profile"
            )

        if (filename := self.filename) is None:
            filename = f"{root_config.id}_profile.txt"

        return PyTorchProfiler(
            group_by_input_shapes=self.group_by_input_shapes,
            emit_nvtx=self.emit_nvtx,
            export_to_chrome=self.export_to_chrome,
            row_limit=self.row_limit,
            sort_by_key=self.sort_by_key,
            record_module_names=self.record_module_names,
            table_kwargs=self.table_kwargs,
            dirpath=dirpath,
            filename=filename,
            **self.additional_profiler_kwargs,
        )


ProfilerConfig: TypeAlias = Annotated[
    SimpleProfilerConfig | AdvancedProfilerConfig | PyTorchProfilerConfig,
    Field(discriminator="kind"),
]


class EnvironmentClassInformationConfig(TypedConfig):
    name: str
    module: str
    full_name: str

    file_path: Path
    source_file_path: Path | None = None


class EnvironmentSLURMInformationConfig(TypedConfig):
    hostname: str
    hostnames: list[str]
    job_id: str
    raw_job_id: str
    array_job_id: str | None
    array_task_id: str | None
    num_tasks: int
    num_nodes: int
    node: str | int | None
    global_rank: int
    local_rank: int

    @classmethod
    def from_current_environment(cls):
        try:
            from lightning.fabric.plugins.environments.slurm import SLURMEnvironment

            if not SLURMEnvironment.detect():
                return None

            hostname = socket.gethostname()
            hostnames = [hostname]
            if node_list := os.environ.get("SLURM_JOB_NODELIST", ""):
                hostnames = parse_slurm_node_list(node_list)

            raw_job_id = os.environ["SLURM_JOB_ID"]
            job_id = raw_job_id
            array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
            array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if array_job_id and array_task_id:
                job_id = f"{array_job_id}_{array_task_id}"

            num_tasks = int(os.environ["SLURM_NTASKS"])
            num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])

            node_id = os.environ.get("SLURM_NODEID")

            global_rank = int(os.environ["SLURM_PROCID"])
            local_rank = int(os.environ["SLURM_LOCALID"])

            return cls(
                hostname=hostname,
                hostnames=hostnames,
                job_id=job_id,
                raw_job_id=raw_job_id,
                array_job_id=array_job_id,
                array_task_id=array_task_id,
                num_tasks=num_tasks,
                num_nodes=num_nodes,
                node=node_id,
                global_rank=global_rank,
                local_rank=local_rank,
            )
        except (ImportError, RuntimeError, ValueError, KeyError):
            return None


class EnvironmentLSFInformationConfig(TypedConfig):
    hostname: str
    hostnames: list[str]
    job_id: str
    array_job_id: str | None
    array_task_id: str | None
    num_tasks: int
    num_nodes: int
    node: str | int | None
    global_rank: int
    local_rank: int

    @classmethod
    def from_current_environment(cls):
        try:
            import os
            import socket

            hostname = socket.gethostname()
            hostnames = [hostname]
            if node_list := os.environ.get("LSB_HOSTS", ""):
                hostnames = node_list.split()

            job_id = os.environ["LSB_JOBID"]
            array_job_id = os.environ.get("LSB_JOBINDEX")
            array_task_id = os.environ.get("LSB_JOBINDEX")

            num_tasks = int(os.environ.get("LSB_DJOB_NUMPROC", 1))
            num_nodes = len(set(hostnames))

            node_id = (
                os.environ.get("LSB_HOSTS", "").split().index(hostname)
                if "LSB_HOSTS" in os.environ
                else None
            )

            # LSF doesn't have direct equivalents for global_rank and local_rank
            # You might need to calculate these based on your specific setup
            global_rank = int(os.environ.get("PMI_RANK", 0))
            local_rank = int(os.environ.get("LSB_RANK", 0))

            return cls(
                hostname=hostname,
                hostnames=hostnames,
                job_id=job_id,
                array_job_id=array_job_id,
                array_task_id=array_task_id,
                num_tasks=num_tasks,
                num_nodes=num_nodes,
                node=node_id,
                global_rank=global_rank,
                local_rank=local_rank,
            )
        except (ImportError, RuntimeError, ValueError, KeyError):
            return None


class EnvironmentLinuxEnvironmentConfig(TypedConfig):
    """
    Information about the Linux environment (e.g., current user, hostname, etc.)
    """

    user: str | None = None
    hostname: str | None = None
    system: str | None = None
    release: str | None = None
    version: str | None = None
    machine: str | None = None
    processor: str | None = None
    cpu_count: int | None = None
    memory: int | None = None
    uptime: timedelta | None = None
    boot_time: float | None = None
    load_avg: tuple[float, float, float] | None = None


class EnvironmentConfig(TypedConfig):
    cwd: Path | None = None

    python_executable: Path | None = None
    python_path: list[Path] | None = None
    python_version: str | None = None

    config: EnvironmentClassInformationConfig | None = None
    model: EnvironmentClassInformationConfig | None = None
    data: EnvironmentClassInformationConfig | None = None

    linux: EnvironmentLinuxEnvironmentConfig | None = None

    slurm: EnvironmentSLURMInformationConfig | None = None
    lsf: EnvironmentLSFInformationConfig | None = None

    base_dir: Path | None = None
    log_dir: Path | None = None
    checkpoint_dir: Path | None = None
    stdio_dir: Path | None = None

    seed: int | None = None
    seed_workers: bool | None = None


class BaseLoggerConfig(TypedConfig, ABC):
    enabled: bool = True
    """Enable this logger."""

    priority: int = 0
    """Priority of the logger. Higher values are logged first."""

    log_dir: DirectoryPath | None = None
    """Directory to save the logs to. If None, will use the default log directory for the trainer."""

    @abstractmethod
    def construct_logger(self, root_config: "BaseConfig") -> Logger | None: ...

    def disable_(self):
        self.enabled = False
        return self


def _project_name(
    root_config: "BaseConfig",
    default_project: str = "lightning_logs",
):
    # If the config has a project name, use that.
    if project := root_config.project:
        return project

    # Otherwise, we should use the name of the module that the config is defined in,
    #   if we can find it.
    # If this isn't in a module, use the default project name.
    if not (module := root_config.__module__):
        return default_project

    # If the module is a package, use the package name.
    if not (module := module.split(".", maxsplit=1)[0].strip()):
        return default_project

    return module


def _wandb_available():
    try:
        from lightning.pytorch.loggers.wandb import _WANDB_AVAILABLE

        if not _WANDB_AVAILABLE:
            log.warning("WandB not found. Disabling WandbLogger.")
            return False
        return True
    except ImportError:
        return False


class WandbLoggerConfig(CallbackConfigBase, BaseLoggerConfig):
    kind: Literal["wandb"] = "wandb"

    enabled: bool = Field(default_factory=lambda: _wandb_available())
    """Enable WandB logging."""

    priority: int = 2
    """Priority of the logger. Higher values are logged first."""

    project: str | None = None
    """WandB project name to use for the logger. If None, will use the root config's project name."""

    log_model: bool | Literal["all"] = False
    """
    Whether to log the model checkpoints to wandb.
    Valid values are:
        - False: Do not log the model checkpoints.
        - True: Log the latest model checkpoint.
        - "all": Log all model checkpoints.
    """

    watch: WandbWatchConfig = WandbWatchConfig()
    """WandB model watch configuration. Used to log model architecture, gradients, and parameters."""

    offline: bool = False
    """Whether to run WandB in offline mode."""

    @override
    def construct_logger(self, root_config):
        if not self.enabled:
            return None

        from lightning.pytorch.loggers.wandb import WandbLogger

        save_dir = root_config.directory.resolve_log_directory_for_logger(
            root_config.id,
            self,
        )
        save_dir = save_dir / "wandb"
        save_dir.mkdir(parents=True, exist_ok=True)
        return WandbLogger(
            save_dir=save_dir,
            project=self.project or _project_name(root_config),
            name=root_config.run_name,
            version=root_config.id,
            log_model=self.log_model,
            notes=(
                "\n".join(f"- {note}" for note in root_config.notes)
                if root_config.notes
                else None
            ),
            tags=root_config.tags,
            offline=self.offline,
        )

    @override
    def construct_callbacks(self, root_config):
        if self.watch:
            yield from self.watch.construct_callbacks(root_config)


class CSVLoggerConfig(BaseLoggerConfig):
    kind: Literal["csv"] = "csv"

    enabled: bool = True
    """Enable CSV logging."""

    priority: int = 0
    """Priority of the logger. Higher values are logged first."""

    prefix: str = ""
    """A string to put at the beginning of metric keys."""

    flush_logs_every_n_steps: int = 100
    """How often to flush logs to disk."""

    @override
    def construct_logger(self, root_config):
        if not self.enabled:
            return None

        from lightning.pytorch.loggers.csv_logs import CSVLogger

        save_dir = root_config.directory.resolve_log_directory_for_logger(
            root_config.id,
            self,
        )
        save_dir = save_dir / "csv"
        save_dir.mkdir(parents=True, exist_ok=True)
        return CSVLogger(
            save_dir=save_dir,
            name=root_config.run_name,
            version=root_config.id,
            prefix=self.prefix,
            flush_logs_every_n_steps=self.flush_logs_every_n_steps,
        )


def _tensorboard_available():
    try:
        from lightning.fabric.loggers.tensorboard import (
            _TENSORBOARD_AVAILABLE,
            _TENSORBOARDX_AVAILABLE,
        )

        if not _TENSORBOARD_AVAILABLE and not _TENSORBOARDX_AVAILABLE:
            log.warning(
                "TensorBoard/TensorBoardX not found. Disabling TensorBoardLogger. "
                "Please install TensorBoard with `pip install tensorboard` or "
                "TensorBoardX with `pip install tensorboardx` to enable TensorBoard logging."
            )
            return False
        return True
    except ImportError:
        return False


class TensorboardLoggerConfig(BaseLoggerConfig):
    kind: Literal["tensorboard"] = "tensorboard"

    enabled: bool = Field(default_factory=lambda: _tensorboard_available())
    """Enable TensorBoard logging."""

    priority: int = 2
    """Priority of the logger. Higher values are logged first."""

    log_graph: bool = False
    """
    Adds the computational graph to tensorboard. This requires that
        the user has defined the `self.example_input_array` attribute in their
        model.
    """

    default_hp_metric: bool = True
    """
    Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
        called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
    """

    prefix: str = ""
    """A string to put at the beginning of metric keys."""

    @override
    def construct_logger(self, root_config):
        if not self.enabled:
            return None

        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

        save_dir = root_config.directory.resolve_log_directory_for_logger(
            root_config.id,
            self,
        )
        save_dir = save_dir / "tensorboard"
        save_dir.mkdir(parents=True, exist_ok=True)
        return TensorBoardLogger(
            save_dir=save_dir,
            name=root_config.run_name,
            version=root_config.id,
            log_graph=self.log_graph,
            default_hp_metric=self.default_hp_metric,
        )


LoggerConfig: TypeAlias = Annotated[
    WandbLoggerConfig | CSVLoggerConfig | TensorboardLoggerConfig,
    Field(discriminator="kind"),
]


class LoggingConfig(CallbackConfigBase):
    enabled: bool = True
    """Enable experiment tracking."""

    loggers: Sequence[LoggerConfig] = [
        WandbLoggerConfig(),
        CSVLoggerConfig(),
        TensorboardLoggerConfig(),
    ]
    """Loggers to use for experiment tracking."""

    log_lr: bool | Literal["step", "epoch"] = True
    """If enabled, will register a `LearningRateMonitor` callback to log the learning rate to the logger."""
    log_epoch: bool = True
    """If enabled, will log the fractional epoch number to the logger."""

    @property
    def wandb(self) -> WandbLoggerConfig | None:
        return next(
            (
                logger
                for logger in self.loggers
                if isinstance(logger, WandbLoggerConfig)
            ),
        )

    @property
    def csv(self) -> CSVLoggerConfig | None:
        return next(
            (logger for logger in self.loggers if isinstance(logger, CSVLoggerConfig)),
        )

    @property
    def tensorboard(self) -> TensorboardLoggerConfig | None:
        return next(
            (
                logger
                for logger in self.loggers
                if isinstance(logger, TensorboardLoggerConfig)
            ),
        )

    def construct_loggers(self, root_config: "BaseConfig"):
        """
        Constructs and returns a list of loggers based on the provided root configuration.

        Args:
            root_config (BaseConfig): The root configuration object.

        Returns:
            list[Logger]: A list of constructed loggers.
        """
        loggers: list[Logger] = []
        if not self.enabled:
            return loggers

        for logger_config in sorted(
            self.loggers,
            key=lambda x: x.priority,
            reverse=True,
        ):
            if not logger_config.enabled:
                continue
            if (logger := logger_config.construct_logger(root_config)) is None:
                continue
            loggers.append(logger)
        return loggers

    @override
    def construct_callbacks(self, root_config):
        if self.log_lr:
            from lightning.pytorch.callbacks import LearningRateMonitor

            logging_interval: str | None = None
            if isinstance(self.log_lr, str):
                logging_interval = self.log_lr

            yield LearningRateMonitor(logging_interval=logging_interval)

        if self.log_epoch:
            from ..callbacks.log_epoch import LogEpochCallback

            yield LogEpochCallback()

        for logger in self.loggers:
            if not logger or not isinstance(logger, CallbackConfigBase):
                continue

            yield from logger.construct_callbacks(root_config)


class GradientClippingConfig(TypedConfig):
    enabled: bool = True
    """Enable gradient clipping."""
    value: int | float
    """Value to use for gradient clipping."""
    algorithm: Literal["value", "norm"] = "norm"
    """Norm type to use for gradient clipping."""


class OptimizationConfig(CallbackConfigBase):
    log_grad_norm: bool | str | float = False
    """If enabled, will log the gradient norm (averaged across all model parameters) to the logger."""
    log_grad_norm_per_param: bool | str | float = False
    """If enabled, will log the gradient norm for each model parameter to the logger."""

    log_param_norm: bool | str | float = False
    """If enabled, will log the parameter norm (averaged across all model parameters) to the logger."""
    log_param_norm_per_param: bool | str | float = False
    """If enabled, will log the parameter norm for each model parameter to the logger."""

    gradient_clipping: GradientClippingConfig | None = None
    """Gradient clipping configuration, or None to disable gradient clipping."""

    @override
    def construct_callbacks(self, root_config):
        from ..callbacks.norm_logging import NormLoggingConfig

        yield from NormLoggingConfig(
            log_grad_norm=self.log_grad_norm,
            log_grad_norm_per_param=self.log_grad_norm_per_param,
            log_param_norm=self.log_param_norm,
            log_param_norm_per_param=self.log_param_norm_per_param,
        ).construct_callbacks(root_config)


LogLevel: TypeAlias = Literal[
    "CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"
]


class PythonLogging(TypedConfig):
    log_level: LogLevel | None = None
    """Log level to use for the Python logger (or None to use the default)."""

    rich: bool = False
    """If enabled, will use the rich library to format the Python logger output."""
    rich_tracebacks: bool = True
    """If enabled, will use the rich library to format the Python logger tracebacks."""

    lovely_tensors: bool = False
    """If enabled, will use the lovely-tensors library to format PyTorch tensors. False by default as it causes issues when used with `torch.vmap`."""
    lovely_numpy: bool = False
    """If enabled, will use the lovely-numpy library to format numpy arrays. False by default as it causes some issues with other libaries."""

    def pretty_(
        self,
        *,
        log_level: LogLevel | None = "INFO",
        torch: bool = True,
        numpy: bool = True,
        rich: bool = True,
        rich_tracebacks: bool = True,
    ):
        self.log_level = log_level
        self.lovely_tensors = torch
        self.lovely_numpy = numpy
        self.rich = rich
        self.rich_tracebacks = rich_tracebacks


TPlugin = TypeVar(
    "TPlugin",
    Precision,
    ClusterEnvironment,
    CheckpointIO,
    LayerSync,
    infer_variance=True,
)


@runtime_checkable
class PluginConfigProtocol(Protocol[TPlugin]):
    def construct_plugin(self) -> TPlugin: ...


@runtime_checkable
class AcceleratorConfigProtocol(Protocol):
    def construct_accelerator(self) -> Accelerator: ...


@runtime_checkable
class StrategyConfigProtocol(Protocol):
    def construct_strategy(self) -> Strategy: ...


AcceleratorLiteral: TypeAlias = Literal[
    "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"
]

StrategyLiteral: TypeAlias = Literal[
    "auto",
    "ddp",
    "ddp_find_unused_parameters_false",
    "ddp_find_unused_parameters_true",
    "ddp_spawn",
    "ddp_spawn_find_unused_parameters_false",
    "ddp_spawn_find_unused_parameters_true",
    "ddp_fork",
    "ddp_fork_find_unused_parameters_false",
    "ddp_fork_find_unused_parameters_true",
    "ddp_notebook",
    "dp",
    "deepspeed",
    "deepspeed_stage_1",
    "deepspeed_stage_1_offload",
    "deepspeed_stage_2",
    "deepspeed_stage_2_offload",
    "deepspeed_stage_3",
    "deepspeed_stage_3_offload",
    "deepspeed_stage_3_offload_nvme",
    "fsdp",
    "fsdp_cpu_offload",
    "single_xla",
    "xla_fsdp",
    "xla",
    "single_tpu",
]


class CheckpointLoadingConfig(TypedConfig):
    path: Literal["best", "last", "hpc"] | str | Path | None = None
    """
    Checkpoint path to use when loading a checkpoint.

    - "best" will load the best checkpoint.
    - "last" will load the last checkpoint.
    - "hpc" will load the SLURM pre-empted checkpoint.
    - Any other string or Path will load the checkpoint from the specified path.
    """


class DirectoryConfig(TypedConfig):
    project_root: Path | None = None
    """
    Root directory for this project.

    This isn't specific to the run; it is the parent directory of all runs.
    """

    log: Path | None = None
    """Base directory for all experiment tracking (e.g., WandB, Tensorboard, etc.) files. If None, will use lltrainer/{id}/log/."""

    stdio: Path | None = None
    """stdout/stderr log directory to use for the trainer. If None, will use lltrainer/{id}/stdio/."""

    checkpoint: Path | None = None
    """Checkpoint directory to use for the trainer. If None, will use lltrainer/{id}/checkpoint/."""

    activation: Path | None = None
    """Activation directory to use for the trainer. If None, will use lltrainer/{id}/activation/."""

    profile: Path | None = None
    """Directory to save profiling information to. If None, will use lltrainer/{id}/profile/."""

    def resolve_run_root_directory(self, run_id: str) -> Path:
        if (project_root_dir := self.project_root) is None:
            project_root_dir = Path.cwd()

        # The default base dir is $CWD/lltrainer/{id}/
        base_dir = project_root_dir / "lltrainer"
        base_dir.mkdir(exist_ok=True)

        # Add a .gitignore file to the lltrainer directory
        #   which will ignore all files except for the .gitignore file itself
        gitignore_path = base_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.touch()
            gitignore_path.write_text("*\n")

        base_dir = base_dir / run_id
        base_dir.mkdir(exist_ok=True)

        return base_dir

    def resolve_subdirectory(
        self,
        run_id: str,
        # subdirectory: Literal["log", "stdio", "checkpoint", "activation", "profile"],
        subdirectory: str,
    ) -> Path:
        # The subdir will be $CWD/lltrainer/{id}/{log, stdio, checkpoint, activation}/
        if (subdir := getattr(self, subdirectory, None)) is not None:
            assert isinstance(
                subdir, Path
            ), f"Expected a Path for {subdirectory}, got {type(subdir)}"
            return subdir

        dir = self.resolve_run_root_directory(run_id)
        dir = dir / subdirectory
        dir.mkdir(exist_ok=True)
        return dir

    def resolve_log_directory_for_logger(
        self,
        run_id: str,
        logger: LoggerConfig,
    ) -> Path:
        if (log_dir := logger.log_dir) is not None:
            return log_dir

        # Save to lltrainer/{id}/log/{logger kind}/{id}/
        log_dir = self.resolve_subdirectory(run_id, "log")
        log_dir = log_dir / logger.kind

        return log_dir


class ReproducibilityConfig(TypedConfig):
    deterministic: bool | Literal["warn"] | None = None
    """
    If ``True``, sets whether PyTorch operations must use deterministic algorithms.
        Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
        that don't support deterministic mode. If not set, defaults to ``False``. Default: ``None``.
    """


class ModelCheckpointCallbackConfig(CallbackConfigBase):
    """Arguments for the ModelCheckpoint callback."""

    kind: Literal["model_checkpoint"] = "model_checkpoint"

    dirpath: str | Path | None = None
    """
    Directory path to save the model file. If `None`, we save to the checkpoint directory set in `config.directory`.
    """

    filename: str | None = None
    """
    Checkpoint filename.
        If None, a default template is used (see :attr:`ModelCheckpoint.CHECKPOINT_JOIN_CHAR`).
    """

    monitor: str | None = None
    """
    Quantity to monitor for saving checkpoints.
        If None, no metric is monitored and checkpoints are saved at the end of every epoch.
    """

    verbose: bool = False
    """Verbosity mode. If True, print additional information about checkpoints."""

    save_last: Literal[True, False, "link"] | None = "link"
    """
    Whether to save the last checkpoint.
        If True, saves a copy of the last checkpoint separately.
        If "link", creates a symbolic link to the last checkpoint.
    """

    save_top_k: int = 1
    """
    Number of best models to save.
        If -1, all models are saved.
        If 0, no models are saved.
    """

    save_weights_only: bool = False
    """Whether to save only the model's weights or the entire model object."""

    mode: str = "min"
    """
    One of "min" or "max".
        If "min", training will stop when the metric monitored has stopped decreasing.
        If "max", training will stop when the metric monitored has stopped increasing.
    """

    auto_insert_metric_name: bool = True
    """Whether to automatically insert the metric name in the checkpoint filename."""

    every_n_train_steps: int | None = None
    """
    Number of training steps between checkpoints.
        If None or 0, no checkpoints are saved during training.
    """

    train_time_interval: timedelta | None = None
    """
    Time interval between checkpoints during training.
        If None, no checkpoints are saved during training based on time.
    """

    every_n_epochs: int | None = None
    """
    Number of epochs between checkpoints.
        If None or 0, no checkpoints are saved at the end of epochs.
    """

    save_on_train_epoch_end: bool | None = None
    """
    Whether to run checkpointing at the end of the training epoch.
        If False, checkpointing runs at the end of the validation.
    """

    enable_version_counter: bool = True
    """Whether to append a version to the existing file name."""

    auto_append_metric: bool = True
    """If enabled, this will automatically add "-{monitor}" to the filename."""

    @staticmethod
    def _convert_string(input_string: str):
        # Find all variables enclosed in curly braces
        variables = re.findall(r"\{(.*?)\}", input_string)

        # Replace each variable with its corresponding key-value pair
        output_string = input_string
        for variable in variables:
            # If the name is something like {variable:format}, we shouldn't process the format.
            key_name = variable
            if ":" in variable:
                key_name, _ = variable.split(":", 1)
                continue

            # Replace '/' with '_' in the key name
            key_name = key_name.replace("/", "_")
            output_string = output_string.replace(
                f"{{{variable}}}", f"{key_name}={{{variable}}}"
            )

        return output_string

    @override
    def construct_callbacks(self, root_config):
        from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

        dirpath = self.dirpath or root_config.directory.resolve_subdirectory(
            root_config.id, "checkpoint"
        )

        # If `monitor` is not provided, we can use `config.primary_metric` if it is set.
        monitor = self.monitor
        mode = self.mode
        if (
            monitor is None
            and (primary_metric := root_config.primary_metric) is not None
        ):
            monitor = primary_metric.validation_monitor
            mode = primary_metric.mode

        filename = self.filename
        if self.auto_append_metric:
            if not filename:
                filename = "{epoch}-{step}"
            filename = f"{filename}-{{{monitor}}}"

        if self.auto_insert_metric_name and filename:
            new_filename = self._convert_string(filename)
            log.critical(
                f"Updated ModelCheckpoint filename: {filename} -> {new_filename}"
            )
            filename = new_filename

        yield ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            verbose=self.verbose,
            save_last=self.save_last,
            save_top_k=self.save_top_k,
            save_weights_only=self.save_weights_only,
            auto_insert_metric_name=False,
            every_n_train_steps=self.every_n_train_steps,
            train_time_interval=self.train_time_interval,
            every_n_epochs=self.every_n_epochs,
            save_on_train_epoch_end=self.save_on_train_epoch_end,
            enable_version_counter=self.enable_version_counter,
        )


class LatestEpochCheckpointCallbackConfig(CallbackConfigBase):
    kind: Literal["latest_epoch_checkpoint"] = "latest_epoch_checkpoint"

    dirpath: str | Path | None = None
    """Directory path to save the checkpoint file."""

    filename: str | None = None
    """Checkpoint filename. This must not include the extension. If `None`, `latest_epoch_{id}_{timestamp}` is used."""

    save_weights_only: bool = False
    """Whether to save only the model's weights or the entire model object."""

    @override
    def construct_callbacks(self, root_config):
        from ..callbacks.latest_epoch_checkpoint import LatestEpochCheckpoint

        dirpath = self.dirpath or root_config.directory.resolve_subdirectory(
            root_config.id, "checkpoint"
        )

        yield LatestEpochCheckpoint(
            dirpath=dirpath,
            filename=self.filename,
            save_weights_only=self.save_weights_only,
        )


class OnExceptionCheckpointCallbackConfig(CallbackConfigBase):
    kind: Literal["on_exception_checkpoint"] = "on_exception_checkpoint"

    dirpath: str | Path | None = None
    """Directory path to save the checkpoint file."""

    filename: str | None = None
    """Checkpoint filename. This must not include the extension. If `None`, `on_exception_{id}_{timestamp}` is used."""

    @override
    def construct_callbacks(self, root_config):
        from ..callbacks.on_exception_checkpoint import OnExceptionCheckpoint

        dirpath = self.dirpath or root_config.directory.resolve_subdirectory(
            root_config.id, "checkpoint"
        )

        if not (filename := self.filename):
            filename = f"on_exception_{root_config.id}"
        yield OnExceptionCheckpoint(dirpath=dirpath, filename=filename)


CheckpointCallbackConfig: TypeAlias = Annotated[
    ModelCheckpointCallbackConfig
    | LatestEpochCheckpointCallbackConfig
    | OnExceptionCheckpointCallbackConfig,
    Field(discriminator="kind"),
]


class CheckpointSavingConfig(CallbackConfigBase):
    enabled: bool = True
    """Enable checkpoint saving."""

    checkpoint_callbacks: Sequence[CheckpointCallbackConfig] = [
        ModelCheckpointCallbackConfig(),
        LatestEpochCheckpointCallbackConfig(),
        OnExceptionCheckpointCallbackConfig(),
    ]
    """Checkpoint callback configurations."""

    def disable_(self):
        self.enabled = False
        return self

    def should_save_checkpoints(self, root_config: "BaseConfig"):
        if not self.enabled:
            return False

        if root_config.trainer.fast_dev_run:
            return False

        return True

    @property
    def model_checkpoint(self) -> ModelCheckpointCallbackConfig | None:
        return next(
            (
                callback
                for callback in self.checkpoint_callbacks
                if isinstance(callback, ModelCheckpointCallbackConfig)
            ),
        )

    @property
    def latest_epoch_checkpoint(self) -> LatestEpochCheckpointCallbackConfig | None:
        return next(
            (
                callback
                for callback in self.checkpoint_callbacks
                if isinstance(callback, LatestEpochCheckpointCallbackConfig)
            ),
        )

    @property
    def on_exception_checkpoint(self) -> OnExceptionCheckpointCallbackConfig | None:
        return next(
            (
                callback
                for callback in self.checkpoint_callbacks
                if isinstance(callback, OnExceptionCheckpointCallbackConfig)
            ),
        )

    @override
    def construct_callbacks(self, root_config: "BaseConfig"):
        if not self.should_save_checkpoints(root_config):
            return

        for callback_config in self.checkpoint_callbacks:
            yield from callback_config.construct_callbacks(root_config)


class LightningTrainerKwargs(TypedDict, total=False):
    accelerator: str | Accelerator
    """Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
    as well as custom accelerator instances."""

    strategy: str | Strategy
    """Supports different training strategies with aliases as well custom strategies.
    Default: ``"auto"``.
    """

    devices: list[int] | str | int
    """The devices to use. Can be set to a positive number (int or str), a sequence of device indices
    (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for
    automatic selection based on the chosen accelerator. Default: ``"auto"``.
    """

    num_nodes: int
    """Number of GPU nodes for distributed training.
    Default: ``1``.
    """

    precision: _PRECISION_INPUT | None
    """Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
    16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
    Can be used on CPU, GPU, TPUs, HPUs or IPUs.
    Default: ``'32-true'``.
    """

    logger: Logger | Iterable[Logger] | bool | None
    """Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
    the default ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``.
    ``False`` will disable logging. If multiple loggers are provided, local files
    (checkpoints, profiler traces, etc.) are saved in the ``log_dir`` of the first logger.
    Default: ``True``.
    """

    callbacks: list[Callback] | Callback | None
    """Add a callback or list of callbacks.
    Default: ``None``.
    """

    fast_dev_run: int | bool
    """Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
    of train, val and test to find any bugs (ie: a sort of unit test).
    Default: ``False``.
    """

    max_epochs: int | None
    """Stop training once this number of epochs is reached. Disabled by default (None).
    If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
    To enable infinite training, set ``max_epochs = -1``.
    """

    min_epochs: int | None
    """Force training for at least these many epochs. Disabled by default (None).
    """

    max_steps: int
    """Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
    and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
    ``max_epochs`` to ``-1``.
    """

    min_steps: int | None
    """Force training for at least these number of steps. Disabled by default (``None``).
    """

    max_time: str | timedelta | dict[str, int] | None
    """Stop training after this amount of time has passed. Disabled by default (``None``).
    The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
    :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
    :class:`datetime.timedelta`.
    """

    limit_train_batches: int | float | None
    """How much of training dataset to check (float = fraction, int = num_batches).
    Default: ``1.0``.
    """

    limit_val_batches: int | float | None
    """How much of validation dataset to check (float = fraction, int = num_batches).
    Default: ``1.0``.
    """

    limit_test_batches: int | float | None
    """How much of test dataset to check (float = fraction, int = num_batches).
    Default: ``1.0``.
    """

    limit_predict_batches: int | float | None
    """How much of prediction dataset to check (float = fraction, int = num_batches).
    Default: ``1.0``.
    """

    overfit_batches: int | float
    """Overfit a fraction of training/validation data (float) or a set number of batches (int).
    Default: ``0.0``.
    """

    val_check_interval: int | float | None
    """How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
    after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
    batches. An ``int`` value can only be higher than the number of training batches when
    ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
    across epochs or during iteration-based training.
    Default: ``1.0``.
    """

    check_val_every_n_epoch: int | None
    """Perform a validation loop every after every `N` training epochs. If ``None``,
    validation will be done solely based on the number of training batches, requiring ``val_check_interval``
    to be an integer value.
    Default: ``1``.
    """

    num_sanity_val_steps: int | None
    """Sanity check runs n validation batches before starting the training routine.
    Set it to `-1` to run all batches in all validation dataloaders.
    Default: ``2``.
    """

    log_every_n_steps: int | None
    """How often to log within steps.
    Default: ``50``.
    """

    enable_checkpointing: bool | None
    """If ``True``, enable checkpointing.
    It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.callbacks`.
    Default: ``True``.
    """

    enable_progress_bar: bool | None
    """Whether to enable to progress bar by default.
    Default: ``True``.
    """

    enable_model_summary: bool | None
    """Whether to enable model summarization by default.
    Default: ``True``.
    """

    accumulate_grad_batches: int
    """Accumulates gradients over k batches before stepping the optimizer.
    Default: 1.
    """

    gradient_clip_val: int | float | None
    """The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
    gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
    Default: ``None``.
    """

    gradient_clip_algorithm: str | None
    """The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
    to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
    be set to ``"norm"``.
    """

    deterministic: bool | Literal["warn"] | None
    """If ``True``, sets whether PyTorch operations must use deterministic algorithms.
    Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
    that don't support deterministic mode. If not set, defaults to ``False``. Default: ``None``.
    """

    benchmark: bool | None
    """The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
    The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
    (``False`` if not manually set). If :paramref:`~lightning.pytorch.trainer.trainer.Trainer.deterministic`
    is set to ``True``, this will default to ``False``. Override to manually set a different value.
    Default: ``None``.
    """

    inference_mode: bool
    """Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during
    evaluation (``validate``/``test``/``predict``).
    """

    use_distributed_sampler: bool
    """Whether to wrap the DataLoader's sampler with
    :class:`torch.utils.data.DistributedSampler`. If not specified this is toggled automatically for
    strategies that require it. By default, it will add ``shuffle=True`` for the train sampler and
    ``shuffle=False`` for validation/test/predict samplers. If you want to disable this logic, you can pass
    ``False`` and add your own distributed sampler in the dataloader hooks. If ``True`` and a distributed
    sampler was already added, Lightning will not replace the existing one. For iterable-style datasets,
    we don't do this automatically.
    """

    profiler: Profiler | str | None
    """To profile individual steps during training and assist in identifying bottlenecks.
    Default: ``None``.
    """

    detect_anomaly: bool
    """Enable anomaly detection for the autograd engine.
    Default: ``False``.
    """

    barebones: bool
    """Whether to run in "barebones mode", where all features that may impact raw speed are
    disabled. This is meant for analyzing the Trainer overhead and is discouraged during regular training
    runs. The following features are deactivated:
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_checkpointing`,
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.logger`,
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_progress_bar`,
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.log_every_n_steps`,
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_model_summary`,
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.num_sanity_val_steps`,
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.fast_dev_run`,
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.detect_anomaly`,
    :paramref:`~lightning.pytorch.trainer.trainer.Trainer.profiler`,
    :meth:`~lightning.pytorch.core.LightningModule.log`,
    :meth:`~lightning.pytorch.core.LightningModule.log_dict`.
    """

    plugins: _PLUGIN_INPUT | list[_PLUGIN_INPUT] | None
    """Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
    Default: ``None``.
    """

    sync_batchnorm: bool
    """Synchronize batch norm layers between process groups/whole world.
    Default: ``False``.
    """

    reload_dataloaders_every_n_epochs: int
    """Set to a positive integer to reload dataloaders every n epochs.
    Default: ``0``.
    """

    default_root_dir: Path | None
    """Default path for logs and weights when no logger/ckpt_callback passed.
    Default: ``os.getcwd()``.
    Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
    """


class EarlyStoppingConfig(CallbackConfigBase):
    monitor: str | None = None
    """
    The metric to monitor for early stopping.
    If None, the primary metric will be used.
    """

    mode: Literal["min", "max"] | None = None
    """
    The mode for the metric to monitor for early stopping.
    If None, the primary metric mode will be used.
    """

    patience: int
    """
    Number of epochs with no improvement after which training will be stopped.
    """

    min_delta: float = 1.0e-8
    """
    Minimum change in the monitored quantity to qualify as an improvement.
    """

    min_lr: float | None = None
    """
    Minimum learning rate. If the learning rate of the model is less than this value,
    the training will be stopped.
    """

    strict: bool = True
    """
    Whether to enforce that the monitored quantity must improve by at least `min_delta`
    to qualify as an improvement.
    """

    @override
    def construct_callbacks(self, root_config: "BaseConfig"):
        from ..callbacks.early_stopping import EarlyStopping

        monitor = self.monitor
        mode = self.mode
        if monitor is None:
            assert mode is None, "If `monitor` is not provided, `mode` must be None."

            primary_metric = root_config.primary_metric
            if primary_metric is None:
                raise ValueError(
                    "No primary metric is set, so `monitor` must be provided in `early_stopping`."
                )
            monitor = primary_metric.validation_monitor
            mode = primary_metric.mode

        if mode is None:
            mode = "min"

        return [
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=self.patience,
                min_delta=self.min_delta,
                min_lr=self.min_lr,
                strict=self.strict,
            )
        ]


class ActSaveConfig(CallbackConfigBase):
    enabled: bool = True
    """Enable activation saving."""

    auto_save_logged_metrics: bool = False
    """If enabled, will automatically save logged metrics (using `LightningModule.log`) as activations."""

    save_dir: Path | None = None
    """Directory to save activations to. If None, will use the activation directory set in `config.directory`."""

    def __bool__(self):
        return self.enabled

    def resolve_save_dir(self, root_config: "BaseConfig"):
        if self.save_dir is not None:
            return self.save_dir

        return root_config.directory.resolve_subdirectory(root_config.id, "activation")

    @override
    def construct_callbacks(self, root_config):
        from ..actsave import ActSaveCallback

        return [ActSaveCallback()]


class SanityCheckingConfig(TypedConfig):
    reduce_lr_on_plateau: Literal["disable", "error", "warn"] = "error"
    """
    If enabled, will do some sanity checks if the `ReduceLROnPlateau` scheduler is used:
        - If the `interval` is step, it makes sure that validation is called every `frequency` steps.
        - If the `interval` is epoch, it makes sure that validation is called every `frequency` epochs.
    Valid values are: "disable", "warn", "error".
    """


class TrainerConfig(TypedConfig):
    checkpoint_loading: CheckpointLoadingConfig = CheckpointLoadingConfig()
    """Checkpoint loading configuration options."""

    checkpoint_saving: CheckpointSavingConfig = CheckpointSavingConfig()
    """Checkpoint saving configuration options."""

    logging: LoggingConfig = LoggingConfig()
    """Logging/experiment tracking (e.g., WandB) configuration options."""

    optimizer: OptimizationConfig = OptimizationConfig()
    """Optimization configuration options."""

    reproducibility: ReproducibilityConfig = ReproducibilityConfig()
    """Reproducibility configuration options."""

    sanity_checking: SanityCheckingConfig = SanityCheckingConfig()
    """Sanity checking configuration options."""

    actsave: ActSaveConfig | None = ActSaveConfig(enabled=False)
    """Activation saving configuration options."""

    early_stopping: EarlyStoppingConfig | None = None
    """Early stopping configuration options."""

    profiler: ProfilerConfig | None = None
    """
    To profile individual steps during training and assist in identifying bottlenecks.
        Default: ``None``.
    """

    callbacks: list[CallbackConfig] = []
    """Callbacks to use during training."""

    detect_anomaly: bool | None = None
    """Enable anomaly detection for the autograd engine.
    Default: ``False``.
    """

    plugins: list[PluginConfigProtocol] | None = None
    """
    Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
        Default: ``None``.
    """

    auto_determine_num_nodes: bool = True
    """
    If enabled, will automatically determine the number of nodes for distributed training.

    This will only work on:
    - SLURM clusters
    - LSF clusters
    """

    fast_dev_run: int | bool = False
    """Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
    of train, val and test to find any bugs (ie: a sort of unit test).
    Default: ``False``.
    """

    precision: (
        Literal[
            "64-true",
            "32-true",
            "fp16-mixed",
            "bf16-mixed",
            "16-mixed-auto",
        ]
        | None
    ) = None
    """
    Training precision. Can be one of:
        - "64-true": Double precision (64-bit).
        - "32-true": Full precision (32-bit).
        - "fp16-mixed": Float16 mixed precision.
        - "bf16-mixed": BFloat16 mixed precision.
        - "16-mixed-auto": Automatic 16-bit: Uses bfloat16 if available, otherwise float16.
    """

    max_epochs: int | None = None
    """Stop training once this number of epochs is reached. Disabled by default (None).
    If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
    To enable infinite training, set ``max_epochs = -1``.
    """

    min_epochs: int | None = None
    """Force training for at least these many epochs. Disabled by default (None).
    """

    max_steps: int = -1
    """Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
    and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
    ``max_epochs`` to ``-1``.
    """

    min_steps: int | None = None
    """Force training for at least these number of steps. Disabled by default (``None``).
    """

    max_time: str | timedelta | dict[str, int] | None = None
    """Stop training after this amount of time has passed. Disabled by default (``None``).
    The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
    :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
    :class:`datetime.timedelta`.
    """

    limit_train_batches: int | float | None = None
    """How much of training dataset to check (float = fraction, int = num_batches).
    Default: ``1.0``.
    """

    limit_val_batches: int | float | None = None
    """How much of validation dataset to check (float = fraction, int = num_batches).
    Default: ``1.0``.
    """

    limit_test_batches: int | float | None = None
    """How much of test dataset to check (float = fraction, int = num_batches).
    Default: ``1.0``.
    """

    limit_predict_batches: int | float | None = None
    """How much of prediction dataset to check (float = fraction, int = num_batches).
    Default: ``1.0``.
    """

    overfit_batches: int | float = 0.0
    """Overfit a fraction of training/validation data (float) or a set number of batches (int).
    Default: ``0.0``.
    """

    val_check_interval: int | float | None = None
    """How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
    after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
    batches. An ``int`` value can only be higher than the number of training batches when
    ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
    across epochs or during iteration-based training.
    Default: ``1.0``.
    """

    check_val_every_n_epoch: int | None = 1
    """Perform a validation loop every after every `N` training epochs. If ``None``,
    validation will be done solely based on the number of training batches, requiring ``val_check_interval``
    to be an integer value.
    Default: ``1``.
    """

    num_sanity_val_steps: int | None = None
    """Sanity check runs n validation batches before starting the training routine.
    Set it to `-1` to run all batches in all validation dataloaders.
    Default: ``2``.
    """

    log_every_n_steps: int | None = None
    """How often to log within steps.
    Default: ``50``.
    """

    inference_mode: bool = True
    """Whether to use :func:`torch.inference_mode` (if `True`) or :func:`torch.no_grad` (if `False`) during evaluation (``validate``/``test``/``predict``).
    Default: ``True``.
    """

    use_distributed_sampler: bool | None = None
    """Whether to wrap the DataLoader's sampler with
    :class:`torch.utils.data.DistributedSampler`. If not specified this is toggled automatically for
    strategies that require it. By default, it will add ``shuffle=True`` for the train sampler and
    ``shuffle=False`` for validation/test/predict samplers. If you want to disable this logic, you can pass
    ``False`` and add your own distributed sampler in the dataloader hooks. If ``True`` and a distributed
    sampler was already added, Lightning will not replace the existing one. For iterable-style datasets,
    we don't do this automatically.
    Default: ``True``.
    """

    accelerator: AcceleratorConfigProtocol | AcceleratorLiteral | None = None
    """Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
    as well as custom accelerator instances.
    Default: ``"auto"``.
    """

    strategy: StrategyConfigProtocol | StrategyLiteral | None = None
    """Supports different training strategies with aliases as well custom strategies.
    Default: ``"auto"``.
    """

    devices: tuple[int, ...] | Sequence[int] | Literal["auto", "all"] | None = None
    """The devices to use. Can be set to a sequence of device indices, "all" to indicate all available devices should be used, or ``"auto"`` for
    automatic selection based on the chosen accelerator. Default: ``"auto"``.
    """

    auto_wrap_trainer: bool = True
    """If enabled, will automatically wrap the `run` function with a `Trainer.context()` context manager. Should be `True` most of the time."""
    auto_set_default_root_dir: bool = True
    """If enabled, will automatically set the default root dir to [cwd/lightning_logs/<id>/]. There is basically no reason to disable this."""
    supports_shared_parameters: bool = True
    """If enabled, the model supports scaling the gradients of shared parameters that are registered using `LightningModuleBase.register_shared_parameters(...)`"""

    lightning_kwargs: LightningTrainerKwargs = LightningTrainerKwargs()
    """
    Additional keyword arguments to pass to the Lightning `pl.Trainer` constructor.

    Please refer to the Lightning documentation for a list of valid keyword arguments.
    """

    additional_lightning_kwargs: dict[str, Any] = {}
    """
    Additional keyword arguments to pass to the Lightning `pl.Trainer` constructor.

    This is essentially a non-type-checked version of `lightning_kwargs`.
    """

    set_float32_matmul_precision: Literal["medium", "high", "highest"] | None = None
    """If enabled, will set the torch float32 matmul precision to the specified value. Useful for faster training on Ampere+ GPUs."""


class SeedConfig(TypedConfig):
    seed: int
    """Seed for the random number generator."""

    seed_workers: bool = False
    """Whether to seed the workers of the dataloader."""


Signal: TypeAlias = Literal[
    "SIGHUP",
    "SIGINT",
    "SIGQUIT",
    "SIGILL",
    "SIGTRAP",
    "SIGABRT",
    "SIGBUS",
    "SIGFPE",
    "SIGKILL",
    "SIGUSR1",
    "SIGSEGV",
    "SIGUSR2",
    "SIGPIPE",
    "SIGALRM",
    "SIGTERM",
    "SIGCHLD",
    "SIGCONT",
    "SIGSTOP",
    "SIGTSTP",
    "SIGTTIN",
    "SIGTTOU",
    "SIGURG",
    "SIGXCPU",
    "SIGXFSZ",
    "SIGVTALRM",
    "SIGPROF",
    "SIGWINCH",
    "SIGIO",
    "SIGPWR",
    "SIGSYS",
    "SIGRTMIN",
    "SIGRTMAX",
]


class SubmitConfig(TypedConfig):
    auto_requeue_signals: list[Signal] = [
        # "SIGUSR1",
        # On SIGURG:
        # Important note from https://amrex-astro.github.io/workflow/olcf-workflow.html:
        # We can also ask the job manager to send a warning signal some amount of time before the allocation expires by passing -wa 'signal' and -wt '[hour:]minute' to bsub. We can then have bash create a dump_and_stop file when it receives the signal, which will tell Castro to output a checkpoint file and exit cleanly after it finishes the current timestep. An important detail that I couldn't find documented anywhere is that the job manager sends the signal to all the processes in the job, not just the submission script, and we have to use a signal that is ignored by default so Castro doesn't immediately crash upon receiving it. SIGCHLD, SIGURG, and SIGWINCH are the only signals that fit this requirement and of these, SIGURG is the least likely to be triggered by other events.
        "SIGURG"
    ]
    """Signals that will trigger an automatic requeue of the job."""

    def _resolved_auto_requeue_signals(self) -> list[signal.Signals]:
        return [getattr(signal.Signals, sig) for sig in self.auto_requeue_signals]


class RunnerConfig(TypedConfig):
    python_logging: PythonLogging = PythonLogging()
    """Python logging configuration options."""

    seed: SeedConfig = SeedConfig(seed=0)
    """Seed everything configuration options."""

    submit: SubmitConfig = SubmitConfig()
    """Submit (e.g., SLURM or LSF) configuration options."""

    dump_run_information: bool = True
    """
    If enabled, will dump different bits of run information to the output directory before starting the run.
    This includes:
        - Run config
        - Full set of environment variables
    """

    additional_env_vars: dict[str, str] = {}
    """Additional environment variables to set when running the script."""


class MetricConfig(TypedConfig):
    name: str
    """The name of the primary metric."""

    mode: Literal["min", "max"]
    """
    The mode of the primary metric:
    - "min" for metrics that should be minimized (e.g., loss)
    - "max" for metrics that should be maximized (e.g., accuracy)
    """

    @property
    def validation_monitor(self) -> str:
        return f"val/{self.name}"

    def __post_init__(self):
        for split in ("train", "val", "test", "predict"):
            if self.name.startswith(f"{split}/"):
                raise ValueError(
                    f"Primary metric name should not start with '{split}/'. "
                    f"Just use '{self.name[len(split) + 1:]}' instead. "
                    "The split name is automatically added depending on the context."
                )

    @classmethod
    def loss(cls, mode: Literal["min", "max"] = "min"):
        return cls(name="loss", mode=mode)


PrimaryMetricConfig: TypeAlias = MetricConfig


class BaseConfig(TypedConfig):
    id: str = Field(default_factory=lambda: BaseConfig.generate_id())
    """ID of the run."""
    name: str | None = None
    """Run name."""
    name_parts: list[str] = []
    """A list of parts used to construct the run name. This is useful for constructing the run name dynamically."""
    project: str | None = None
    """Project name."""
    tags: list[str] = []
    """Tags for the run."""
    notes: list[str] = []
    """Human readable notes for the run."""

    debug: bool = False
    """Whether to run in debug mode. This will enable debug logging and enable debug code paths."""
    environment: Annotated[EnvironmentConfig, Field(repr=False)] = EnvironmentConfig()
    """A snapshot of the current environment information (e.g. python version, slurm info, etc.). This is automatically populated by the run script."""

    directory: DirectoryConfig = DirectoryConfig()
    """Directory configuration options."""
    trainer: TrainerConfig = TrainerConfig()
    """PyTorch Lightning trainer configuration options. Check Lightning's `Trainer` documentation for more information."""
    runner: RunnerConfig = RunnerConfig()
    """`ll.Runner` configuration options."""

    primary_metric: PrimaryMetricConfig | None = None
    """Primary metric configuration options. This is used in the following ways:
    - To determine the best model checkpoint to save with the ModelCheckpoint callback.
    - To monitor the primary metric during training and stop training based on the `early_stopping` configuration.
    - For the ReduceLROnPlateau scheduler.
    """

    meta: dict[str, Any] = {}
    """Additional metadata for this run. This can be used to store arbitrary data that is not part of the config schema."""

    @property
    def run_name(self) -> str:
        parts = self.name_parts.copy()
        if self.name is not None:
            parts = [self.name] + parts
        name = "-".join(parts)
        if not name:
            name = self.id
        return name

    def clone(self, with_new_id: bool = True) -> Self:
        c = copy.deepcopy(self)
        if with_new_id:
            c.id = BaseConfig.generate_id()
        return c

    def subdirectory(self, subdirectory: str) -> Path:
        return self.directory.resolve_subdirectory(self.id, subdirectory)

    # region Helper methods
    def with_project_root_(self, project_root: str | Path | os.PathLike) -> Self:
        """
        Set the project root directory for the trainer.

        Args:
            project_root (Path): The base directory to use.

        Returns:
            self: The current instance of the class.
        """
        self.directory.project_root = Path(project_root)
        return self

    def reset_(
        self,
        *,
        id: bool = True,
        basic: bool = True,
        project_root: bool = True,
        environment: bool = True,
        meta: bool = True,
    ):
        """
        Reset the configuration object to its initial state.

        Parameters:
        - id (bool): If True, generate a new ID for the configuration object.
        - basic (bool): If True, reset basic attributes like name, project, tags, and notes.
        - project_root (bool): If True, reset the directory configuration to its initial state.
        - environment (bool): If True, reset the environment configuration to its initial state.
        - meta (bool): If True, reset the meta dictionary to an empty dictionary.

        Returns:
        - self: The updated configuration object.

        """
        if id:
            self.id = self.generate_id()

        if basic:
            self.name = None
            self.name_parts = []
            self.project = None
            self.tags = []
            self.notes = []

        if project_root:
            self.directory = DirectoryConfig()

        if environment:
            self.environment = EnvironmentConfig()

        if meta:
            self.meta = {}

        return self

    def concise_repr(self) -> str:
        """Get a concise representation of the configuration object."""

        def _truncate(s: str, max_len: int = 50):
            return s if len(s) <= max_len else f"{s[:max_len - 3]}..."

        cls_name = self.__class__.__name__

        parts: list[str] = []
        parts.append(f"name={self.run_name}")
        if self.project:
            parts.append(f"project={_truncate(self.project)}")

        return f"{cls_name}({', '.join(parts)})"

    # endregion

    # region Seeding

    _rng: ClassVar[np.random.Generator | None] = None

    @staticmethod
    def generate_id(
        *,
        length: int = 8,
        ignore_rng: bool = False,
    ) -> str:
        """
        Generate a random ID of specified length.

        Args:
            length (int): The length of the generated ID. Default is 8.
            ignore_rng (bool): If True, ignore the global random number generator and use a new one. Default is False.

        Returns:
            str: The generated random ID.

        Raises:
            IdSeedWarning: If the global random number generator is None and ignore_rng is False.

        Notes:
            - The generated IDs will not be reproducible if the global random number generator is None and ignore_rng is False.
            - To ensure reproducibility, call BaseConfig.set_seed(...) before generating any IDs.
        """
        rng = BaseConfig._rng if not ignore_rng else np.random.default_rng()
        if rng is None:
            warnings.warn(
                "BaseConfig._rng is None. The generated IDs will not be reproducible. "
                + "To fix this, call BaseConfig.set_seed(...) before generating any IDs.",
                category=IdSeedWarning,
            )
            rng = np.random.default_rng()

        alphabet = list(string.ascii_lowercase + string.digits)

        id = "".join(rng.choice(alphabet) for _ in range(length))
        return id

    @staticmethod
    def set_seed(seed: int | None = None) -> None:
        """
        Set the seed for the random number generator.

        Args:
            seed (int | None, optional): The seed value to set. If None, a seed based on the current time will be used. Defaults to None.

        Returns:
            None
        """
        if seed is None:
            seed = int(time.time() * 1000)
        log.critical(f"Seeding BaseConfig with seed {seed}")
        BaseConfig._rng = np.random.default_rng(seed)

    # endregion

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        hparams_key: str = "hyper_parameters",
    ):
        ckpt = torch.load(path)
        if (hparams := ckpt.get(hparams_key)) is None:
            raise ValueError(
                f"The checkpoint does not contain the `{hparams_key}` attribute. "
                "Are you sure this is a valid Lightning checkpoint?"
            )
        return cls.model_validate(hparams)

    def ll_all_callback_configs(self) -> Iterable[CallbackConfigBase | None]:
        yield self.trainer.actsave
        yield self.trainer.early_stopping
        yield self.trainer.checkpoint_saving
        yield self.trainer.logging
        yield self.trainer.optimizer
        yield from self.trainer.callbacks
