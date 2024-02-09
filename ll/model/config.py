import string
import time
import warnings
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import numpy as np
from lightning.pytorch.profilers import Profiler
from typing_extensions import override

from ..config import Field, TypedConfig

logger = getLogger(__name__)


class IdSeedWarning(Warning):
    pass


class BaseProfilerConfig(TypedConfig, ABC):
    dirpath: str | Path | None = None
    filename: str | None = None

    @abstractmethod
    def construct_profiler(self) -> Profiler: ...


class SimpleProfilerConfig(BaseProfilerConfig):
    kind: Literal["simple"] = "simple"

    extended: bool = True

    @override
    def construct_profiler(self):
        from lightning.pytorch.profilers.simple import SimpleProfiler

        return SimpleProfiler(
            extended=self.extended,
            dirpath=self.dirpath,
            filename=self.filename,
        )


class AdvancedProfilerConfig(BaseProfilerConfig):
    kind: Literal["advanced"] = "advanced"

    line_count_restriction: float = 1.0

    @override
    def construct_profiler(self):
        from lightning.pytorch.profilers.advanced import AdvancedProfiler

        return AdvancedProfiler(
            line_count_restriction=self.line_count_restriction,
            dirpath=self.dirpath,
            filename=self.filename,
        )


class PyTorchProfilerConfig(BaseProfilerConfig):
    kind: Literal["pytorch"] = "pytorch"

    group_by_input_shapes: bool = False
    emit_nvtx: bool = False
    export_to_chrome: bool = True
    row_limit: int = 20
    sort_by_key: str | None = None
    record_module_names: bool = True
    table_kwargs: dict[str, Any] | None = None
    additional_profiler_kwargs: dict[str, Any] = {}

    @override
    def construct_profiler(self):
        from lightning.pytorch.profilers.pytorch import PyTorchProfiler

        return PyTorchProfiler(
            group_by_input_shapes=self.group_by_input_shapes,
            emit_nvtx=self.emit_nvtx,
            export_to_chrome=self.export_to_chrome,
            row_limit=self.row_limit,
            sort_by_key=self.sort_by_key,
            record_module_names=self.record_module_names,
            table_kwargs=self.table_kwargs,
            dirpath=self.dirpath,
            filename=self.filename,
            **self.additional_profiler_kwargs,
        )


ProfilerConfig: TypeAlias = Annotated[
    SimpleProfilerConfig | AdvancedProfilerConfig | PyTorchProfilerConfig,
    Field(discriminator="kind"),
]


class EnvironmentConfig(TypedConfig):
    cwd: str | None = None

    python_executable: str | None = None
    python_path: list[str] | None = None
    python_version: str | None = None

    config: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    data: dict[str, Any] | None = None

    slurm: dict[str, Any] | None = None

    log_dir: str | None = None

    seed: int | None = None
    seed_workers: bool | None = None

    sweep_id: str | None = None
    sweep_config: dict[str, Any] | None = None


class WandbWatchConfig(TypedConfig):
    enabled: bool = True
    """Enable watching the model for wandb."""

    log: str | None = None
    log_graph: bool = True
    log_freq: int = 100


class WandbLoggingConfig(TypedConfig):
    enabled: bool = True
    """Enable logging to wandb."""

    log_model: bool | str = False
    """
    Whether to log the model checkpoints to wandb.
    Valid values are:
        - False: Do not log the model checkpoints.
        - True: Log the latest model checkpoint.
        - "all": Log all model checkpoints.
    """

    watch: WandbWatchConfig = WandbWatchConfig()
    """WandB model watch configuration. Used to log model architecture, gradients, and parameters."""


class CSVLoggingConfig(TypedConfig):
    enabled: bool = True
    """Enable logging to CSV files."""


class TensorboardLoggingConfig(TypedConfig):
    enabled: bool = False
    """Enable logging to tensorboard."""


class LoggingConfig(TypedConfig):
    enabled: bool = True
    """Enable logging."""

    log_lr: bool | str = True
    """If enabled, will register a `LearningRateMonitor` callback to log the learning rate to the logger."""
    log_epoch: bool = True
    """If enabled, will log the fractional epoch number to the logger."""

    wandb: WandbLoggingConfig = WandbLoggingConfig()
    """WandB configuration"""

    csv: CSVLoggingConfig = CSVLoggingConfig()
    """CSV configuration"""

    tensorboard: TensorboardLoggingConfig = TensorboardLoggingConfig()
    """Tensorboard configuration"""


class GradientSkippingConfig(TypedConfig):
    enabled: bool = False
    """Enable gradient skipping."""
    norm_type: str | float = 2.0
    """Norm type to use for gradient skipping."""
    threshold: float = float("inf")
    """Threshold to use for gradient skipping."""
    start_after_n_steps: int | None = 100
    """Number of steps to wait before starting gradient skipping."""


class OptimizerConfig(TypedConfig):
    grad_finite_checks: bool = False
    """If enabled, will check that the gradients are finite after each backward pass."""
    grad_none_checks: bool = False
    """If enabled, will check that the gradients are not None after each backward pass."""

    log_grad_norm: bool | str | float = False
    """If enabled, will log the gradient norm (averaged across all model parameters) to the logger."""
    log_grad_norm_per_param: bool | str | float = False
    """If enabled, will log the gradient norm for each model parameter to the logger."""

    log_param_norm: bool | str | float = False
    """If enabled, will log the parameter norm (averaged across all model parameters) to the logger."""
    log_param_norm_per_param: bool | str | float = False
    """If enabled, will log the parameter norm for each model parameter to the logger."""

    gradient_skipping: GradientSkippingConfig = GradientSkippingConfig(enabled=False)
    """Gradient skipping configuration."""


class PythonLogging(TypedConfig):
    log_level: (
        Literal["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"] | None
    ) = None
    """Log level to use for the Python logger (or None to use the default)."""

    rich: bool = True
    """If enabled, will use the rich library to format the Python logger output."""
    rich_tracebacks: bool = True
    """If enabled, will use the rich library to format the Python logger tracebacks."""

    lovely_tensors: bool = True
    """If enabled, will use the lovely-tensors library to format PyTorch tensors."""
    lovely_numpy: bool = False
    """If enabled, will use the lovely-numpy library to format numpy arrays. False by default as it causes some issues with other libaries."""


class TrainerConfig(TypedConfig):
    python_logging: PythonLogging = PythonLogging()
    """Python logging configuration options."""

    logging: LoggingConfig = LoggingConfig()
    """Logging (e.g., WandB logging) configuration options."""

    optimizer: OptimizerConfig = OptimizerConfig()
    """Optimizer configuration options."""

    seed: int | None = 0
    """Seed for the random number generator. If None, will use a random seed."""
    seed_workers: bool = False
    """Whether to seed the workers of the dataloader."""
    default_ckpt_path: str | None = None
    """Default checkpoint path to use when loading a checkpoint. "last" will load the last checkpoint. "hpc" will load the SLURM pre-empted checkpoint."""

    auto_wrap_trainer: bool = True
    """If enabled, will automatically wrap the `run` function with a `Trainer.context()` context manager. Should be `True` most of the time."""
    auto_call_trainer_init_from_runner: bool = True
    """If enabled, will automatically call the Trainer.runner_init() function from the Runner. Should be `True` most of the time."""
    auto_set_default_root_dir: bool = True
    """If enabled, will automatically set the default root dir to [cwd/lightning_logs/<id>/]. Should be `True` most of the time."""
    auto_set_loggers: bool = True
    """If enabled, will automatically set the loggers to [WandbLogger, CSVLogger, TensorboardLogger] as defined in `config.logging`. Should be `True` most of the time."""
    checkpoint_last_by_default: bool = True
    """If enabled, will update the trainer to save the last checkpoint by default."""
    on_exception_checkpoint: bool = True
    """If enabled, will checkpoint the model when an exception is thrown during training."""
    auto_add_trainer_finalizer: bool = True
    """If enabled, will automatically finalize the trainer (e.g., call `wandb.finish()`) when the run ends. Should be `True` most of the time."""
    enable_logger_validation: bool = True
    """If enabled, will validate loggers. This makes sure that the logger's log_dirs are correct given the current config id. Should be `True` most of the time."""
    patch_hpc_checkpoint_connector: bool | None = None
    """If enabled, will patch Lightning's trainer to load the HPC checkpoint by default, even if `default_ckpt_path` is set to something else. This is deprecated and doesn't do anything."""

    supports_skip_batch_exception: bool = True
    """If enabled, the model supports skipping an entire batch by throwing a `SkipBatch` exception."""
    supports_shared_parameters: bool = True
    """If enabled, the model supports scaling the gradients of shared parameters that are registered using `LightningModuleBase.register_shared_parameters(...)`"""
    supports_parameter_hooks: bool = True
    """If enabled, the model supports registering parameter hooks using `LightningModuleBase.register_parameter_hook(...)`"""
    log_batch_info_on_error: bool = False
    """If enabled, will log the batch info (e.g. batch index, batch object, etc.) when an exception is thrown during training."""
    reduce_lr_on_plateau_sanity_checks: Literal["disable", "error", "warn"] = "error"
    """
    Valid values are: "disable", "warn", "error"
    If enabled, will do some sanity checks if the `ReduceLROnPlateau` scheduler is used:
        - If the `interval` is step, it makes sure that validation is called every `frequency` steps.
        - If the `interval` is epoch, it makes sure that validation is called every `frequency` epochs.
    """

    additional_trainer_kwargs: dict[str, Any] = {}
    """Additional keyword arguments to pass to the Lightning `pl.Trainer` constructor."""
    additional_env_vars: dict[str, str] = {}
    """Additional environment variables to set when running the trainer."""
    set_nccl_optimal_params: bool = False
    """If enabled, will set the NCCL optimal parameters when running on multiple GPUs + nodes."""

    set_float32_matmul_precision: Literal["medium", "high", "highest"] | None = None
    """If enabled, will set the torch float32 matmul precision to the specified value. Useful for faster training on Ampere+ GPUs."""

    accelerator: str = "auto"
    strategy: str = "auto"
    devices: str | int = "auto"
    num_nodes: str | int = "auto"
    precision: Literal[
        "32-true", "bf16-true", "bf16-mixed", "16-true", "16-mixed", 32, 16
    ] = "32-true"
    logger: bool | None = None
    fast_dev_run: int | bool = False
    max_epochs: int | None = None
    min_epochs: int | None = None
    max_steps: int = -1
    min_steps: int | None = None
    max_time: str | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None
    overfit_batches: int | float = 0.0
    val_check_interval: int | float | None = None
    check_val_every_n_epoch: int | None = 1
    num_sanity_val_steps: int | None = None
    log_every_n_steps: int = 50
    enable_checkpointing: bool | None = None
    enable_progress_bar: bool | None = None
    enable_model_summary: bool | None = None
    accumulate_grad_batches: int = 1
    automatic_gradient_clip: bool = True
    gradient_clip_val: int | float | None = None
    gradient_clip_algorithm: Literal["norm", "value"] | None = None
    deterministic: bool | str | None = None
    benchmark: bool | None = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    profiler: str | ProfilerConfig | None = None
    detect_anomaly: bool = False
    barebones: bool = False
    plugins: list[str] | None = None
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: str | Path | None = None


class BaseConfig(TypedConfig):
    id: str = Field(default_factory=lambda: BaseConfig.generate_id())
    """ID of the run."""
    name: str | None = None
    """Run name."""
    project: str | None = None
    """Project name."""
    tags: list[str] = []
    """Tags for the run."""
    notes: list[str] = []
    """Human readable notes for the run."""

    debug: bool = False
    """Whether to run in debug mode. This will enable debug logging and enable debug code paths."""
    environment: EnvironmentConfig = EnvironmentConfig()
    """A snapshot of the current environment information (e.g. python version, slurm info, etc.). This is automatically populated by the run script."""
    trainer: TrainerConfig = TrainerConfig()
    """PyTorch Lightning trainer configuration options. Check Lightning's `Trainer` documentation for more information."""

    """Additional metadata for this run. This can be used to store arbitrary data that is not part of the config schema."""
    meta: dict[str, Any] = {}

    # region Seeding

    _rng: ClassVar[np.random.Generator | None] = None

    @staticmethod
    def generate_id(
        *,
        length: int = 8,
        ignore_rng: bool = False,
    ) -> str:
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
        if seed is None:
            seed = int(time.time() * 1000)
        logger.critical(f"Seeding BaseConfig with seed {seed}")
        BaseConfig._rng = np.random.default_rng(seed)

    # endregion
