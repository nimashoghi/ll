import contextlib
import datetime
import hashlib
import logging
import os
from collections import abc
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import NoneType
from typing import Any

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT
from typing_extensions import override

from ..model.config import (
    BaseConfig,
    BaseProfilerConfig,
    CheckpointLoadingConfig,
    RunnerOutputSaveConfig,
)
from ..util import seed
from ..util.environment import set_additional_env_vars
from ..util.typing_utils import copy_method_with_param
from .logging import finalize_loggers

log = logging.getLogger(__name__)


def _stdio_log_dir(
    root_config: BaseConfig,
    config: RunnerOutputSaveConfig,
):
    """
    Save the output directory for the runner.

    Args:
        root_config (BaseConfig): The root configuration object.
        config (RunnerOutputSaveConfig): The configuration object for saving the output.

    Returns:
        Path: The resolved output directory path.
    """
    if not (dirpath := config.dirpath):
        dirpath = root_config.trainer.directory.resolve_subdirectory(
            root_config.id, "stdio"
        )

    dirpath = Path(dirpath).resolve()

    # Make sure that the directory exists
    dirpath.mkdir(parents=True, exist_ok=True)

    return dirpath


def _default_log_handlers(root_config: BaseConfig):
    """
    Returns a generator of log handlers based on the provided root configuration.

    Args:
        root_config (BaseConfig): The root configuration object.

    Yields:
        logging.Handler: A log handler object.

    """
    # Implementation goes here
    if (config := root_config.runner.save_output) is None or not config.enabled:
        return

    # Get the directory path
    dirpath = _stdio_log_dir(root_config, config)

    # Capture the logs to `dirpath`/log.log
    log_file = dirpath / "log.log"
    log_file.touch(exist_ok=True)
    yield logging.FileHandler(log_file)


class Trainer(LightningTrainer):
    _finalizers: list[Callable[[], None]] = []

    def finalize(self):
        """
        Call this method to clean up after training.
        """
        finalize_loggers(self.loggers)

    @classmethod
    def setup_python_logging(cls, root_config: BaseConfig):
        """
        Sets up the logger with the specified configurations.

        Args:
            root_config (BaseConfig): The root configuration object.
            config (PythonLogging): The Python logging configuration object.

        Returns:
            None
        """
        config = root_config.trainer.python_logging

        if config.lovely_tensors:
            try:
                import lovely_tensors

                lovely_tensors.monkey_patch()
            except ImportError:
                log.warning(
                    "Failed to import lovely-tensors. Ignoring pretty PyTorch tensor formatting"
                )

        if config.lovely_numpy:
            try:
                import lovely_numpy

                lovely_numpy.set_config(repr=lovely_numpy.lovely)
            except ImportError:
                log.warning(
                    "Failed to import lovely-numpy. Ignoring pretty numpy array formatting"
                )

        log_handlers: list[logging.Handler] = [*_default_log_handlers(root_config)]
        if config.rich:
            try:
                from rich.logging import RichHandler  # type: ignore

                log_handlers.append(RichHandler())
            except ImportError:
                log.warning(
                    "Failed to import rich. Falling back to default Python logging."
                )

        logging.basicConfig(
            level=config.log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=log_handlers,
        )

    @classmethod
    @contextlib.contextmanager
    def output_save_context(cls, root_config: BaseConfig):
        """
        A context manager that saves the output logs to a specified directory.

        Args:
            root_config (BaseConfig): The root configuration object.

        Yields:
            None: This context manager does not return any value.

        Example:
            with Trainer.output_save_context(root_config):
                # Code block where the output logs will be saved
        """
        if (config := root_config.runner.save_output) is None or not config.enabled:
            yield
            return

        # Get the directory path
        dirpath = _stdio_log_dir(root_config, config)

        # Capture the stdout and stderr logs to `dirpath`/stdout.log and `dirpath`/stderr.log
        stdout_log = dirpath / "stdout.log"
        stderr_log = dirpath / "stderr.log"
        stdout_log.touch(exist_ok=True)
        stderr_log.touch(exist_ok=True)
        with stdout_log.open("a") as file:
            with contextlib.redirect_stdout(file):
                with stderr_log.open("a") as file:
                    with contextlib.redirect_stderr(file):
                        yield

    @classmethod
    @contextlib.contextmanager
    def ll_initialize(cls, config: BaseConfig):
        """
        Context manager for initializing the trainer.

        Args:
            config (BaseConfig): The configuration object.

        Yields:
            None

        Raises:
            ValueError: If both `config.trainer.default_root_dir` and `config.trainer.auto_set_default_root_dir` are set.

        Example:
            with Trainer.ll_initialize(config):
                # Code to initialize the trainer
        """
        with contextlib.ExitStack() as stack:
            if not config.runner.auto_call_trainer_init_from_runner:
                stack.enter_context(cls.runner_init(config))

            # Set the default root directory
            if config.trainer.auto_set_default_root_dir:
                if config.trainer.default_root_dir:
                    raise ValueError(
                        "You have set `config.trainer.default_root_dir`. "
                        "But we are trying to set it automatically. "
                        "Please use `config.trainer.directory.base` rather than `config.trainer.default_root_dir`. "
                        "If you want to set it manually, please set `config.trainer.auto_set_default_root_dir=False`."
                    )
                config.trainer.default_root_dir = (
                    config.trainer.directory.resolve_base_directory(config.id)
                )
                log.info(f"Setting {config.trainer.default_root_dir=}.")

            yield

    @classmethod
    @contextlib.contextmanager
    def runner_init(cls, config: BaseConfig):
        """
        Context manager for initializing the runner.

        Used to set up the Python logging and save the stdout/stderr to a file.

        Args:
            config (BaseConfig): The configuration object.

        Yields:
            None

        """
        with contextlib.ExitStack() as stack:
            # Set up the Python logging
            cls.setup_python_logging(config)

            # Save stdout/stderr to a file.
            stack.enter_context(Trainer.output_save_context(config))
            yield

    @classmethod
    def ll_default_callbacks(cls, config: BaseConfig):
        """
        Returns a generator of default callbacks for the LL trainer, based on the provided configuration.

        Args:
            config (BaseConfig): The configuration object.

        Yields:
            Callback: The default callbacks for the LL trainer.
        """
        if config.trainer.checkpoint_saving.enabled:
            yield from config.trainer.checkpoint_saving.construct_callbacks(config)

        if config.trainer.python_logging.use_rich_progress_bar:
            yield RichProgressBar()

    @classmethod
    @contextlib.contextmanager
    def context(cls, config: BaseConfig):
        with contextlib.ExitStack() as stack:
            stack.enter_context(cls.ll_initialize(config))

            cls._finalizers.clear()
            if (
                seed_config := config.trainer.reproducibility.seed_everything
            ) is not None:
                stack.enter_context(
                    seed.seed_context(
                        seed_config.seed,
                        workers=seed_config.seed_workers,
                    )
                )

            additional_env_vars: dict[str, str] = {**config.trainer.additional_env_vars}
            if config.trainer.set_nccl_optimal_params:
                # We need to set these env vars before the NCCL library is loaded.
                # Reportedly, the training performance can be improved quite a bit, see
                #   https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html#on-a-multi-node-cluster-set-nccl-parameters
                # Details on all available env vars: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
                additional_env_vars["NCCL_NSOCKS_PERTHREAD"] = "4"
                additional_env_vars["NCCL_SOCKET_NTHREADS"] = "2"

            if (precision := config.trainer.set_float32_matmul_precision) is not None:
                torch.set_float32_matmul_precision(precision)

            stack.enter_context(set_additional_env_vars(additional_env_vars))

            try:
                yield
            finally:
                n_finalizers = 0
                for finalizer in reversed(cls._finalizers):
                    finalizer()
                    n_finalizers += 1

                cls._finalizers.clear()
                log.critical(
                    f"Ran {n_finalizers} finalizers for {cls.__name__} cleanup."
                )

    @classmethod
    def _update_kwargs(cls, config: BaseConfig, kwargs_ctor: dict[str, Any]):
        kwargs: dict[str, Any] = {
            "accelerator": config.trainer.accelerator,
            "strategy": config.trainer.strategy,
            "devices": config.trainer.devices,
            "num_nodes": config.trainer.num_nodes,
            "precision": config.trainer.precision,
            "logger": config.trainer.logging.enabled,
            "fast_dev_run": config.trainer.fast_dev_run,
            "max_epochs": config.trainer.max_epochs,
            "min_epochs": config.trainer.min_epochs,
            "max_steps": config.trainer.max_steps,
            "min_steps": config.trainer.min_steps,
            "max_time": config.trainer.max_time,
            "limit_train_batches": config.trainer.limit_train_batches,
            "limit_val_batches": config.trainer.limit_val_batches,
            "limit_test_batches": config.trainer.limit_test_batches,
            "limit_predict_batches": config.trainer.limit_predict_batches,
            "overfit_batches": config.trainer.overfit_batches,
            "val_check_interval": config.trainer.val_check_interval,
            "check_val_every_n_epoch": config.trainer.check_val_every_n_epoch,
            "num_sanity_val_steps": config.trainer.num_sanity_val_steps,
            "log_every_n_steps": config.trainer.log_every_n_steps,
            "enable_progress_bar": config.trainer.enable_progress_bar,
            "enable_model_summary": config.trainer.enable_model_summary,
            "accumulate_grad_batches": config.trainer.accumulate_grad_batches,
            "deterministic": config.trainer.reproducibility.deterministic,
            "benchmark": config.trainer.benchmark,
            "inference_mode": config.trainer.inference_mode,
            "use_distributed_sampler": config.trainer.use_distributed_sampler,
            "detect_anomaly": config.trainer.detect_anomaly,
            "barebones": config.trainer.barebones,
            "plugins": config.trainer.plugins,
            "sync_batchnorm": config.trainer.sync_batchnorm,
            "reload_dataloaders_every_n_epochs": config.trainer.reload_dataloaders_every_n_epochs,
            # Moved to `lightning_kwargs`:
            # "enable_checkpointing": config.trainer.enable_checkpointing,
        }

        if (
            grad_clip_config := config.trainer.optimizer.gradient_clipping
        ) is not None and grad_clip_config.enabled:
            kwargs["gradient_clip_algorithm"] = grad_clip_config.algorithm
            kwargs["gradient_clip_val"] = grad_clip_config.value

        if profiler := config.trainer.profiler:
            # If the profiler is an ProfilerConfig instance, then we instantiate it.
            if isinstance(profiler, BaseProfilerConfig):
                profiler = profiler.construct_profiler()
                # Make sure that the profiler is an instance of `Profiler`.
                if not isinstance(profiler, Profiler):
                    raise ValueError(f"{profiler=} is not an instance of `{Profiler}`.")

            # Otherwise, if the profiler is a string (e.g., "simpe", "advanced", "pytorch"),
            #   then we just pass it through.
            kwargs["profiler"] = profiler

        kwargs.update(kwargs_ctor)

        kwargs["plugins"] = []
        if config.trainer.plugins is not None:
            kwargs["plugins"].extend(config.trainer.plugins)
        if (plugins := kwargs_ctor.get("plugins")) is not None:
            plugins = [plugins] if not isinstance(plugins, list) else plugins
            kwargs["plugins"].extend(plugins)

        if config.trainer.logger is False:
            log.critical(f"Disabling logger because {config.trainer.logger=}.")
            kwargs["logger"] = False
        elif kwargs.get("logger") is False:
            log.critical(f"Disabling logger because {kwargs.get('logger')=}.")

        if (
            existing_loggers := kwargs.get("logger")
        ) is not False and config.trainer.auto_set_loggers:
            if int(config.trainer.fast_dev_run) > 0:
                log.critical("Disabling loggers because fast_dev_run is enabled.")
            else:
                loggers = config.trainer.logging.construct_loggers(config)
                if existing_loggers is not None and not isinstance(
                    existing_loggers, bool
                ):
                    if not isinstance(existing_loggers, Sequence):
                        existing_loggers = [existing_loggers]
                    loggers.extend(existing_loggers)

                kwargs["logger"] = loggers

        if kwargs.get("num_nodes") == "auto":
            # when num_nodes is auto, we need to detect the number of nodes
            # when on slurm, this would be the number of SLURM nodes allocated
            if SLURMEnvironment.detect():
                from ll.submitit import JobEnvironment

                job = JobEnvironment()
                if not job.activated():
                    raise ValueError(
                        "SLURMEnvironment detected through PL but not submitit. This is a bug."
                    )

                kwargs["num_nodes"] = job.num_nodes
                log.critical(
                    f"Setting num_nodes to {job.num_nodes} (detected through submitit)."
                )
            # otherweise, we assume 1 node
            else:
                kwargs["num_nodes"] = 1
                log.critical("Setting num_nodes to 1 (no SLURM detected).")

        if config.trainer.default_root_dir:
            kwargs["default_root_dir"] = str(config.trainer.default_root_dir)

        # Update the kwargs with the additional trainer kwargs
        kwargs.update(config.trainer.additional_trainer_kwargs)
        kwargs.update(config.trainer.lightning_kwargs)

        # Set the callbacks
        callbacks = kwargs.get("callbacks", [])
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        callbacks.extend(cls.ll_default_callbacks(config))
        kwargs["callbacks"] = callbacks

        return kwargs

    @override
    @copy_method_with_param(
        LightningTrainer.__init__,
        param_type=BaseConfig,
        return_type=NoneType,
    )
    def __init__(self, config: BaseConfig, *args, **kwargs):
        self._ll_config = config
        kwargs = self._update_kwargs(config, kwargs)
        log.critical(f"LightningTrainer.__init__ with {args=} and {kwargs=}.")
        super().__init__(*args, **kwargs)

        if config.trainer.auto_add_trainer_finalizer:
            type(self)._finalizers.append(self.finalize)

        # Print out the log dir, so that we can easily find it in the logs.
        if log_dir := self.log_dir:
            log_dir = str(Path(log_dir).resolve())
        log.critical(f"LightningTrainer log directory: {self.log_dir}.")

    @override
    def _run(
        self, model: LightningModule, ckpt_path: str | Path | None = None
    ) -> _EVALUATE_OUTPUT | _PREDICT_OUTPUT | None:
        """
        Lightning doesn't support gradient clipping with manual optimization.
        We patch the `Trainer._run` method to throw if gradient clipping is enabled
        and `model.automatic_optimization` is False.
        """
        if not model.automatic_optimization and (
            self.gradient_clip_val is not None
            or self.gradient_clip_algorithm is not None
        ):
            raise ValueError(
                "Gradient clipping is not supported with manual optimization. "
                f"Please set {model.__class__.__name__}.automatic_optimization to True "
                "or disable automatic gradient clipping. "
                "If you want to use gradient clipping with manual optimization, you can "
                "set `config.trainer.automatic_gradient_clip=False` and "
                "use the values in `config.trainer.gradient_clip_val` and `config.trainer.gradient_clip_algorithm`."
            )

        return super()._run(model, ckpt_path)

    def _resolve_ckpt_path_get_valid_path(
        self,
        ckpt_path: str | Path | None,
        config: CheckpointLoadingConfig,
    ):
        if ckpt_path is not None:
            return ckpt_path

        if (candidate_path := config.path) is not None:
            return candidate_path

        return None

    def _resolve_ckpt_path(self, ckpt_path: str | Path | None):
        config = self._ll_config.trainer.checkpoint_loading

        # First, let's just try to get a non-None path.
        ckpt_path = self._resolve_ckpt_path_get_valid_path(ckpt_path, config)

        # Now, let's do some resolutions.
        # If the `load_on_init_only` is set, then we only load the checkpoint on initialization.
        if config.load_on_init_only:
            # Here, we should check to see if this checkpoint has already been loaded by a previous process.
            # If it has, then we should just return `None`.
            ckpt_path_value_str = str(ckpt_path)
            ckpt_path_value_hash = hashlib.md5(ckpt_path_value_str.encode()).hexdigest()

            # See if we have a file with the same hash in the log directory.
            if (log_dir := self.log_dir) is None:
                log.warning(
                    "The `log_dir` is not set. Skipping `load_on_init_only` resolution."
                )
                return ckpt_path

            marker_path = Path(log_dir) / f".loaded_ckpt_{ckpt_path_value_hash}"
            if marker_path.exists():
                log.critical(
                    f"Checkpoint {ckpt_path_value_str} has already been loaded. Skipping `ckpt_path` argument."
                )
                return None

            # Synchronize all processes to prevent a scenario where rank 0 makes the file and rank 1 reads it.
            self.strategy.barrier()

            # Otherwise, we create the file to indicate that the checkpoint has been loaded.
            # However, we should only do this on the main process.
            if self.global_rank == 0:
                marker_path.touch(exist_ok=True)

        return ckpt_path

    @override
    def fit(
        self,
        model: LightningModule,
        train_dataloaders: Any | LightningDataModule | None = None,
        val_dataloaders: Any | None = None,
        datamodule: LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
    ) -> None:
        return super().fit(
            model,
            train_dataloaders,
            val_dataloaders,
            datamodule,
            self._resolve_ckpt_path(ckpt_path),
        )

    @override
    def validate(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> list[Mapping[str, float]]:
        return super().validate(
            model,
            dataloaders,
            self._resolve_ckpt_path(ckpt_path),
            verbose,
            datamodule,
        )

    @override
    def test(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> list[Mapping[str, float]]:
        return super().test(
            model,
            dataloaders,
            self._resolve_ckpt_path(ckpt_path),
            verbose,
            datamodule,
        )

    @override
    def predict(
        self,
        model: LightningModule | None = None,
        dataloaders: Any | LightningDataModule | None = None,
        datamodule: LightningDataModule | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | Path | None = None,
    ) -> list[Any] | None:
        return super().predict(
            model,
            dataloaders,
            datamodule,
            return_predictions,
            self._resolve_ckpt_path(ckpt_path),
        )
