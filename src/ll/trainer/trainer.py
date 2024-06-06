import contextlib
import logging
import os
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import torch
import yaml
from lightning.fabric.plugins.environments.lsf import LSFEnvironment
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import Logger
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT
from typing_extensions import Unpack, assert_never, override

from ..actsave import ActSave, ActSaveCallback
from ..log import init_python_logging
from ..model.config import (
    AcceleratorConfigProtocol,
    BaseConfig,
    BaseProfilerConfig,
    CheckpointLoadingConfig,
    LightningTrainerKwargs,
    StrategyConfigProtocol,
)
from ..util import seed
from ..util.environment import set_additional_env_vars

log = logging.getLogger(__name__)


@runtime_checkable
class _FinalizableLogger(Protocol):
    def finish(self) -> Any: ...


def _finalize_loggers(loggers: Sequence[Logger]):
    for logger in loggers:
        if not isinstance(logger, _FinalizableLogger):
            continue
        logger.finish()


def _stdio_log_dir(root_config: BaseConfig):
    if (config := root_config.runner.save_output) is None or not config.enabled:
        return None

    if not (dirpath := config.dirpath):
        dirpath = root_config.directory.resolve_subdirectory(root_config.id, "stdio")

    dirpath = Path(dirpath).resolve()

    # Make sure that the directory exists
    dirpath.mkdir(parents=True, exist_ok=True)

    return dirpath


class Trainer(LightningTrainer):
    _finalizers: list[Callable[[], None]] = []

    def finalize(self):
        """
        Call this method to clean up after training.
        """
        _finalize_loggers(self.loggers)

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

        return init_python_logging(
            lovely_tensors=config.lovely_tensors,
            lovely_numpy=config.lovely_numpy,
            rich=config.rich,
            log_level=config.log_level,
            log_save_dir=_stdio_log_dir(root_config),
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
        # Get the directory path
        if (dirpath := _stdio_log_dir(root_config)) is None:
            yield
            return

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

            # Dump the configuration to the log
            if config.runner.dump_run_information:
                dump_dir = (
                    config.directory.resolve_subdirectory(config.id, "stdio") / "dump"
                )

                # Create a different directory for each rank.
                # Easy way for now: Add a random subdir.
                dump_dir = dump_dir / f"rank_{str(uuid.uuid4())}"
                dump_dir.mkdir(parents=True, exist_ok=True)

                # First, dump the full config
                full_config_path = dump_dir / "config.yaml"
                config_dict = config.model_dump(mode="json")
                with full_config_path.open("w") as file:
                    yaml.dump(config_dict, file)

                # Dump all environment variables
                env_vars_path = dump_dir / "env.yaml"
                env_vars = dict(os.environ)
                with env_vars_path.open("w") as file:
                    yaml.dump(env_vars, file)

                # Dump the output of `nvidia-smi` to a file (if available)
                nvidia_smi_path = dump_dir / "nvidia_smi_output.log"
                try:
                    with nvidia_smi_path.open("w") as file:
                        subprocess.run(
                            ["nvidia-smi"], stdout=file, stderr=subprocess.PIPE
                        )
                except FileNotFoundError:
                    log.warning("Failed to run `nvidia-smi`.")

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
        if config.trainer.actsave:
            yield ActSaveCallback()

        if config.trainer.early_stopping is not None:
            yield config.trainer.early_stopping.construct_callback(config)

        if config.trainer.checkpoint_saving.should_save_checkpoints(config):
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
    def _update_kwargs(
        cls,
        config: BaseConfig,
        kwargs_ctor: LightningTrainerKwargs,
    ):
        kwargs: LightningTrainerKwargs = {
            "deterministic": config.trainer.reproducibility.deterministic,
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
            "num_sanity_val_steps": config.trainer.num_sanity_val_steps,
            "log_every_n_steps": config.trainer.log_every_n_steps,
            "inference_mode": config.trainer.inference_mode,
            "callbacks": [],
            "plugins": [],
            "logger": [],
            # Moved to `lightning_kwargs`:
            # "enable_checkpointing": config.trainer.enable_checkpointing,
            # "accelerator": config.trainer.accelerator,
            # "strategy": config.trainer.strategy,
            # "num_nodes": config.trainer.num_nodes,
            # "precision": config.trainer.precision,
            # "logger": config.trainer.logging.enabled,
            # "log_every_n_steps": config.trainer.log_every_n_steps,
            # "enable_progress_bar": config.trainer.enable_progress_bar,
            # "enable_model_summary": config.trainer.enable_model_summary,
            # "accumulate_grad_batches": config.trainer.accumulate_grad_batches,
            # "benchmark": config.trainer.benchmark,
            # "use_distributed_sampler": config.trainer.use_distributed_sampler,
            # "detect_anomaly": config.trainer.detect_anomaly,
            # "barebones": config.trainer.barebones,
            # "plugins": config.trainer.plugins,
            # "sync_batchnorm": config.trainer.sync_batchnorm,
            # "reload_dataloaders_every_n_epochs": config.trainer.reload_dataloaders_every_n_epochs,
        }

        def _update_key(key: str, new_value: Any):
            # First, check to see if the key is already in the kwargs.
            if key not in kwargs:
                kwargs[key] = new_value
                return

            # If the key is already in the kwargs, then we check the type:
            # - If the type is a sequence, then we extend the sequence.
            # - Otherwise, we just update the value but warn the user.

            match existing_value := kwargs[key]:
                case Sequence() as existing_value:
                    # Make sure value is a sequence too
                    if not isinstance(new_value, Sequence):
                        new_value = [new_value]
                    kwargs[key] = [*existing_value, *new_value]
                case _:
                    log.warning(
                        f"Trainer.__init__: Overwriting existing value {existing_value=} with {new_value=} for key {key=}."
                    )
                    kwargs[key] = new_value

        def _update_kwargs(**update: Unpack[LightningTrainerKwargs]):
            for key, value in update.items():
                _update_key(key, value)

        # Set `default_root_dir` if `auto_set_default_root_dir` is enabled.
        if config.trainer.auto_set_default_root_dir:
            if kwargs.get("default_root_dir"):
                raise ValueError(
                    "You have set `config.trainer.default_root_dir`. "
                    "But we are trying to set it automatically. "
                    "Please use `config.directory.base` rather than `config.trainer.default_root_dir`. "
                    "If you want to set it manually, please set `config.trainer.auto_set_default_root_dir=False`."
                )

            _update_kwargs(
                default_root_dir=config.directory.resolve_run_root_directory(config.id)
            )

        if (devices_input := config.trainer.devices) is not None:
            match devices_input:
                case "all":
                    devices = -1
                case "auto":
                    devices = "auto"
                case Sequence():
                    devices = list(devices_input)
                case _:
                    raise ValueError(f"Invalid value for devices={devices_input}.")

            _update_kwargs(devices=devices)

        if (
            use_distributed_sampler := config.trainer.use_distributed_sampler
        ) is not None:
            _update_kwargs(use_distributed_sampler=use_distributed_sampler)

        if (accelerator := config.trainer.accelerator) is not None:
            if isinstance(accelerator, AcceleratorConfigProtocol):
                accelerator = accelerator.construct_accelerator()
            _update_kwargs(accelerator=accelerator)

        if (strategy := config.trainer.strategy) is not None:
            if isinstance(strategy, StrategyConfigProtocol):
                strategy = strategy.construct_strategy()
            _update_kwargs(strategy=strategy)

        if (precision := config.trainer.precision) is not None:
            resolved_precision: _PRECISION_INPUT
            match precision:
                case "64-true" | "32-true" | "bf16-mixed":
                    resolved_precision = precision
                case "fp16-mixed":
                    resolved_precision = "16-mixed"
                case "16-mixed-auto":
                    resolved_precision = (
                        "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
                    )
                    log.critical(
                        f"Auto-resolving {precision=} to {resolved_precision=}."
                    )
                case _:
                    assert_never(precision)

            _update_kwargs(precision=resolved_precision)

        if (detect_anomaly := config.trainer.detect_anomaly) is not None:
            _update_kwargs(detect_anomaly=detect_anomaly)

        if (
            grad_clip_config := config.trainer.optimizer.gradient_clipping
        ) is not None and grad_clip_config.enabled:
            # kwargs["gradient_clip_algorithm"] = grad_clip_config.algorithm
            # kwargs["gradient_clip_val"] = grad_clip_config.value
            _update_kwargs(
                gradient_clip_algorithm=grad_clip_config.algorithm,
                gradient_clip_val=grad_clip_config.value,
            )

        if profiler := config.trainer.profiler:
            # If the profiler is an ProfilerConfig instance, then we instantiate it.
            if isinstance(profiler, BaseProfilerConfig):
                profiler = profiler.construct_profiler(config)
                # Make sure that the profiler is an instance of `Profiler`.
                if not isinstance(profiler, Profiler):
                    raise ValueError(f"{profiler=} is not an instance of `{Profiler}`.")

            # Otherwise, if the profiler is a string (e.g., "simpe", "advanced", "pytorch"),
            #   then we just pass it through.
            # kwargs["profiler"] = profiler
            _update_kwargs(profiler=profiler)

        if callbacks := config.trainer.callbacks:
            _update_kwargs(
                callbacks=[callback.construct_callback() for callback in callbacks]
            )

        # Additional callbacks from the logging config
        if callbacks := config.trainer.logging.construct_callbacks():
            _update_kwargs(callbacks=callbacks)

        if plugin_configs := config.trainer.plugins:
            _update_kwargs(
                plugins=[
                    plugin_config.construct_plugin() for plugin_config in plugin_configs
                ]
            )

        if not config.trainer.logging.enabled:
            log.critical(f"Disabling logger because {config.trainer.logging.enabled=}.")
            kwargs["logger"] = False
        else:
            _update_kwargs(logger=config.trainer.logging.construct_loggers(config))

        if config.trainer.auto_determine_num_nodes:
            # When num_nodes is auto, we need to detect the number of nodes.
            if SLURMEnvironment.detect():
                if (num_nodes := os.environ.get("SLURM_NNODES")) is not None:
                    num_nodes = int(num_nodes)
                    log.critical(f"SLURM detected with {num_nodes=}.")
                    _update_kwargs(num_nodes=num_nodes)
                else:
                    log.critical(
                        "SLURM detected, but SLURM_NNODES not found. "
                        "We'll continue without setting num_nodes, but this may cause issues."
                    )

            elif LSFEnvironment.detect():
                num_nodes = LSFEnvironment().world_size()
                log.critical(f"LSF detected with {num_nodes=}.")
                _update_kwargs(num_nodes=num_nodes)

        # Update the kwargs with the additional trainer kwargs
        _update_kwargs(**cast(Any, config.trainer.additional_trainer_kwargs))
        _update_kwargs(**config.trainer.lightning_kwargs)
        _update_kwargs(**kwargs_ctor)

        # Set the callbacks
        _update_kwargs(callbacks=[*cls.ll_default_callbacks(config)])

        return kwargs

    @override
    def __init__(
        self,
        config: BaseConfig,
        /,
        **kwargs: Unpack[LightningTrainerKwargs],
    ):
        self._ll_config = config
        kwargs = self._update_kwargs(config, kwargs)
        log.critical(f"LightningTrainer.__init__ with {kwargs=}.")
        super().__init__(**kwargs)

        if config.trainer.auto_add_trainer_finalizer:
            type(self)._finalizers.append(self.finalize)

        # Print out the log dir, so that we can easily find it in the logs.
        if log_dir := self.log_dir:
            log_dir = str(Path(log_dir).resolve())
        log.critical(f"LightningTrainer log directory: {self.log_dir}.")

    @contextlib.contextmanager
    def _actsave_context(self, model: LightningModule):
        hparams = cast(BaseConfig, model.hparams)
        if not (actsave_config := hparams.trainer.actsave):
            yield
            return

        # Enter actsave context
        with ActSave.enabled(actsave_config.resolve_save_dir(hparams)):
            yield

    @override
    def _run(
        self, model: LightningModule, ckpt_path: str | Path | None = None
    ) -> _EVALUATE_OUTPUT | _PREDICT_OUTPUT | None:
        """
        Two things done here:
            1. Lightning doesn't support gradient clipping with manual optimization.
            We patch the `Trainer._run` method to throw if gradient clipping is enabled
            and `model.automatic_optimization` is False.

            2. We actually set up actsave here.
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

        with self._actsave_context(model):
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
