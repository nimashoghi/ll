import inspect
import os
import platform
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, MutableMapping
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import Any, Generic, cast

import psutil
import torch
import torch.nn as nn
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from typing_extensions import TypeVar, deprecated, override

from .. import actsave
from ..nn.mlp import MLP
from ..trainer import Trainer as LLTrainer
from ..util import log_batch_info, skip_batch
from .config import (
    BaseConfig,
    EnvironmentClassInformationConfig,
    EnvironmentLinuxEnvironmentConfig,
    EnvironmentSLURMInformationConfig,
)
from .modules.callback import CallbackModuleMixin, CallbackRegistrarModuleMixin
from .modules.debug import DebugModuleMixin
from .modules.distributed import DistributedMixin
from .modules.finite_checks import FiniteChecksModuleMixin
from .modules.log_dir import LogDirMixin
from .modules.log_epoch import LogEpochMixin
from .modules.logger import LoggerModuleMixin
from .modules.lr_monitor import LRMonitorMixin
from .modules.optimizer import OptimizerModuleMixin
from .modules.parameter_hooks import ParameterHookModuleMixin
from .modules.profiler import ProfilerMixin
from .modules.rlp_sanity_checks import RLPSanityCheckModuleMixin
from .modules.shared_parameters import SharedParametersModuleMixin
from .modules.wandb import WandbWrapperMixin

log = getLogger(__name__)

THparams = TypeVar("THparams", bound=BaseConfig, infer_variance=True)


class Base(DebugModuleMixin, Generic[THparams]):
    @torch.jit.unused
    @deprecated("Use ll.nn.MLP instead")
    def mlp(
        self,
        dims: list[int],
        *,
        activation: Callable[[], nn.Module],
        bias: bool = True,
        no_bias_scalar: bool = True,
        ln: bool | str = False,
        dropout: float | None = None,
        residual: bool = False,
        pre_layers: list[nn.Module] = [],
        post_layers: list[nn.Module] = [],
    ) -> nn.Sequential:
        """
        Constructs a multi-layer perceptron (MLP) with the given dimensions and activation function.

        Args:
            dims (list[int]): List of integers representing the dimensions of the MLP.
            activation (Callable[[], nn.Module]): Activation function to use between layers.
            bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
            no_bias_scalar (bool, optional): Whether to exclude bias terms when the output dimension is 1. Defaults to True.
            ln (bool | str, optional): Whether to apply layer normalization before or after the linear layers. Defaults to False.
            dropout (float | None, optional): Dropout probability to apply between layers. Defaults to None.
            residual (bool, optional): Whether to use residual connections between layers. Defaults to False.
            pre_layers (list[nn.Module], optional): List of layers to insert before the linear layers. Defaults to [].
            post_layers (list[nn.Module], optional): List of layers to insert after the linear layers. Defaults to [].

        Returns:
            nn.Sequential: The constructed MLP.
        """

        return MLP(
            dims,
            activation=activation,
            bias=bias,
            no_bias_scalar=no_bias_scalar,
            ln=ln,
            dropout=dropout,
            residual=residual,
            pre_layers=pre_layers,
            post_layers=post_layers,
        )

    @torch.jit.unused
    @property
    def config(self) -> THparams:
        return self.hparams

    @torch.jit.unused
    @property
    def C(self) -> THparams:
        return self.hparams

    @property
    def debug(self) -> bool:
        if torch.jit.is_scripting():
            return False
        return self.config.debug

    @property
    def dev(self) -> bool:
        if torch.jit.is_scripting():
            return False
        return self.config.debug

    @override
    def __init__(self, hparams: THparams):
        super().__init__()

        if not hasattr(self, "hparams"):
            self.hparams = hparams


class DebugFlagCallback(Callback):
    """
    Sets the debug flag to true in the following circumstances:
    - fast_dev_run is enabled
    - sanity check is running
    """

    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        if not getattr(trainer, "fast_dev_run", False):
            return

        hparams = cast(BaseConfig, pl_module.hparams)
        if not hparams.debug:
            log.critical("Fast dev run detected, setting debug flag to True.")
        hparams.debug = True

    @override
    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule):
        hparams = cast(BaseConfig, pl_module.hparams)
        self._debug = hparams.debug
        if not self._debug:
            log.critical("Enabling debug flag during sanity check routine.")
        hparams.debug = True

    @override
    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule):
        hparams = cast(BaseConfig, pl_module.hparams)
        if not self._debug:
            log.critical("Sanity check routine complete, disabling debug flag.")
        hparams.debug = self._debug


def _slurm_session_info():
    try:
        from ll.submitit import JobEnvironment

        job = JobEnvironment()
        if not job.activated():
            return None

        # return {
        #     "hostname": job.hostname,
        #     "hostnames": job.hostnames,
        #     "job_id": job.job_id,
        #     "raw_job_id": job.raw_job_id,
        #     "array_job_id": job.array_job_id,
        #     "array_task_id": job.array_task_id,
        #     "num_tasks": job.num_tasks,
        #     "num_nodes": job.num_nodes,
        #     "node": job.node,
        #     "global_rank": job.global_rank,
        #     "local_rank": job.local_rank,
        # }
        return EnvironmentSLURMInformationConfig(
            hostname=job.hostname,
            hostnames=list(job.hostnames),
            job_id=job.job_id,
            raw_job_id=job.raw_job_id,
            array_job_id=job.array_job_id,
            array_task_id=job.array_task_id,
            num_tasks=job.num_tasks,
            num_nodes=job.num_nodes,
            node=job.node,
            global_rank=job.global_rank,
            local_rank=job.local_rank,
        )
    except (ImportError, RuntimeError):
        return None


def _cls_info(cls: type):
    name = cls.__name__
    module = cls.__module__
    full_name = f"{cls.__module__}.{cls.__qualname__}"

    file_path = inspect.getfile(cls)
    source_file_path = inspect.getsourcefile(cls)
    return EnvironmentClassInformationConfig(
        name=name,
        module=module,
        full_name=full_name,
        file_path=Path(file_path),
        source_file_path=Path(source_file_path) if source_file_path else None,
    )


class LightningModuleBase(
    ProfilerMixin,
    LogDirMixin,
    WandbWrapperMixin,
    OptimizerModuleMixin,
    RLPSanityCheckModuleMixin,
    LogEpochMixin,
    LoggerModuleMixin,
    LRMonitorMixin,
    FiniteChecksModuleMixin,
    SharedParametersModuleMixin,
    ParameterHookModuleMixin,
    DistributedMixin,
    CallbackModuleMixin,
    Base[THparams],
    LightningModule,
    ABC,
    Generic[THparams],
):
    hparams: THparams
    hparams_initial: THparams

    @classmethod
    @abstractmethod
    def config_cls(cls) -> type[THparams]: ...

    @classmethod
    def _update_environment(cls, hparams: THparams):
        hparams.environment.cwd = Path(os.getcwd())
        hparams.environment.python_executable = Path(sys.executable)
        hparams.environment.python_path = [Path(path) for path in sys.path]
        hparams.environment.python_version = sys.version
        hparams.environment.config = _cls_info(cls.config_cls())
        hparams.environment.model = _cls_info(cls)
        hparams.environment.slurm = _slurm_session_info()
        hparams.environment.log_dir = Path(
            hparams.trainer.default_root_dir
            or LLTrainer.ll_default_root_dir(hparams).absolute()
        )
        hparams.environment.seed = (
            int(seed_str) if (seed_str := os.environ.get("PL_GLOBAL_SEED")) else None
        )
        hparams.environment.seed_workers = (
            bool(int(seed_everything))
            if (seed_everything := os.environ.get("PL_SEED_WORKERS"))
            else None
        )
        hparams.environment.linux = EnvironmentLinuxEnvironmentConfig(
            user=os.getlogin(),
            hostname=platform.node(),
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
            processor=platform.processor(),
            cpu_count=os.cpu_count(),
            memory=psutil.virtual_memory().total,
            uptime=timedelta(seconds=psutil.boot_time()),
            boot_time=psutil.boot_time(),
            load_avg=os.getloadavg(),
        )

    def pre_init_update_hparams_dict(self, hparams: MutableMapping[str, Any]):
        """
        Override this method to update the hparams dictionary before it is used to create the hparams object.
        Mapping-based parameters are passed to the constructor of the hparams object when we're loading the model from a checkpoint.
        """
        return hparams

    def pre_init_update_hparams(self, hparams: THparams):
        """
        Override this method to update the hparams object before it is used to create the hparams_initial object.
        """
        return hparams

    @override
    def __init__(self, hparams: THparams | MutableMapping[str, Any]):
        if not isinstance(hparams, BaseConfig):
            if not isinstance(hparams, MutableMapping):
                raise TypeError(
                    f"hparams must be a BaseConfig or a MutableMapping: {type(hparams)}"
                )

            hparams = self.pre_init_update_hparams_dict(hparams)
            hparams = self.config_cls().from_dict(hparams)
        self._update_environment(hparams)
        hparams = self.pre_init_update_hparams(hparams)
        super().__init__(hparams)

        self.save_hyperparameters(hparams)

        self.register_callback(lambda: DebugFlagCallback())

        actsave.wrap_lightning_module(self)

        if self.config.trainer.log_batch_info_on_error:
            log_batch_info.wrap_lightning_module(self)

        if self.config.trainer.supports_skip_batch_exception:
            skip_batch.wrap_lightning_module(self)

    def zero_loss(self):
        """
        Returns a loss tensor with the value 0.
        It multiples each weight by 0 and returns the sum, so we don't run into issues with ununsed parameters in DDP.
        """
        loss = sum((0.0 * v).sum() for v in self.parameters())
        loss = cast(torch.Tensor, loss)
        return loss

    def skip_batch_training_step(self, *args: Any, **kwargs: Any):
        """
        This function gets called when a `SkipBatch` exception is raised during any point in training.
        If `automatic_optimization` is enabled, it should return a loss tensor that will be used for the backward pass. By default, it returns a zero loss tensor.
        If `automatic_optimization` is disabled, this function needs to be implemented and should handle the backward pass itself.
        """
        if not self.automatic_optimization:
            raise NotImplementedError(
                "To use `SkipBatch` with manual optimization, you must implement `skip_batch_training_step`."
            )

        loss = self.zero_loss()
        return loss

    @property
    def datamodule(self):
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return None

        if not isinstance(datamodule, LightningDataModuleBase):
            raise TypeError(
                f"datamodule must be a LightningDataModuleBase: {type(datamodule)}"
            )

        datamodule = cast(LightningDataModuleBase[THparams], datamodule)
        return datamodule

    # @abstractmethod
    # @override
    # def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    #     ...

    # @abstractmethod
    # @override
    # def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    #     ...


class LightningDataModuleBase(
    LogDirMixin,
    CallbackRegistrarModuleMixin,
    Base[THparams],
    LightningDataModule,
    ABC,
    Generic[THparams],
):
    hparams: THparams
    hparams_initial: THparams

    def pre_init_update_hparams_dict(self, hparams: MutableMapping[str, Any]):
        """
        Override this method to update the hparams dictionary before it is used to create the hparams object.
        Mapping-based parameters are passed to the constructor of the hparams object when we're loading the model from a checkpoint.
        """
        return hparams

    def pre_init_update_hparams(self, hparams: THparams):
        """
        Override this method to update the hparams object before it is used to create the hparams_initial object.
        """
        return hparams

    @classmethod
    def _update_environment(cls, hparams: THparams):
        hparams.environment.data = _cls_info(cls)

    @override
    def __init__(self, hparams: THparams):
        if not isinstance(hparams, BaseConfig):
            if not isinstance(hparams, MutableMapping):
                raise TypeError(
                    f"hparams must be a BaseConfig or a MutableMapping: {type(hparams)}"
                )

            hparams = self.pre_init_update_hparams_dict(hparams)
            hparams = self.config_cls().from_dict(hparams)
        self._update_environment(hparams)
        hparams = self.pre_init_update_hparams(hparams)
        super().__init__(hparams)

        self.save_hyperparameters(hparams)

    @property
    def lightning_module(self):
        if not self.trainer:
            raise ValueError("Trainer has not been set.")

        module = self.trainer.lightning_module
        if not isinstance(module, LightningModuleBase):
            raise ValueError(
                f"Trainer's lightning_module is not a LightningModuleBase: {type(module)}"
            )

        module = cast(LightningModuleBase[THparams], module)
        return module

    @property
    def device(self):
        return self.lightning_module.device
