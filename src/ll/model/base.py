import getpass
import inspect
import os
import platform
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, MutableMapping
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Generic, cast

import torch
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import Self, TypeVar, deprecated, override

from .config import (
    BaseConfig,
    EnvironmentClassInformationConfig,
    EnvironmentLinuxEnvironmentConfig,
    EnvironmentLSFInformationConfig,
    EnvironmentSLURMInformationConfig,
)
from .modules.callback import CallbackModuleMixin, CallbackRegistrarModuleMixin
from .modules.debug import DebugModuleMixin
from .modules.distributed import DistributedMixin
from .modules.logger import LoggerLightningModuleMixin, LoggerModuleMixin
from .modules.profiler import ProfilerMixin
from .modules.rlp_sanity_checks import RLPSanityCheckModuleMixin
from .modules.shared_parameters import SharedParametersModuleMixin

log = getLogger(__name__)

THparams = TypeVar("THparams", bound=BaseConfig, infer_variance=True)


class Base(DebugModuleMixin, Generic[THparams]):
    @deprecated("Use `ll.nn.MLP` instead.")
    def mlp(self, *args, **kwargs):
        from ..nn.mlp import MLP

        return MLP(*args, **kwargs)

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


T = TypeVar("T")


def _psutil():
    import psutil

    return psutil


def _try_get(fn: Callable[[], T]) -> T | None:
    try:
        return fn()
    except Exception as e:
        log.warning(f"Failed to get value: {e}")
        return None


class LightningModuleBase(  # pyright: ignore[reportIncompatibleMethodOverride]
    ProfilerMixin,
    RLPSanityCheckModuleMixin,
    LoggerLightningModuleMixin,
    SharedParametersModuleMixin,
    DistributedMixin,
    CallbackModuleMixin,
    Base[THparams],
    LightningModule,
    ABC,
    Generic[THparams],
):
    # Our own custom __repr__ method.
    # Torch's __repr__ method is too verbose and doesn't provide any useful information.
    @override
    def __repr__(self):
        parts: list[str] = []
        parts.append(f"config={self.hparams.concise_repr()}")
        parts.append(f"device={self.device}")
        if self.debug:
            parts.append("debug=True")

        parts_str = ", ".join(parts)
        return f"{self.__class__.__name__}({parts_str})"

    @classmethod
    def _validate_class_for_ckpt_loading(cls):
        # Make sure that the `__init__` method takes a single argument, `hparams`.
        if (init_fn := getattr(cls, "__init__", None)) is None:
            return

        if not inspect.isfunction(init_fn):
            raise TypeError(f"__init__ must be a function: {init_fn}")

        parameters = dict(inspect.signature(init_fn).parameters)
        # Remove the "self" parameter.
        _ = parameters.pop("self", None)
        if len(parameters) != 1:
            raise TypeError(
                f"__init__ must take a single argument, got {len(parameters)}: {init_fn}"
            )

        if "hparams" not in parameters:
            raise TypeError(
                f"__init__'s argument must be named 'hparams', got {parameters}"
            )

    hparams: THparams  # pyright: ignore[reportIncompatibleMethodOverride]
    hparams_initial: THparams  # pyright: ignore[reportIncompatibleMethodOverride]

    @classmethod
    @abstractmethod
    def config_cls(cls) -> type[THparams]: ...

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: _PATH | IO,
        hparams: THparams | MutableMapping[str, Any] | None = None,
        map_location: _MAP_LOCATION_TYPE = None,
        strict: bool = True,
    ) -> Self:
        if strict:
            cls._validate_class_for_ckpt_loading()

        kwargs: dict[str, Any] = {}
        if hparams is not None:
            kwargs["hparams"] = hparams

        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=None,
            strict=strict,
            **kwargs,
        )

    @classmethod
    def _update_environment(cls, hparams: THparams):
        hparams.environment.cwd = Path(os.getcwd())
        hparams.environment.python_executable = Path(sys.executable)
        hparams.environment.python_path = [Path(path) for path in sys.path]
        hparams.environment.python_version = sys.version
        hparams.environment.config = _cls_info(cls.config_cls())
        hparams.environment.model = _cls_info(cls)
        hparams.environment.slurm = (
            EnvironmentSLURMInformationConfig.from_current_environment()
        )
        hparams.environment.lsf = (
            EnvironmentLSFInformationConfig.from_current_environment()
        )
        hparams.environment.base_dir = hparams.directory.resolve_run_root_directory(
            hparams.id
        )
        hparams.environment.log_dir = hparams.directory.resolve_subdirectory(
            hparams.id, "log"
        )
        hparams.environment.checkpoint_dir = hparams.directory.resolve_subdirectory(
            hparams.id, "checkpoint"
        )
        hparams.environment.stdio_dir = hparams.directory.resolve_subdirectory(
            hparams.id, "stdio"
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
            user=_try_get(lambda: getpass.getuser()),
            hostname=_try_get(lambda: platform.node()),
            system=_try_get(lambda: platform.system()),
            release=_try_get(lambda: platform.release()),
            version=_try_get(lambda: platform.version()),
            machine=_try_get(lambda: platform.machine()),
            processor=_try_get(lambda: platform.processor()),
            cpu_count=_try_get(lambda: os.cpu_count()),
            memory=_try_get(lambda: _psutil().virtual_memory().total),
            uptime=_try_get(lambda: timedelta(seconds=_psutil().boot_time())),
            boot_time=_try_get(lambda: _psutil().boot_time()),
            load_avg=_try_get(lambda: os.getloadavg()),
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
            hparams = self.config_cls().model_validate(hparams)
        self._update_environment(hparams)
        hparams = self.pre_init_update_hparams(hparams)
        super().__init__(hparams)

        self.save_hyperparameters(hparams)

        self.register_callback(lambda: DebugFlagCallback())

    def zero_loss(self):
        """
        Returns a loss tensor with the value 0.
        It multiples each weight by 0 and returns the sum, so we don't run into issues with ununsed parameters in DDP.
        """
        loss = sum((0.0 * v).sum() for v in self.parameters() if v.requires_grad)
        loss = cast(torch.Tensor, loss)
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

    if TYPE_CHECKING:

        @override
        def training_step(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            batch: Any,
            batch_idx: int,
        ) -> Any:
            r"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
            logger.

            Args:
                batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
                batch_idx: The index of this batch.
                dataloader_idx: The index of the dataloader that produced this batch.
                    (only if multiple dataloaders used)

            Return:
                - :class:`~torch.Tensor` - The loss tensor
                - ``dict`` - A dictionary which can include any keys, but must include the key ``'loss'`` in the case of
                automatic optimization.
                - ``None`` - In automatic optimization, this will skip to the next batch (but is not supported for
                multi-GPU, TPU, or DeepSpeed). For manual optimization, this has no special meaning, as returning
                the loss is not required.

            In this step you'd normally do the forward pass and calculate the loss for a batch.
            You can also do fancier things like multiple forward passes or something model specific.

            Example::

                def training_step(self, batch, batch_idx):
                    x, y, z = batch
                    out = self.encoder(x)
                    loss = self.loss(out, x)
                    return loss

            To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:

            .. code-block:: python

                def __init__(self):
                    super().__init__()
                    self.automatic_optimization = False


                # Multiple optimizers (e.g.: GANs)
                def training_step(self, batch, batch_idx):
                    opt1, opt2 = self.optimizers()

                    # do training_step with encoder
                    ...
                    opt1.step()
                    # do training_step with decoder
                    ...
                    opt2.step()

            Note:
                When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
                normalized by ``accumulate_grad_batches`` internally.

            """
            raise NotImplementedError

        @override
        def validation_step(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            batch: Any,
            batch_idx: int,
        ) -> STEP_OUTPUT:
            r"""Operates on a single batch of data from the validation set. In this step you'd might generate examples or
            calculate anything of interest like accuracy.

            Args:
                batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
                batch_idx: The index of this batch.
                dataloader_idx: The index of the dataloader that produced this batch.
                    (only if multiple dataloaders used)

            Return:
                - :class:`~torch.Tensor` - The loss tensor
                - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
                - ``None`` - Skip to the next batch.

            .. code-block:: python

                # if you have one val dataloader:
                def validation_step(self, batch, batch_idx): ...


                # if you have multiple val dataloaders:
                def validation_step(self, batch, batch_idx, dataloader_idx=0): ...

            Examples::

                # CASE 1: A single validation dataset
                def validation_step(self, batch, batch_idx):
                    x, y = batch

                    # implement your own
                    out = self(x)
                    loss = self.loss(out, y)

                    # log 6 example images
                    # or generated text... or whatever
                    sample_imgs = x[:6]
                    grid = torchvision.utils.make_grid(sample_imgs)
                    self.logger.experiment.add_image('example_images', grid, 0)

                    # calculate acc
                    labels_hat = torch.argmax(out, dim=1)
                    val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                    # log the outputs!
                    self.log_dict({'val_loss': loss, 'val_acc': val_acc})

            If you pass in multiple val dataloaders, :meth:`validation_step` will have an additional argument. We recommend
            setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

            .. code-block:: python

                # CASE 2: multiple validation dataloaders
                def validation_step(self, batch, batch_idx, dataloader_idx=0):
                    # dataloader_idx tells you which dataset this is.
                    ...

            Note:
                If you don't need to validate you don't need to implement this method.

            Note:
                When the :meth:`validation_step` is called, the model has been put in eval mode
                and PyTorch gradients have been disabled. At the end of validation,
                the model goes back to training mode and gradients are enabled.

            """
            raise NotImplementedError

        @override
        def test_step(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            batch: Any,
            batch_idx: int,
        ) -> STEP_OUTPUT:
            r"""Operates on a single batch of data from the test set. In this step you'd normally generate examples or
            calculate anything of interest such as accuracy.

            Args:
                batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
                batch_idx: The index of this batch.
                dataloader_idx: The index of the dataloader that produced this batch.
                    (only if multiple dataloaders used)

            Return:
                - :class:`~torch.Tensor` - The loss tensor
                - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
                - ``None`` - Skip to the next batch.

            .. code-block:: python

                # if you have one test dataloader:
                def test_step(self, batch, batch_idx): ...


                # if you have multiple test dataloaders:
                def test_step(self, batch, batch_idx, dataloader_idx=0): ...

            Examples::

                # CASE 1: A single test dataset
                def test_step(self, batch, batch_idx):
                    x, y = batch

                    # implement your own
                    out = self(x)
                    loss = self.loss(out, y)

                    # log 6 example images
                    # or generated text... or whatever
                    sample_imgs = x[:6]
                    grid = torchvision.utils.make_grid(sample_imgs)
                    self.logger.experiment.add_image('example_images', grid, 0)

                    # calculate acc
                    labels_hat = torch.argmax(out, dim=1)
                    test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                    # log the outputs!
                    self.log_dict({'test_loss': loss, 'test_acc': test_acc})

            If you pass in multiple test dataloaders, :meth:`test_step` will have an additional argument. We recommend
            setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

            .. code-block:: python

                # CASE 2: multiple test dataloaders
                def test_step(self, batch, batch_idx, dataloader_idx=0):
                    # dataloader_idx tells you which dataset this is.
                    ...

            Note:
                If you don't need to test you don't need to implement this method.

            Note:
                When the :meth:`test_step` is called, the model has been put in eval mode and
                PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
                to training mode and gradients are enabled.

            """
            raise NotImplementedError

    @override
    def predict_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        batch: Any,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """Step function called during :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`. By default, it calls
        :meth:`~lightning.pytorch.core.LightningModule.forward`. Override to add any processing logic.

        The :meth:`~lightning.pytorch.core.LightningModule.predict_step` is used
        to scale inference on multi-devices.

        To prevent an OOM error, it is possible to use :class:`~lightning.pytorch.callbacks.BasePredictionWriter`
        callback to write the predictions to disk or database after each batch or on epoch end.

        The :class:`~lightning.pytorch.callbacks.BasePredictionWriter` should be used while using a spawn
        based accelerator. This happens for ``Trainer(strategy="ddp_spawn")``
        or training on 8 TPU cores with ``Trainer(accelerator="tpu", devices=8)`` as predictions won't be returned.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)

        Return:
            Predicted output (optional).

        Example ::

            class MyModel(LightningModule):

                def predict_step(self, batch, batch_idx, dataloader_idx=0):
                    return self(batch)

            dm = ...
            model = MyModel()
            trainer = Trainer(accelerator="gpu", devices=2)
            predictions = trainer.predict(model, dm)

        """
        prediction = self(batch)
        return {
            "prediction": prediction,
            "batch": batch,
            "batch_idx": batch_idx,
        }


class LightningDataModuleBase(
    LoggerModuleMixin,
    CallbackRegistrarModuleMixin,
    Base[THparams],
    LightningDataModule,
    ABC,
    Generic[THparams],
):
    hparams: THparams  # pyright: ignore[reportIncompatibleMethodOverride]
    hparams_initial: THparams  # pyright: ignore[reportIncompatibleMethodOverride]

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
