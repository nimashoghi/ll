import contextlib
import fnmatch
import string
import tempfile
import uuid
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning_utilities.core.apply_func import apply_to_collection
from typing_extensions import ParamSpec, override

log = getLogger(__name__)

Value = Union[int, float, complex, bool, str, np.ndarray, torch.Tensor]
ValueOrLambda = Union[Value, Callable[..., Value]]

Transform = Callable[[str, Any], dict[str, Any] | None]


def _make_id(length: int = 4) -> str:
    alphabet = list(string.ascii_lowercase + string.digits)
    id = "".join(np.random.choice(alphabet) for _ in range(length))
    return id


P = ParamSpec("P")


def _ensure_supported():
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError("Only single GPU is supported at the moment")
    except ImportError:
        pass


def _ignore_if_scripting(fn: Callable[P, None]) -> Callable[P, None]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        if torch.jit.is_scripting():
            return

        _ensure_supported()
        fn(*args, **kwargs)

    return wrapper


class ActivationSaver:
    def __init__(
        self,
        save_dir: Path,
        prefixes_fn: Callable[[], list[str]],
        *,
        filters: list[str] | None = None,
        transforms: list[tuple[str, Transform]] | None = None,
    ):
        self._save_dir = save_dir / _make_id()
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._prefixes_fn = prefixes_fn
        self._filters = filters
        self._transforms = transforms

    @staticmethod
    def _to_numpy(activation: Value) -> np.ndarray:
        if isinstance(activation, np.ndarray):
            return activation
        if isinstance(activation, torch.Tensor):
            if activation.is_floating_point():
                # NOTE: We need to convert to float32 because [b]float16 is not supported by numpy
                activation = activation.float()
            return activation.detach().cpu().numpy()
        if isinstance(activation, (int, float, complex, str, bool)):
            return np.array(activation)
        return activation

    def _save_activation(self, name: str, activation_or_lambda: ValueOrLambda):
        # Make sure name matches at least one filter if filters are specified
        if self._filters and not any(fnmatch.fnmatch(name, f) for f in self._filters):
            return

        name = ".".join(self._prefixes_fn() + [name])
        activation = activation_or_lambda

        # If we have a lambda, we need to call it
        if callable(activation_or_lambda):
            activation = activation_or_lambda()
        activation = apply_to_collection(activation, torch.Tensor, self._to_numpy)
        activation = self._to_numpy(activation)

        # Save the activation to self._save_dir / name / {id}.npz, where id is an auto-incrementing integer
        path = self._save_dir / name
        path.mkdir(parents=True, exist_ok=True)

        # Get the next id
        np.save(path / f"{len(list(path.glob('*.npy'))):04d}.npy", activation)

    def save(
        self,
        acts: dict[str, ValueOrLambda] | None = None,
        /,
        **kwargs: ValueOrLambda,
    ):
        kwargs.update(acts or {})
        for name, activation in kwargs.items():
            # If we have any transforms, we need to apply them
            if self._transforms:
                # Iterate through transforms and apply them
                for name, transform in self._transforms:
                    # If the transform doesn't match, we skip it
                    if not fnmatch.fnmatch(name, name):
                        continue

                    # Apply the transform
                    transform_out = transform(name, activation)

                    # If the transform returns empty, we skip it
                    if not transform_out:
                        continue

                    # Otherwise, add the transform to the activations
                    for k, v in transform_out.items():
                        self._save_activation(k, v)

            self._save_activation(name, activation)


class ActSaveProvider:
    _saver: ActivationSaver | None = None
    _prefixes: list[str] = []

    def initialize(
        self,
        save_dir: Path | None = None,
        *,
        filters: list[str] | None = None,
        transforms: list[tuple[str, Transform]] | None = None,
    ):
        if self._saver is None:
            if save_dir is None:
                save_dir = Path(tempfile.gettempdir()) / f"actsave-{uuid.uuid4()}"
                log.critical(f"No save_dir specified, using {save_dir=}")
            self._saver = ActivationSaver(
                save_dir,
                lambda: self._prefixes,
                filters=filters,
                transforms=transforms,
            )

    @contextlib.contextmanager
    def enabled(
        self,
        save_dir: Path | None = None,
        *,
        filters: list[str] | None = None,
        transforms: list[tuple[str, Transform]] | None = None,
    ):
        prev = self._saver
        self.initialize(save_dir, filters=filters, transforms=transforms)
        try:
            yield
        finally:
            self._saver = prev

    @override
    def __init__(self):
        super().__init__()

        self._saver = None
        self._prefixes = []

    @contextlib.contextmanager
    def context(self, label: str):
        if torch.jit.is_scripting():
            return

        _ensure_supported()

        log.debug(f"Entering ActSave context {label}")
        self._prefixes.append(label)
        try:
            yield
        finally:
            _ = self._prefixes.pop()

    prefix = context

    @_ignore_if_scripting
    def __call__(
        self,
        acts: dict[str, ValueOrLambda] | None = None,
        /,
        **kwargs: ValueOrLambda,
    ):
        if self._saver is None:
            raise RuntimeError("ActSave is not initialized")

        self._saver.save(acts, **kwargs)

    save = __call__


ActSave = ActSaveProvider()


def _wrap_fn(module: LightningModule, fn_name: str):
    old_step = getattr(module, fn_name).__func__

    @wraps(old_step)
    def new_step(module: LightningModule, batch, batch_idx, *args, **kwargs):
        with ActSave.context(fn_name):
            return old_step(module, batch, batch_idx, *args, **kwargs)

    setattr(module, fn_name, new_step.__get__(module))
    log.info(f"Wrapped {fn_name} for actsave")


def wrap_lightning_module(module: LightningModule):
    log.info(
        "Wrapping training_step/validation_step/test_step/predict_step for actsave"
    )

    _wrap_fn(module, "training_step")
    _wrap_fn(module, "validation_step")
    _wrap_fn(module, "test_step")
    _wrap_fn(module, "predict_step")

    log.info("Wrapped training_step/validation_step/test_step/predict_step for actsave")
