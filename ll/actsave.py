import contextlib
import fnmatch
from collections import defaultdict
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning_utilities.core.apply_func import apply_to_collection
from typing_extensions import override

log = getLogger(__name__)

Value = Union[int, float, complex, bool, str, np.ndarray, torch.Tensor]
ValueOrLambda = Union[Value, Callable[..., Value]]

Transform = Callable[[str, Any], dict[str, Any] | None]


class _ActivationContext:
    _provider: "ActSaveProvider | None"
    _dict: dict[str, list[Any]] | None

    def __init__(self, provider: "ActSaveProvider"):
        self._provider = provider
        self._dict = None

    def finalize(self):
        if self._provider is None:
            raise RuntimeError("Cannot finalize twice")

        self._dict = dict(self._provider)
        self._provider = None

    def unwrap(self):
        if self._dict is None:
            if self._provider is None:
                raise RuntimeError("Cannot get after finalizing")
            return dict(self._provider)

        return self._dict


class ActSaveProvider(defaultdict[str, list[Any]]):
    @staticmethod
    def load(path: str | Path):
        if isinstance(path, str):
            path = Path(path)

        if path.is_dir():
            return {p.stem: np.load(p, allow_pickle=True) for p in path.glob("*.npz")}

        # Otherwise, it must be an npz file
        if not path.suffix == ".npz" or not path.exists():
            raise ValueError(f"Invalid path {path}")
        return np.load(path, allow_pickle=True)

    @contextlib.contextmanager
    def enabled(
        self,
        *,
        filters: list[str] | None = None,
        transforms: list[tuple[str, Transform]] | None = None,
        dump: Path | None = None,
        dump_filters: list[str] | None = None,
    ):
        prev = self._enabled
        self.initialize(enabled=True, filters=filters, transforms=transforms)
        context = _ActivationContext(self)
        try:
            yield context
        finally:
            if dump:
                self.dump(dump, dump_filters=dump_filters)
            context.finalize()
            self.initialize(enabled=prev)

    def initialize(
        self,
        *,
        enabled: bool = True,
        clear: bool = True,
        filters: list[str] | None = None,
        transforms: list[tuple[str, Transform]] | None = None,
    ):
        self._enabled = enabled
        self._filters = filters
        self._transforms = transforms

        if clear:
            self.clear()

    @override
    def __init__(self, *, enabled: bool = False):
        super().__init__(list)

        self._enabled = enabled
        self._filters: list[str] | None = None
        self._transforms: list[tuple[str, Transform]] | None = None
        self.prefixes: list[str] = []

    @staticmethod
    def _ensure_supported():
        try:
            import torch.distributed as dist

            if dist.is_initialized() and dist.get_world_size() > 1:
                raise RuntimeError("Only single GPU is supported at the moment")
        except ImportError:
            pass

    @contextlib.contextmanager
    def prefix(self, label: str):
        if not self._enabled or torch.jit.is_scripting():
            yield
            return

        self._ensure_supported()
        log.debug(f"Entering ActSave context {label}")
        self.prefixes.append(label)
        try:
            yield
        finally:
            _ = self.prefixes.pop()

    @staticmethod
    def _to_numpy(activation: Value):
        if not isinstance(activation, torch.Tensor):
            return activation

        if activation.is_floating_point():
            # NOTE: We need to convert to float32 because [b]float16 is not supported by numpy
            activation = activation.float()
        return activation.detach().cpu().numpy()

    def _add_activation_(self, name: str, activation: ValueOrLambda):
        # Make sure name matches at least one filter if filters are specified
        if self._filters and not any(fnmatch.fnmatch(name, f) for f in self._filters):
            return

        name = ".".join(self.prefixes + [name])
        # If we have a lambda, we need to call it
        if callable(activation):
            activation = activation()
        self[name].append(apply_to_collection(activation, torch.Tensor, self._to_numpy))

    def save(
        self,
        acts: dict[str, ValueOrLambda] | None = None,
        /,
        **kwargs: ValueOrLambda,
    ):
        if not self._enabled or torch.jit.is_scripting():
            return

        if acts:
            kwargs.update(acts)
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
                        self._add_activation_(k, v)

            self._add_activation_(name, activation)

    __call__ = save

    def dump(self, root_dir: Path, dump_filters: list[str] | None = None):
        if not self._enabled or torch.jit.is_scripting():
            return

        root_dir.mkdir(parents=True, exist_ok=True)
        # First, we save each activation to a separate file
        for name, activations in self.items():
            # Make sure name matches at least one filter
            if dump_filters and not any(fnmatch.fnmatch(name, f) for f in dump_filters):
                continue

            path = root_dir / f"{name}.npz"
            np.savez_compressed(path, *activations, allow_pickle=True)
        log.debug(f"Saved activations to {root_dir}")


ActSave = ActSaveProvider()


def _wrap_fn(module: LightningModule, fn_name: str):
    old_step = getattr(module, fn_name).__func__

    @wraps(old_step)
    def new_step(module: LightningModule, batch, batch_idx, *args, **kwargs):
        with ActSave.prefix(fn_name):
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
