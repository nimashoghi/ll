import contextlib
import fnmatch
from collections import defaultdict
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning_utilities.core.apply_func import apply_to_collection
from typing_extensions import override

log = getLogger(__name__)


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
    @contextlib.contextmanager
    def enabled(
        self,
        dump: Path | None = None,
        dump_filters: list[str] | None = None,
    ):
        prev = self._enabled
        self.initialize(enabled=True)
        context = _ActivationContext(self)
        try:
            yield context
        finally:
            if dump:
                self.dump(dump, filters=dump_filters)
            context.finalize()
            self.initialize(enabled=prev)

    def initialize(self, *, enabled: bool = True, clear: bool = True):
        self._enabled = enabled

        if clear:
            self.clear()

    @override
    def __init__(self, *, enabled: bool = False):
        super().__init__(list)

        self._enabled = enabled
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
    def _clone(activation: torch.Tensor):
        if not isinstance(activation, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(activation)}")
        return activation.cpu().detach().clone()

    def save(self, acts: dict[str, Any] | None = None, /, **kwargs: Any):
        if not self._enabled or torch.jit.is_scripting():
            return

        if acts:
            kwargs.update(acts)
        for name, activation in kwargs.items():
            name = ".".join(self.prefixes + [name])
            self[name].append(
                apply_to_collection(activation, torch.Tensor, self._clone)
            )

    __call__ = save

    def dump(
        self,
        root_dir: Path,
        save_all: bool = True,
        save_each: bool = False,
        filters: list[str] | None = None,
    ):
        if not self._enabled or torch.jit.is_scripting():
            return

        root_dir.mkdir(parents=True, exist_ok=True)
        # First, we save each activation to a separate file
        for name, activations in self.items():
            # Make sure name matches at least one filter
            if filters and not any(fnmatch.fnmatch(name, f) for f in filters):
                continue

            base_path = root_dir / f"{name}"
            base_path.mkdir(parents=True, exist_ok=True)

            if save_all:
                all_path = base_path / "all.pt"
                torch.save(activations, all_path)

            if save_each:
                for i, activation in enumerate(activations):
                    path = base_path / f"{i}.pt"
                    torch.save(activation, path)
            log.debug(f"Saved activations to {base_path}")

        # Then, we just dump the entire dict to a file
        all_path = root_dir / "all.pt"
        torch.save(dict(self), all_path)


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
