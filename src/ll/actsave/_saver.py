import contextlib
import fnmatch
import tempfile
import uuid
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Generic, TypeAlias, cast, overload

import numpy as np
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from typing_extensions import ParamSpec, TypeVar, override

log = getLogger(__name__)

Value: TypeAlias = int | float | complex | bool | str | np.ndarray | torch.Tensor
ValueOrLambda = Value | Callable[..., Value]


def _to_numpy(activation: Value) -> np.ndarray:
    if isinstance(activation, np.ndarray):
        return activation
    if isinstance(activation, torch.Tensor):
        activation = activation.detach()
        if activation.is_floating_point():
            # NOTE: We need to convert to float32 because [b]float16 is not supported by numpy
            activation = activation.float()
        return activation.cpu().numpy()
    if isinstance(activation, (int, float, complex, str, bool)):
        return np.array(activation)

    return activation


T = TypeVar("T", infer_variance=True)


# A wrapper around weakref.ref that allows for primitive types
# To get around errors like:
# TypeError: cannot create weak reference to 'int' object
class WeakRef(Generic[T]):
    _ref: Callable[[], T] | None

    def __init__(self, obj: T):
        try:
            self._ref = cast(Callable[[], T], weakref.ref(obj))
        except TypeError as e:
            if "cannot create weak reference" not in str(e):
                raise
            self._ref = lambda: obj

    def __call__(self) -> T:
        if self._ref is None:
            raise RuntimeError("WeakRef is deleted")
        return self._ref()

    def delete(self):
        del self._ref
        self._ref = None


@dataclass
class Activation:
    name: str
    ref: WeakRef[ValueOrLambda] | None
    transformed: np.ndarray | None = None

    def __post_init__(self):
        # Update the `name` to replace `/` with `::`
        self.name = self.name.replace("/", "::")

    def __call__(self) -> np.ndarray:
        # If we have a transformed value, we return it
        if self.transformed is not None:
            return self.transformed

        if self.ref is None:
            raise RuntimeError("Activation is deleted")

        # If we have a lambda, we need to call it
        unrwapped_ref = self.ref()
        activation = unrwapped_ref
        if callable(unrwapped_ref):
            activation = unrwapped_ref()
        activation = apply_to_collection(activation, torch.Tensor, _to_numpy)
        activation = _to_numpy(activation)

        # Set the transformed value
        self.transformed = activation

        # Delete the reference
        self.ref.delete()
        del self.ref
        self.ref = None

        return self.transformed

    @classmethod
    def from_value_or_lambda(cls, name: str, value_or_lambda: ValueOrLambda):
        return cls(name, WeakRef(value_or_lambda))

    @classmethod
    def from_dict(cls, d: Mapping[str, ValueOrLambda]):
        return [cls.from_value_or_lambda(k, v) for k, v in d.items()]


Transform = Callable[[Activation], Mapping[str, ValueOrLambda]]


def _ensure_supported():
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError("Only single GPU is supported at the moment")
    except ImportError:
        pass


P = ParamSpec("P")


def _ignore_if_scripting(fn: Callable[P, None]) -> Callable[P, None]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        if torch.jit.is_scripting():
            return

        _ensure_supported()
        fn(*args, **kwargs)

    return wrapper


class _Saver:
    def __init__(
        self,
        save_dir: Path,
        prefixes_fn: Callable[[], list[str]],
        *,
        filters: list[str] | None = None,
        transforms: list[tuple[str, Transform]] | None = None,
    ):
        # Create a directory under `save_dir` by autoincrementing
        # (i.e., every activation save context, we create a new directory)
        # The id = the number of activation subdirectories
        self._id = sum(1 for subdir in save_dir.glob("*") if subdir.is_dir())
        save_dir.mkdir(parents=True, exist_ok=True)

        # Add a .activationbase file to the save_dir to indicate that this is an activation base
        (save_dir / ".activationbase").touch(exist_ok=True)

        self._save_dir = save_dir / f"{self._id:04d}"
        # Make sure `self._save_dir` does not exist and create it
        self._save_dir.mkdir(exist_ok=False)

        self._prefixes_fn = prefixes_fn
        self._filters = filters
        self._transforms = transforms

    def _save_activation(self, activation: Activation):
        # Save the activation to self._save_dir / name / {id}.npz, where id is an auto-incrementing integer
        file_name = ".".join(self._prefixes_fn() + [activation.name])
        path = self._save_dir / file_name
        path.mkdir(exist_ok=True, parents=True)

        # Get the next id and save the activation
        id = len(list(path.glob("*.npy")))
        np.save(path / f"{id:04d}.npy", activation())

    @_ignore_if_scripting
    def save(
        self,
        acts: dict[str, ValueOrLambda] | None = None,
        /,
        **kwargs: ValueOrLambda,
    ):
        kwargs.update(acts or {})

        # Build activations
        activations = Activation.from_dict(kwargs)

        transformed_activations: list[Activation] = []

        for activation in activations:
            # Make sure name matches at least one filter if filters are specified
            if self._filters is None or any(
                fnmatch.fnmatch(activation.name, f) for f in self._filters
            ):
                self._save_activation(activation)

            # If we have any transforms, we need to apply them
            if self._transforms:
                # Iterate through transforms and apply them
                for name, transform in self._transforms:
                    # If the transform doesn't match, we skip it
                    if not fnmatch.fnmatch(activation.name, name):
                        continue

                    # Apply the transform
                    # If the transform returns empty, we skip it
                    if not (transform_out := transform(activation)):
                        continue

                    # Otherwise, add the transform to the activations
                    transformed_activations.extend(Activation.from_dict(transform_out))

        # Now, we save the transformed activations
        for transformed_activation in transformed_activations:
            self._save_activation(transformed_activation)

        del activations
        del transformed_activations


class ActSaveProvider:
    _saver: _Saver | None = None
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
            self._saver = _Saver(
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
            yield
            return

        if self._saver is None:
            yield
            return

        _ensure_supported()

        log.debug(f"Entering ActSave context {label}")
        self._prefixes.append(label)
        try:
            yield
        finally:
            _ = self._prefixes.pop()

    prefix = context

    @overload
    def __call__(
        self,
        acts: dict[str, ValueOrLambda] | None = None,
        /,
        **kwargs: ValueOrLambda,
    ): ...

    @overload
    def __call__(self, acts: Callable[[], dict[str, ValueOrLambda]], /): ...

    def __call__(
        self,
        acts: (
            dict[str, ValueOrLambda] | Callable[[], dict[str, ValueOrLambda]] | None
        ) = None,
        /,
        **kwargs: ValueOrLambda,
    ):
        if torch.jit.is_scripting():
            return

        if self._saver is None:
            return

        if acts is not None and callable(acts):
            acts = acts()
        self._saver.save(acts, **kwargs)

    save = __call__


ActSave = ActSaveProvider()
ActivationSaver = ActSave
