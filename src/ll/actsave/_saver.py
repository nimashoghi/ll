import concurrent.futures
import contextlib
import fnmatch
import io
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial, wraps
from logging import getLogger
from pathlib import Path
from typing import Generic, TypeAlias, cast, overload

import numpy as np
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from typing_extensions import ParamSpec, TypeVar, assert_never, override

from ._config import ActSaveAsyncSaverConfig, ActSaveConfig, ActSaveSyncSaverConfig

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
class _Activation:
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

    def to_bytes(self):
        np_self = self()

        with io.BytesIO() as f:
            np.save(f, np_self)
            return f.getvalue()

    @classmethod
    def from_value_or_lambda(cls, name: str, value_or_lambda: ValueOrLambda):
        return cls(name, WeakRef(value_or_lambda))

    @classmethod
    def from_dict(cls, d: Mapping[str, ValueOrLambda]):
        return [cls.from_value_or_lambda(k, v) for k, v in d.items()]


Transform = Callable[[_Activation], Mapping[str, ValueOrLambda]]


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


@dataclass(frozen=True)
class ActSaveContextInfo:
    label: str
    _unique_id: str = field(
        init=False,
        default_factory=lambda: str(uuid.uuid4()),
    )


class _SaverBase(ABC):
    def __init__(
        self,
        actsave_config: ActSaveConfig,
        save_dir: Path,
        prefixes_fn: Callable[[], Sequence[ActSaveContextInfo]],
        *,
        filters: list[str] | None = None,
        transforms: list[tuple[str, Transform]] | None = None,
    ):
        self.actsave_config = actsave_config

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

        self._acts_to_save: list[tuple[bytes, Path]] = []

    @abstractmethod
    def save_to_file(
        self,
        activation_bytes: bytes,
        save_dir: Path,
    ) -> Path: ...

    def on_new_step(self):
        if self.actsave_config.write_mode != "explicit":
            return

        try:
            # Write all activations to disk
            for activation_bytes, save_dir in self._acts_to_save:
                self.save_to_file(activation_bytes, save_dir)
        finally:
            # Clear the activations
            self._acts_to_save.clear()

    def _save_activation(self, activation: _Activation):
        # Save the activation to self._save_dir / {name} /
        prefixes = [prefix.label for prefix in self._prefixes_fn()]
        file_name = ".".join(prefixes + [activation.name])
        save_dir = self._save_dir / file_name
        save_dir.mkdir(exist_ok=True, parents=True)

        # We condition on the write mode
        match self.actsave_config.write_mode:
            case "implicit":
                # This mode automatically saves all logged activations immediately after they are logged using `ActSave({...})`.
                # Save the activation to a file
                self.save_to_file(activation.to_bytes(), save_dir)
            case "explicit":
                # This mode stores activations in memory until they are explicitly saved using `ActSave.write()`.
                # If the activations are not explicitly saved by the beginning of the next step, they are discarded. They can
                # also be discarded explicitly using `ActSave.discard()`. This mode is useful for saving activations only when,
                # e.g. when the training loss ends up being too high or the gradient explodes. This mode is the recommended
                # mode for saving activations.
                self._acts_to_save.append((activation.to_bytes(), save_dir))
            case _:
                assert_never(self.actsave_config.write_mode)

    @_ignore_if_scripting
    def save(
        self,
        acts: dict[str, ValueOrLambda] | None = None,
        /,
        **kwargs: ValueOrLambda,
    ):
        kwargs.update(acts or {})

        # Build activations
        activations = _Activation.from_dict(kwargs)

        transformed_activations: list[_Activation] = []

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
                    transformed_activations.extend(_Activation.from_dict(transform_out))

        # Now, we save the transformed activations.
        for transformed_activation in transformed_activations:
            self._save_activation(transformed_activation)


class _SyncNumpySaver(_SaverBase):
    def __init__(
        self,
        actsave_config: ActSaveConfig,
        config: ActSaveSyncSaverConfig,
        save_dir: Path,
        prefixes_fn: Callable[[], Sequence[ActSaveContextInfo]],
        *,
        filters: list[str] | None = None,
        transforms: list[tuple[str, Transform]] | None = None,
    ):
        super().__init__(
            actsave_config,
            save_dir,
            prefixes_fn,
            filters=filters,
            transforms=transforms,
        )

        self.config = config

    @override
    def save_to_file(self, activation_bytes: bytes, save_dir: Path) -> Path:
        # Get the next id and save the activation
        id = len(list(save_dir.glob("*.npy")))
        fpath = save_dir / f"{id:04d}.npy"
        assert not fpath.exists(), f"File {fpath} already exists"

        # Save the activation to a file
        fpath.write_bytes(activation_bytes)
        return fpath


class _AsyncNumpySaver(_SaverBase):
    """
    Very similar to _SyncNumpySaver, but saves activations asynchronously
    (i.e., in a separate thread)
    """

    @override
    def __init__(
        self,
        actsave_config: ActSaveConfig,
        config: ActSaveAsyncSaverConfig,
        save_dir: Path,
        prefixes_fn: Callable[[], Sequence[ActSaveContextInfo]],
        *,
        filters: list[str] | None = None,
        transforms: list[tuple[str, Transform]] | None = None,
    ):
        super().__init__(
            actsave_config,
            save_dir,
            prefixes_fn,
            filters=filters,
            transforms=transforms,
        )

        self.config = config
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        )

    @staticmethod
    def _write_fn(fpath: Path, activation_bytes: bytes):
        fpath.write_bytes(activation_bytes)

    @override
    def save_to_file(self, activation_bytes: bytes, save_dir: Path) -> Path:
        # Get the next id and save the activation
        id = len(list(save_dir.glob("*.npy")))
        fpath = save_dir / f"{id:04d}.npy"
        assert not fpath.exists(), f"File {fpath} already exists"

        # Save the activation to a file
        # We save the activation asynchronously
        self._executor.submit(self._write_fn, fpath, activation_bytes)
        return fpath


class ActSaveProvider:
    _saver: _SaverBase | None = None
    _contexts: list[ActSaveContextInfo] = []

    def initialize(self, config: ActSaveConfig, save_dir: Path):
        if self._saver is None:
            # Create the actsave directory
            save_dir.mkdir(parents=True, exist_ok=True)

            # Resolve the transforms
            if (transforms := config.transforms) is not None:
                transforms = [
                    (transform.filter, transform.transform) for transform in transforms
                ]

            match config.saver:
                case ActSaveSyncSaverConfig():
                    saver_cls = partial(_SyncNumpySaver, config, config.saver)
                case ActSaveAsyncSaverConfig():
                    saver_cls = partial(_AsyncNumpySaver, config, config.saver)
                case _:
                    assert_never(config.saver)

            self._saver = saver_cls(
                save_dir,
                lambda: self._contexts,
                filters=config.filters,
                transforms=transforms,
            )

    @contextlib.contextmanager
    def enabled(self, config: ActSaveConfig, save_dir: Path):
        prev = self._saver
        self.initialize(config, save_dir)
        try:
            yield
        finally:
            self._saver = prev

    @override
    def __init__(self):
        super().__init__()

        self._saver = None
        self._contexts = []

    @contextlib.contextmanager
    def context(self, label: str):
        if torch.jit.is_scripting():
            yield
            return

        if self._saver is None:
            yield
            return

        _ensure_supported()

        context = ActSaveContextInfo(label)
        log.debug(f"Entering ActSave context {context}")
        self._contexts.append(context)
        try:
            yield
        finally:
            # Pop until we find our context
            while self._contexts:
                if self._contexts.pop() == context:
                    break

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

    def _start_stage(self):
        # If we have a saver, we should alert it that we're starting a new stage.
        if (saver := self._saver) is None:
            return

        saver.on_new_step()


ActSave = ActSaveProvider()
ActivationSaver = ActSave
