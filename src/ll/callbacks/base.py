from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, TypedDict

from lightning.pytorch import Callback

from ..config import TypedConfig

if TYPE_CHECKING:
    from ..model.config import BaseConfig


class CallbackMetadataDict(TypedDict, total=False):
    ignore_if_exists: bool
    """If `True`, the callback will not be added if another callback with the same class already exists."""

    priority: int
    """Priority of the callback. Callbacks with higher priority will be loaded first."""


class CallbackMetadataConfig(TypedConfig):
    ignore_if_exists: bool = False
    """If `True`, the callback will not be added if another callback with the same class already exists."""

    priority: int = 0
    """Priority of the callback. Callbacks with higher priority will be loaded first."""


@dataclass(frozen=True)
class CallbackWithMetadata:
    callback: Callback
    metadata: CallbackMetadataConfig


ConstructedCallback: TypeAlias = Callback | CallbackWithMetadata


class CallbackConfigBase(TypedConfig, ABC):
    metadata: CallbackMetadataConfig = CallbackMetadataConfig()
    """Metadata for the callback."""

    def with_metadata(self, callback: Callback, **metadata: CallbackMetadataDict):
        return CallbackWithMetadata(
            callback=callback, metadata=self.metadata.model_copy(update=metadata)
        )

    @abstractmethod
    def construct_callbacks(
        self, root_config: "BaseConfig"
    ) -> Iterable[Callback | CallbackWithMetadata]: ...

    def _construct_callbacks_with_metadata(
        self, root_config: "BaseConfig"
    ) -> Iterable[CallbackWithMetadata]:
        for callback in self.construct_callbacks(root_config):
            if isinstance(callback, CallbackWithMetadata):
                yield callback
                continue

            callback = self.with_metadata(callback)
            yield callback


def _process_and_filter_callbacks(
    callbacks: Iterable[CallbackWithMetadata],
) -> list[Callback]:
    callbacks = list(callbacks)

    # Sort by priority (higher priority first)
    callbacks.sort(key=lambda callback: callback.metadata.priority, reverse=True)

    # Process `ignore_if_exists`
    callbacks = _filter_ignore_if_exists(callbacks)

    return [callback.callback for callback in callbacks]


def _filter_ignore_if_exists(callbacks: list[CallbackWithMetadata]):
    # First, let's do a pass over all callbacks to hold the count of each callback class
    callback_classes = Counter(callback.callback.__class__ for callback in callbacks)

    # Remove non-duplicates
    callbacks_filtered: list[CallbackWithMetadata] = []
    for callback in callbacks:
        # If `ignore_if_exists` is `True` and there is already a callback of the same class, skip this callback
        if (
            callback.metadata.ignore_if_exists
            and callback_classes[callback.callback.__class__] > 1
        ):
            continue

        callbacks_filtered.append(callback)

    return callbacks_filtered
