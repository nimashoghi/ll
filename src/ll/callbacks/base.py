from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from lightning.pytorch import Callback

from ..config import TypedConfig

if TYPE_CHECKING:
    from ..model.config import BaseConfig


class CallbackConfigBase(TypedConfig, ABC):
    @abstractmethod
    def construct_callbacks(self, root_config: "BaseConfig") -> Iterable[Callback]: ...
