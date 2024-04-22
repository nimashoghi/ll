from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from typing_extensions import TypeAlias

from ..actsave._saver import Transform
from ..config import Field, TypedConfig

if TYPE_CHECKING:
    from ..model.config import BaseConfig


class ActSaveTransformConfig(TypedConfig):
    filter: str
    """Filter to use for selecting activations to apply this transform to."""

    transform: Transform
    """Transform to apply to the activations."""


class ActSaveSaverConfigBase(TypedConfig):
    pass


class ActSaveSyncSaverConfig(ActSaveSaverConfigBase):
    kind: Literal["sync"] = "sync"


class ActSaveAsyncSaverConfig(ActSaveSaverConfigBase):
    kind: Literal["async"] = "async"

    max_workers: int
    """Maximum number of workers to use for saving activations asynchronously."""


ActSaveSaverConfig: TypeAlias = Annotated[
    ActSaveSyncSaverConfig | ActSaveAsyncSaverConfig,
    Field(discriminator="kind"),
]


class ActSaveConfig(TypedConfig):
    enabled: bool = True
    """Enable activation saving."""

    write_mode: Literal["explicit", "implicit"] = "explicit"
    """Mode to use for writing activations:
    - `explicit`: This mode stores activations in memory until they are explicitly saved using `ActSave.write()`.
        If the activations are not explicitly saved by the beginning of the next step, they are discarded. They can
        also be discarded explicitly using `ActSave.discard()`. This mode is useful for saving activations only when,
        e.g. when the training loss ends up being too high or the gradient explodes. This mode is the recommended
        mode for saving activations.
    - `implicit`: This mode automatically saves all logged activations immediately after they are logged using `ActSave({...})`.
    """

    auto_save_logged_metrics: bool = False
    """If enabled, will automatically save logged metrics (using `LightningModule.log`) as activations."""

    save_dir: Path | None = None
    """Directory to save activations to. If None, will use the activation directory set in `config.directory`."""

    saver: ActSaveSaverConfig = ActSaveSyncSaverConfig()
    """Saver to use for saving activations."""

    filters: list[str] | None = None
    """List of filters (matched using `fnmatch`) to use for filtering activations to save."""

    transforms: list[ActSaveTransformConfig] | None = None
    """List of transforms to apply to the activations."""

    def __bool__(self):
        return self.enabled

    def resolve_save_dir(self, root_config: "BaseConfig"):
        if self.save_dir is not None:
            return self.save_dir

        return root_config.directory.resolve_subdirectory(root_config.id, "activation")
