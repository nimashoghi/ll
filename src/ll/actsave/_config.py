from pathlib import Path
from typing import Annotated, Literal

from typing_extensions import TypeAlias

from ..actsave._saver import Transform
from ..config import Field, TypedConfig


class ActSaveTransformConfig(TypedConfig):
    filter: str
    """Filter to use for selecting activations to apply this transform to."""

    transform: Transform
    """Transform to apply to the activations."""


class ActSaveSyncSaverConfig(TypedConfig):
    kind: Literal["sync"] = "sync"

    def _to_saver_arg(self):
        from ..actsave._saver import _SyncKwargs

        return "sync", _SyncKwargs()


class ActSaveAsyncSaverConfig(TypedConfig):
    kind: Literal["async"] = "async"

    max_workers: int = 4
    """Maximum number of workers to use for saving activations asynchronously."""

    def _to_saver_arg(self):
        from ..actsave._saver import _AsyncKwargs

        return "async", _AsyncKwargs(max_workers=self.max_workers)


ActSaveSaverConfig: TypeAlias = Annotated[
    ActSaveSyncSaverConfig | ActSaveAsyncSaverConfig,
    Field(discriminator="kind"),
]


class ActSaveConfig(TypedConfig):
    enabled: bool = True
    """Enable activation saving."""

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
