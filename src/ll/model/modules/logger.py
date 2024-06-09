from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torchmetrics
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.utilities.types import _METRIC
from lightning_utilities.core.rank_zero import rank_zero_warn
from typing_extensions import override

from ...actsave import ActSave
from ...util.typing_utils import mixin_base_type
from ..config import BaseConfig


@dataclass(frozen=True, kw_only=True)
class _LogContext:
    prefix: str | None = None
    disabled: bool | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class LoggerModuleMixin:
    @property
    def log_dir(self):
        if not isinstance(self, (LightningModule, LightningDataModule)):
            raise TypeError(
                "log_dir can only be used on LightningModule or LightningDataModule"
            )

        if (trainer := self.trainer) is None:
            raise RuntimeError("trainer is not defined")

        if (logger := trainer.logger) is None:
            raise RuntimeError("trainer.logger is not defined")

        if (log_dir := logger.log_dir) is None:
            raise RuntimeError("trainer.logger.log_dir is not defined")

        return Path(log_dir)

    @property
    def should_update_logs(self):
        if not isinstance(self, (LightningModule, LightningDataModule)):
            raise TypeError(
                "should_update_logs can only be used on LightningModule or LightningDataModule"
            )

        trainer = self._trainer if isinstance(self, LightningModule) else self.trainer
        if trainer is None:
            return True

        return trainer._logger_connector.should_update_logs


class LoggerLightningModuleMixin(LoggerModuleMixin, mixin_base_type(LightningModule)):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._logger_prefix_stack = deque[_LogContext]()

    if TYPE_CHECKING:

        @contextmanager
        def log_context(
            self,
            prefix: str | None = None,
            *,
            disabled: bool | None = None,
            prog_bar: bool | None = None,
            logger: bool | None = None,
            on_step: bool | None = None,
            on_epoch: bool | None = None,
            reduce_fx: str | Callable | None = None,
            enable_graph: bool | None = None,
            sync_dist: bool | None = None,
            sync_dist_group: Any | None = None,
            add_dataloader_idx: bool | None = None,
            batch_size: int | None = None,
            rank_zero_only: bool | None = None,
        ) -> Generator[None, None, None]: ...

    else:

        @contextmanager
        def log_context(
            self, prefix: str | None = None, *, disabled: bool | None = None, **kwargs
        ) -> Generator[None, None, None]:
            self._logger_prefix_stack.append(
                _LogContext(
                    prefix=prefix,
                    disabled=disabled,
                    kwargs=kwargs,
                )
            )
            try:
                yield
            finally:
                _ = self._logger_prefix_stack.pop()

    if TYPE_CHECKING:

        @override
        def log(  # type: ignore[override]
            self,
            name: str,
            value: _METRIC,
            *,
            prog_bar: bool = False,
            logger: bool | None = None,
            on_step: bool | None = None,
            on_epoch: bool | None = None,
            reduce_fx: str | Callable = "mean",
            enable_graph: bool = False,
            sync_dist: bool = False,
            sync_dist_group: Any | None = None,
            add_dataloader_idx: bool = True,
            batch_size: int | None = None,
            metric_attribute: str | None = None,
            rank_zero_only: bool = False,
        ) -> None: ...

    else:

        @override
        def log(self, name: str, value: _METRIC, **kwargs) -> None:
            # join all prefixes
            prefix = "".join(c.prefix for c in self._logger_prefix_stack if c.prefix)
            name = f"{prefix}{name}"

            # check for disabled context:
            # if the topmost non-null context is disabled, then we don't log
            for c in reversed(self._logger_prefix_stack):
                if c.disabled is not None:
                    if c.disabled:
                        rank_zero_warn(
                            f"Skipping logging of {name} due to disabled context"
                        )
                        return
                    else:
                        break

            fn_kwargs = {}
            for c in self._logger_prefix_stack:
                fn_kwargs.update(c.kwargs)
            fn_kwargs.update(kwargs)

            self._logger_actsave(name, value)

            return super().log(name, value, **fn_kwargs)

    def _logger_actsave(self, name: str, value: _METRIC) -> None:
        hparams = cast(BaseConfig, self.hparams)
        if (
            not hparams.trainer.actsave
            or not hparams.trainer.actsave.auto_save_logged_metrics
        ):
            return

        ActSave.save(
            {
                f"logger.{name}": lambda: value.compute()
                if isinstance(value, torchmetrics.Metric)
                else value
            }
        )
