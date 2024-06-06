import copy
import fnmatch
import importlib.util
import logging
from typing import Literal

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from typing_extensions import override

from .base import CallbackConfigBase

log = logging.getLogger(__name__)


class PrintTableMetricsCallback(Callback):
    """Prints a table with the metrics in columns on every epoch end."""

    def __init__(
        self,
        metric_patterns: list[str] | None = None,
    ) -> None:
        self.metrics: list = []
        self.rich_available = importlib.util.find_spec("rich") is not None
        self.metric_patterns = metric_patterns

        if not self.rich_available:
            log.warning(
                "rich is not installed. Please install it to use PrintTableMetricsCallback."
            )

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.rich_available:
            return

        metrics_dict = copy.copy(trainer.callback_metrics)
        # Filter metrics based on the patterns
        if self.metric_patterns is not None:
            metrics_dict = {
                key: value
                for key, value in metrics_dict.items()
                if any(
                    fnmatch.fnmatch(key, pattern) for pattern in self.metric_patterns
                )
            }
        self.metrics.append(metrics_dict)

        from rich.console import Console

        console = Console()
        table = self.create_metrics_table()
        console.print(table)

    def create_metrics_table(self):
        from rich.table import Table

        table = Table(show_header=True, header_style="bold magenta")

        # Add columns to the table based on the keys in the first metrics dictionary
        for key in self.metrics[0].keys():
            table.add_column(key)

        # Add rows to the table based on the metrics dictionaries
        for metric_dict in self.metrics:
            values: list[str] = []
            for value in metric_dict.values():
                if torch.is_tensor(value):
                    value = float(value.item())
                values.append(str(value))
            table.add_row(*values)

        return table


class PrintTableMetricsConfig(CallbackConfigBase):
    """Configuration class for PrintTableMetricsCallback."""

    name: Literal["print_table_metrics"] = "print_table_metrics"

    enabled: bool = True
    """Whether to enable the callback or not."""

    metric_patterns: list[str] | None = None
    """List of patterns to filter the metrics to be displayed. If None, all metrics are displayed."""

    @override
    def construct_callbacks(self, root_config):
        yield PrintTableMetricsCallback(metric_patterns=self.metric_patterns)
