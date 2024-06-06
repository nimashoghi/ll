import copy
import importlib.util
import logging

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from typing_extensions import override

log = logging.getLogger(__name__)


class PrintTableMetricsCallback(Callback):
    """Prints a table with the metrics in columns on every epoch end."""

    def __init__(self) -> None:
        self.metrics: list = []
        self.rich_available = importlib.util.find_spec("rich") is not None

        if not self.rich_available:
            log.warning(
                "rich is not installed. Please install it to use PrintTableMetricsCallback."
            )

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.rich_available:
            return

        metrics_dict = copy.copy(trainer.callback_metrics)
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
