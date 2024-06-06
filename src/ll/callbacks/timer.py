from datetime import timedelta
from logging import getLogger
from typing import Literal

from ..config import TypedConfig

log = getLogger(__name__)


class TimerConfig(TypedConfig):
    name: Literal["timer"] = "timer"

    duration: str | timedelta | dict[str, int] | None = None
    interval: Literal["step", "epoch"] = "step"
    verbose: bool = True

    def construct_callback(self):
        from lightning.pytorch.callbacks.timer import Timer

        return Timer(
            duration=self.duration, interval=self.interval, verbose=self.verbose
        )
