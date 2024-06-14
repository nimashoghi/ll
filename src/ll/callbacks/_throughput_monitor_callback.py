# type: ignore
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

import torch
from lightning.fabric.plugins import Precision as FabricPrecision
from lightning.fabric.utilities.throughput import (
    _plugin_to_compute_dtype as fabric_plugin_to_compute_dtype,
)
from lightning.fabric.utilities.throughput import get_available_flops
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.plugins import (
    BitsandbytesPrecision,
    DeepSpeedPrecision,
    DoublePrecision,
    FSDPPrecision,
    HalfPrecision,
    MixedPrecision,
    Precision,
    TransformerEnginePrecision,
    XLAPrecision,
)
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn
from typing_extensions import override

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


_THROUGHPUT_METRICS = Dict[str, Union[int, float]]

T = TypeVar("T", bound=float)


class _MonotonicWindow(List[T]):
    """Custom fixed size list that only supports right-append and ensures that all values increase monotonically."""

    def __init__(self, maxlen: int) -> None:
        super().__init__()
        self.maxlen = maxlen

    @property
    def last(self) -> Optional[T]:
        if len(self) > 0:
            return self[-1]
        return None

    @override
    def append(self, x: T) -> None:
        last = self.last
        if last is not None and last >= x:
            rank_zero_warn(
                f"Expected the value to increase, last: {last}, current: {x}"
            )
        list.append(self, x)
        # truncate excess
        if len(self) > self.maxlen:
            del self[0]

    @override
    def __setitem__(self, key: Any, value: Any) -> None:
        # assigning is not implemented since we don't use it. it could be by checking all previous values
        raise NotImplementedError("__setitem__ is not supported")


class Throughput:
    """Computes throughput.

    +------------------------+-------------------------------------------------------------------------------------+
    | Key                    | Value                                                                               |
    +========================+=====================================================================================+
    | batches_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of batches |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples_per_sec        | Rolling average (over ``window_size`` most recent updates) of the number of samples |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | items_per_sec          | Rolling average (over ``window_size`` most recent updates) of the number of items   |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | flpps_per_sec          | Rolling average (over ``window_size`` most recent updates) of the number of flops   |
    |                        | processed per second                                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/batches_per_sec | batches_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/samples_per_sec | samples_per_sec divided by world size                                               |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/items_per_sec   | items_per_sec divided by world size. This may include padding depending on the data |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/flops_per_sec   | flops_per_sec divided by world size.                                                |
    +--------------------------+-----------------------------------------------------------------------------------+
    | device/mfu             | device/flops_per_sec divided by world size.                                         |
    +--------------------------+-----------------------------------------------------------------------------------+
    | time                   | Total elapsed time                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | batches                | Total batches seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | samples                | Total samples seen                                                                  |
    +--------------------------+-----------------------------------------------------------------------------------+
    | lengths                | Total items seen                                                                    |
    +--------------------------+-----------------------------------------------------------------------------------+

    Example::

        throughput = Throughput()
        t0 = time()
        for i in range(1000):
            do_work()
            if torch.cuda.is_available(): torch.cuda.synchronize()  # required or else time() won't be correct
            throughput.update(time=time() - t0, samples=i)
            if i % 10 == 0:
                print(throughput.compute())

    Notes:
        - The implementation assumes that devices FLOPs are all the same as it normalizes by the world size and only
          takes a single ``available_flops`` value.
        - items_per_sec, flops_per_sec and MFU do not account for padding if present. We suggest using
          samples_per_sec or batches_per_sec to measure throughput under this circumstance.

    Args:
        available_flops: Number of theoretical flops available for a single device.
        world_size: Number of devices available across hosts. Global metrics are not included if the world size is 1.
        window_size: Number of batches to use for a rolling average.
        separator: Key separator to use when creating per-device and global metrics.

    """

    def __init__(
        self,
        available_flops: Optional[float] = None,
        world_size: int = 1,
        window_size: int = 100,
        separator: str = "/",
    ) -> None:
        self.available_flops = available_flops
        self.separator = separator
        assert world_size > 0
        self.world_size = world_size

        # throughput is computed over a window of values. at least 2 is enforced since it looks at the difference
        # between the first and last elements
        assert window_size > 1
        # custom class instead of `deque(maxlen=)` because it's easy for users to mess up their timer/counters and log
        # values that do not increase monotonically. this class will raise an error if that happens.
        self._time: _MonotonicWindow[float] = _MonotonicWindow(maxlen=window_size)
        self._batches: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._samples: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._lengths: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._flops: Deque[int] = deque(maxlen=window_size)

    def update(
        self,
        *,
        time: float,
        batches: int,
        samples: int,
        lengths: Optional[int] = None,
        flops: Optional[int] = None,
    ) -> None:
        """Update throughput metrics.

        Args:
            time: Total elapsed time in seconds. It should monotonically increase by the iteration time with each
                call.
            batches: Total batches seen per device. It should monotonically increase with each call.
            samples: Total samples seen per device. It should monotonically increase by the batch size with each call.
            lengths: Total length of the samples seen. It should monotonically increase by the lengths of a batch with
                each call.
            flops: Flops elapased per device since last ``update()`` call. You can easily compute this by using
                :func:`measure_flops` and multiplying it by the number of batches that have been processed.
                The value might be different in each device if the batch size is not the same.

        """
        self._time.append(time)
        if samples < batches:
            raise ValueError(
                f"Expected samples ({samples}) to be greater or equal than batches ({batches})"
            )
        self._batches.append(batches)
        self._samples.append(samples)
        if lengths is not None:
            if lengths < samples:
                raise ValueError(
                    f"Expected lengths ({lengths}) to be greater or equal than samples ({samples})"
                )
            self._lengths.append(lengths)
            if len(self._samples) != len(self._lengths):
                raise RuntimeError(
                    f"If lengths are passed ({len(self._lengths)}), there needs to be the same number of samples"
                    f" ({len(self._samples)})"
                )
        if flops is not None:
            # sum of flops across ranks
            self._flops.append(flops * self.world_size)

    def compute(self) -> _THROUGHPUT_METRICS:
        """Compute throughput metrics."""
        metrics = {
            "time": self._time[-1],
            "batches": self._batches[-1],
            "samples": self._samples[-1],
        }
        if self._lengths:
            metrics["lengths"] = self._lengths[-1]

        add_global_metrics = self.world_size > 1
        # a different but valid design choice would be to still compute all these metrics even if the window of values
        # has not been filled
        if len(self._time) == self._time.maxlen:
            elapsed_time = self._time[-1] - self._time[0]
            elapsed_batches = self._batches[-1] - self._batches[0]
            elapsed_samples = self._samples[-1] - self._samples[0]
            # we are safe from ZeroDivisionError thanks to `_MonotonicWindow`
            dev_samples_per_sec = elapsed_samples / elapsed_time
            dev_batches_per_sec = elapsed_batches / elapsed_time
            metrics.update(
                {
                    f"device{self.separator}batches_per_sec": elapsed_batches
                    / elapsed_time,
                    f"device{self.separator}samples_per_sec": dev_samples_per_sec,
                }
            )
            if add_global_metrics:
                samples_per_sec = dev_batches_per_sec * self.world_size
                metrics.update(
                    {
                        "batches_per_sec": samples_per_sec,
                        "samples_per_sec": dev_samples_per_sec * self.world_size,
                    }
                )

            if len(self._lengths) == self._lengths.maxlen:
                elapsed_lengths = self._lengths[-1] - self._lengths[0]
                dev_items_per_sec = elapsed_lengths / elapsed_time
                metrics[f"device{self.separator}items_per_sec"] = dev_items_per_sec
                if add_global_metrics:
                    items_per_sec = dev_items_per_sec * self.world_size
                    metrics["items_per_sec"] = items_per_sec

        if len(self._flops) == self._flops.maxlen:
            elapsed_flops = sum(self._flops) - self._flops[0]
            elapsed_time = self._time[-1] - self._time[0]
            flops_per_sec = elapsed_flops / elapsed_time
            dev_flops_per_sec = flops_per_sec / self.world_size
            if add_global_metrics:
                metrics["flops_per_sec"] = flops_per_sec
            metrics[f"device{self.separator}flops_per_sec"] = dev_flops_per_sec
            if self.available_flops:
                metrics[f"device{self.separator}mfu"] = (
                    dev_flops_per_sec / self.available_flops
                )

        return metrics

    def reset(self) -> None:
        self._time.clear()
        self._batches.clear()
        self._samples.clear()
        self._lengths.clear()
        self._flops.clear()


class ThroughputMonitor(Callback):
    r"""Computes and logs throughput with the :class:`~lightning.fabric.utilities.throughput.Throughput`

    Example::

        class MyModel(LightningModule):
            def setup(self, stage):
                with torch.device("meta"):
                    model = MyModel()

                    def sample_forward():
                        batch = torch.randn(..., device="meta")
                        return model(batch)

                    self.flops_per_batch = measure_flops(model, sample_forward, loss_fn=torch.Tensor.sum)


        logger = ...
        throughput = ThroughputMonitor(batch_size_fn=lambda batch: batch.size(0))
        trainer = Trainer(max_steps=1000, log_every_n_steps=10, callbacks=throughput, logger=logger)
        model = MyModel()
        trainer.fit(model)

    Notes:
        - It assumes that the batch size is the same during all iterations.
        - It will try to access a ``flops_per_batch`` attribute on your ``LightningModule`` on every iteration.
          We suggest using the :func:`~lightning.fabric.utilities.throughput.measure_flops` function for this.
          You might want to compute it differently each time based on your setup.

    Args:
        batch_size_fn: A function to compute the number of samples given a batch.
        length_fn: A function to compute the number of items in a sample given a batch.
        \**kwargs: See available parameters in
            :class:`~lightning.fabric.utilities.throughput.Throughput`

    """

    def __init__(
        self,
        batch_size_fn: Callable[[Any], int],
        length_fn: Optional[Callable[[Any], int | None]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.batch_size_fn = batch_size_fn
        self.length_fn = length_fn
        self.available_flops: Optional[int] = None
        self._throughputs: Dict[RunningStage, Throughput] = {}
        self._t0s: Dict[RunningStage, float] = {}
        self._samples: Dict[RunningStage, int] = {}
        self._lengths: Dict[RunningStage, int] = {}

    @override
    def setup(
        self, trainer: "Trainer", pl_module: "LightningModule", stage: str
    ) -> None:
        dtype = _plugin_to_compute_dtype(trainer.precision_plugin)
        self.available_flops = get_available_flops(trainer.strategy.root_device, dtype)

        if stage == TrainerFn.FITTING and trainer.enable_validation:
            # `fit` includes validation inside
            throughput = Throughput(
                available_flops=self.available_flops,
                world_size=trainer.world_size,
                **self.kwargs,
            )
            self._throughputs[RunningStage.VALIDATING] = throughput

        throughput = Throughput(
            available_flops=self.available_flops,
            world_size=trainer.world_size,
            **self.kwargs,
        )
        stage = trainer.state.stage
        assert stage is not None
        self._throughputs[stage] = throughput

    def _start(self, trainer: "Trainer") -> None:
        stage = trainer.state.stage
        assert stage is not None
        self._throughputs[stage].reset()
        self._samples[stage] = 0
        self._lengths[stage] = 0
        self._t0s[stage] = time.perf_counter()

    def _update(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        batch: Any,
        iter_num: int,
    ) -> None:
        stage = trainer.state.stage
        assert stage is not None
        throughput = self._throughputs[stage]

        if trainer.strategy.root_device.type == "cuda":
            # required or else perf_counter() won't be correct
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._t0s[stage]
        if self.length_fn is not None:
            with torch.inference_mode():
                if (length := self.length_fn(batch)) is not None:
                    self._lengths[stage] += length

        if hasattr(pl_module, "flops_per_batch"):
            flops_per_batch = pl_module.flops_per_batch
        else:
            rank_zero_warn(
                "When using the `ThroughputMonitor`, you need to define a `flops_per_batch` attribute or property"
                f" in {type(pl_module).__name__} to compute the FLOPs."
            )
            flops_per_batch = None

        with torch.inference_mode():
            self._samples[stage] += self.batch_size_fn(batch)

        throughput.update(
            time=elapsed,
            batches=iter_num,
            samples=self._samples[stage],
            lengths=None if self.length_fn is None else self._lengths[stage],
            flops=flops_per_batch,
        )

    def _compute(self, trainer: "Trainer", iter_num: Optional[int] = None) -> None:
        if not trainer._logger_connector.should_update_logs:
            return
        stage = trainer.state.stage
        assert stage is not None
        throughput = self._throughputs[stage]
        metrics = throughput.compute()
        # prefix with the stage to avoid collisions
        metrics = {
            f"{stage.value}{throughput.separator}{k}": v for k, v in metrics.items()
        }
        trainer._logger_connector.log_metrics(metrics, step=iter_num)  # type: ignore[arg-type]

    @override
    @rank_zero_only
    def on_train_start(self, trainer: "Trainer", *_: Any) -> None:
        self._start(trainer)

    @override
    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: Any,
        batch: Any,
        *_: Any,
    ) -> None:
        self._update(trainer, pl_module, batch, trainer.fit_loop.total_batch_idx + 1)
        # log only when gradient accumulation is over. this ensures that we only measure when the effective batch has
        # finished and the `optimizer.step()` time is included
        if not trainer.fit_loop._should_accumulate():
            self._compute(trainer)

    @override
    @rank_zero_only
    def on_validation_start(self, trainer: "Trainer", *_: Any) -> None:
        if trainer.sanity_checking:
            return
        self._start(trainer)

    @override
    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: Any,
        batch: Any,
        *_: Any,
        **__: Any,
    ) -> None:
        if trainer.sanity_checking:
            return
        iter_num = trainer._evaluation_loop.batch_progress.total.ready
        self._update(trainer, pl_module, batch, iter_num)
        self._compute(trainer, iter_num)

    @override
    @rank_zero_only
    def on_validation_end(self, trainer: "Trainer", *_: Any) -> None:
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return
        # add the validation time to the training time before continuing to avoid sinking the training throughput
        training_finished = self._t0s[RunningStage.TRAINING] + sum(
            self._throughputs[RunningStage.TRAINING]._time
        )
        time_between_train_and_val = (
            self._t0s[RunningStage.VALIDATING] - training_finished
        )
        val_time = sum(self._throughputs[RunningStage.VALIDATING]._time)
        self._t0s[RunningStage.TRAINING] += time_between_train_and_val + val_time

    @override
    @rank_zero_only
    def on_test_start(self, trainer: "Trainer", *_: Any) -> None:
        self._start(trainer)

    @override
    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: Any,
        batch: Any,
        *_: Any,
        **__: Any,
    ) -> None:
        iter_num = trainer._evaluation_loop.batch_progress.total.ready
        self._update(trainer, pl_module, batch, iter_num)
        self._compute(trainer, iter_num)

    @override
    @rank_zero_only
    def on_predict_start(self, trainer: "Trainer", *_: Any) -> None:
        self._start(trainer)

    @override
    @rank_zero_only
    def on_predict_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: Any,
        batch: Any,
        *_: Any,
        **__: Any,
    ) -> None:
        iter_num = trainer.predict_loop.batch_progress.total.ready
        self._update(trainer, pl_module, batch, iter_num)
        self._compute(trainer, iter_num)


def _plugin_to_compute_dtype(plugin: Union[FabricPrecision, Precision]) -> torch.dtype:
    # TODO: integrate this into the precision plugins
    if not isinstance(plugin, Precision):
        return fabric_plugin_to_compute_dtype(plugin)
    if isinstance(plugin, BitsandbytesPrecision):
        return plugin.dtype
    if isinstance(plugin, HalfPrecision):
        return plugin._desired_input_dtype
    if isinstance(plugin, MixedPrecision):
        return torch.bfloat16 if plugin.precision == "bf16-mixed" else torch.half
    if isinstance(plugin, DoublePrecision):
        return torch.double
    if isinstance(plugin, (XLAPrecision, DeepSpeedPrecision)):
        return plugin._desired_dtype
    if isinstance(plugin, TransformerEnginePrecision):
        return torch.int8
    if isinstance(plugin, FSDPPrecision):
        return plugin.mixed_precision_config.reduce_dtype or torch.float32
    if isinstance(plugin, Precision):
        return torch.float32
    raise NotImplementedError(plugin)
