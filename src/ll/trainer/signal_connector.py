import logging
import os
import re
import signal
import subprocess
from collections import defaultdict
from collections.abc import Callable
from types import FrameType
from typing import Any, TypeAlias

from lightning.fabric.plugins.environments.lsf import LSFEnvironment
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.pytorch.trainer.connectors.signal_connector import _HandlersCompose
from lightning.pytorch.trainer.connectors.signal_connector import (
    _SignalConnector as _LightningSignalConnector,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_info

log = logging.getLogger(__name__)

_SIGNUM = int | signal.Signals
_HANDLER: TypeAlias = Callable[[_SIGNUM, FrameType], Any] | int | signal.Handlers | None


class _SignalConnector(_LightningSignalConnector):
    def _auto_requeue_signals(self) -> list[signal.Signals]:
        from ..model.base import BaseConfig

        if not isinstance(config := self.trainer.lightning_module.hparams, BaseConfig):
            return []

        return config.runner.submit._resolved_auto_requeue_signals()

    def _compose_and_register(
        self,
        signum: _SIGNUM,
        handlers: list[_HANDLER],
        replace_existing: bool = False,
    ):
        if not handlers or self._is_on_windows():
            return

        if self._has_already_handler(signum) and not replace_existing:
            handlers.append(signal.getsignal(signum))

        self._register_signal(signum, _HandlersCompose(handlers))

    def register_signal_handlers(self) -> None:
        if not (auto_requeue_signals := self._auto_requeue_signals()):
            return super().register_signal_handlers()

        self.received_sigterm = False
        self._original_handlers = self._get_current_signal_handlers()

        signals = defaultdict[signal.Signals, list[_HANDLER]](lambda: [])

        signals[signal.SIGTERM].append(self._sigterm_notifier_fn)

        environment = self.trainer._accelerator_connector.cluster_environment
        if isinstance(environment, SLURMEnvironment):
            log.info("SLURM auto-requeueing enabled. Setting signal handlers.")
            for signal_handler in auto_requeue_signals:
                signals[signal_handler].append(self._slurm_sigusr_handler_fn)

        if isinstance(environment, LSFEnvironment):
            log.info("LSF auto-requeueing enabled. Setting signal handlers.")
            for signal_handler in auto_requeue_signals:
                signals[signal_handler].append(self._lsf_sigusr_handler_fn)

        for signum, handlers in signals.items():
            self._compose_and_register(signum, handlers)

    def _slurm_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        rank_zero_info(f"Handling auto-requeue signal: {signum}")

        # save logger to make sure we get all the metrics
        for logger in self.trainer.loggers:
            logger.finalize("finished")

        hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(
            self.trainer.default_root_dir
        )
        self.trainer.save_checkpoint(hpc_save_path)

        if self.trainer.is_global_zero:
            # find job id
            array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
            if array_job_id is not None:
                array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
                job_id = f"{array_job_id}_{array_task_id}"
            else:
                job_id = os.environ["SLURM_JOB_ID"]

            assert re.match("[0-9_-]+", job_id)
            cmd = ["scontrol", "requeue", job_id]

            # requeue job
            log.info(f"requeing job {job_id}...")
            try:
                result = subprocess.call(cmd)
            except FileNotFoundError:
                # This can occur if a subprocess call to `scontrol` is run outside a shell context
                # Re-attempt call (now with shell context). If any error is raised, propagate to user.
                # When running a shell command, it should be passed as a single string.
                result = subprocess.call(" ".join(cmd), shell=True)

            # print result text
            if result == 0:
                log.info(f"Requeued SLURM job: {job_id}")
            else:
                log.warning(
                    f"Requeuing SLURM job {job_id} failed with error code {result}"
                )

    def _lsf_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
        rank_zero_info(f"Handling auto-requeue signal: {signum}")

        # Save logger to make sure we get all the metrics
        for logger in self.trainer.loggers:
            logger.finalize("finished")

        # Save checkpoint
        hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(
            self.trainer.default_root_dir
        )
        self.trainer.save_checkpoint(hpc_save_path)

        if self.trainer.is_global_zero:
            # Find job id
            job_id = os.getenv("LSB_JOBID")

            if job_id is not None:
                assert re.match("[0-9_-]+", job_id)
                cmd = ["brequeue", job_id]

                # Requeue job
                log.info(f"Requeuing job {job_id}...")
                try:
                    result = subprocess.call(cmd)
                except FileNotFoundError:
                    # Retry with shell context if subprocess call fails
                    result = subprocess.call(" ".join(cmd), shell=True)

                # Print result text
                if result == 0:
                    log.info(f"Requeued LSF job: {job_id}")
                else:
                    log.warning(
                        f"Requeuing LSF job {job_id} failed with error code {result}"
                    )
            else:
                log.warning(
                    "LSB_JOBID environment variable not found. Unable to requeue job."
                )
