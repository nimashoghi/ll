import copy
import getpass
import os
import subprocess
import tempfile
import traceback
from collections import Counter
from contextlib import ExitStack
from datetime import timedelta
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Generic, Protocol, Sequence, cast, overload, runtime_checkable

import cloudpickle as pickle
from tqdm.auto import tqdm
from typing_extensions import TypeVar, TypeVarTuple, Unpack, deprecated, override

from submitit import AutoExecutor

from .model.config import BaseConfig
from .trainer import Trainer
from .util.environment import (
    remove_slurm_environment_variables,
    remove_wandb_environment_variables,
)
from .util.snapshot import snapshot_modules

log = getLogger(__name__)


TConfig = TypeVar("TConfig", bound=BaseConfig, infer_variance=True)
TReturn = TypeVar("TReturn", default=None, infer_variance=True)
TArguments = TypeVarTuple("TArguments", default=Unpack[tuple[()]])


@runtime_checkable
class RunProtocol(Protocol[TConfig, TReturn, Unpack[TArguments]]):
    def __call__(self, config: TConfig, *args: Unpack[TArguments]) -> TReturn: ...


class Runner(Generic[TConfig, TReturn, Unpack[TArguments]]):
    DEFAULT_ENV = {}
    SNAPSHOT_ENV_NAME = "LL_SNAPSHOT"

    @classmethod
    def active_snapshot(cls) -> Path | None:
        if (snapshot := os.environ.get(cls.SNAPSHOT_ENV_NAME)) is not None:
            return Path(snapshot)
        return None

    @override
    def __init__(
        self,
        run: RunProtocol[TConfig, TReturn, Unpack[TArguments]],
        *,
        slurm_job_name: str = "ll",
        validate_config_before_run: bool = True,
        validate_strict: bool = True,
    ):
        """This is the initialization function for a class that takes in a run protocol, an auto wrap run
        boolean, and a slurm job name string.

        Parameters
        ----------
        run : RunProtocol[TConfig, Unpack[TArguments]]
            `run` is an instance of a class that implements the `RunProtocol` interface. It represents the main function or entry point of the program that will be executed.
        slurm_job_name : str, optional
            The `slurm_job_name` parameter is a string that represents the name of the job when submitting it to a SLURM cluster.
        validate_config_before_run : bool, optional
            The `validate_config_before_run` parameter is a boolean that represents whether or not to validate the configuration before running the program.
        validate_strict: bool, optional
            Should `validate_config_before_run` be strict? If `True`, the configuration will be validated strictly. If `False`, the configuration will be validated non-strictly.
        """

        super().__init__()

        self._run = run
        self.slurm_job_name = slurm_job_name
        self.validate_config_before_run = validate_config_before_run
        self.validate_strict = validate_strict
        self._init_kwargs = {
            "slurm_job_name": slurm_job_name,
            "validate_config_before_run": validate_config_before_run,
            "validate_strict": validate_strict,
        }

    @property
    def run(self) -> RunProtocol[TConfig, TReturn, Unpack[TArguments]]:
        run = self._run

        @wraps(run)
        def wrapped_run(config: TConfig, *args: Unpack[TArguments]) -> TReturn:
            nonlocal self

            # If `auto_call_trainer_init_from_runner`, we call `Trainer.runner_init` before running the program.
            if config.trainer.auto_call_trainer_init_from_runner:
                Trainer.runner_init(config)

            # If `validate_config_before_run`, we validate the configuration before running the program.
            if self.validate_config_before_run:
                config = type(config).model_validate(
                    config, strict=self.validate_strict
                )

            with ExitStack() as stack:
                nonlocal run

                if config.trainer.auto_wrap_trainer:
                    stack.enter_context(Trainer.context(config))
                    log.critical("Auto-wrapping run in Trainer context")

                return run(config, *args)

            raise RuntimeError("ExitStack should never raise an exception")

        return wrapped_run

    @staticmethod
    def _resolve_run(
        run: TConfig | tuple[TConfig, Unpack[TArguments]],
        copy_config: bool = True,
        reset_id: bool = False,
    ):
        if isinstance(run, tuple):
            (config, *args) = run
        else:
            config = cast(TConfig, run)
            args = []
        args = cast(tuple[Unpack[TArguments]], args)
        if copy_config:
            config = copy.deepcopy(config)
        if reset_id:
            config.id = BaseConfig.generate_id(ignore_rng=True)
        return (config, args)

    @staticmethod
    def _resolve_runs(
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        copy_config: bool = True,
        reset_id: bool = False,
    ):
        resolved: list[tuple[TConfig, tuple[Unpack[TArguments]]]] = []
        for run in runs:
            resolved.append(
                Runner._resolve_run(run, copy_config=copy_config, reset_id=reset_id)
            )

        return resolved

    @deprecated("Use __call__ instead")
    @overload
    def local(
        self,
        run: TConfig | tuple[TConfig, Unpack[TArguments]],
        /,
        *,
        env: dict[str, str] | None = None,
        reset_id: bool = True,
    ) -> TReturn: ...

    @deprecated("Use __call__ instead")
    @overload
    def local(
        self,
        run_1: TConfig | tuple[TConfig, Unpack[TArguments]],
        run_2: TConfig | tuple[TConfig, Unpack[TArguments]],
        /,
        *runs: TConfig | tuple[TConfig, Unpack[TArguments]],
        env: dict[str, str] | None = None,
        reset_id: bool = True,
    ) -> list[TReturn]: ...

    @deprecated("Use __call__ instead")
    def local(
        self,
        *runs: TConfig | tuple[TConfig, Unpack[TArguments]],
        env: dict[str, str] | None = None,
        reset_id: bool = True,
    ):

        return_values: list[TReturn] = []
        for run in runs:
            config, args = self._resolve_run(run)
            if reset_id:
                config.id = BaseConfig.generate_id(ignore_rng=True)

            env = {**self.DEFAULT_ENV, **(env or {})}
            env_old = {k: os.environ.get(k, None) for k in env}
            os.environ.update(env)
            try:
                return_value = self.run(config, *args)
                return_values.append(return_value)
            finally:
                for k, v in env_old.items():
                    if v is None:
                        _ = os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        return return_values[0] if len(return_values) == 1 else return_values

    def __call__(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        env: dict[str, str] | None = None,
        reset_id: bool = True,
    ):
        return_values: list[TReturn] = []
        for run in runs:
            config, args = self._resolve_run(run)
            if reset_id:
                config.id = BaseConfig.generate_id(ignore_rng=True)

            env = {**self.DEFAULT_ENV, **(env or {})}
            env_old = {k: os.environ.get(k, None) for k in env}
            os.environ.update(env)
            try:
                return_value = self.run(config, *args)
                return_values.append(return_value)
            finally:
                for k, v in env_old.items():
                    if v is None:
                        _ = os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        return return_values[0] if len(return_values) == 1 else return_values

    def _launch_session(
        self,
        config_paths: list[Path],
        conda_env: str | None,
        session_name: str,
        env: dict[str, str],
        wait_for_process: bool = True,
        what_if: bool = True,
    ):
        # All we need to do here is launch `python -m ll.local_sessions_runner` with the config paths as arguments. The `local_sessions_runner` will take care of the rest.
        # Obviously, the command above needs to be run in a screen session, so we can come back to it later.

        if not conda_env:
            command = (
                ["screen", "-dmS", session_name]
                + ["python", "-m", "ll.local_sessions_runner"]
                + [str(p.absolute()) for p in config_paths]
            )
        else:
            command = (
                ["screen", "-dmS", session_name]
                + [
                    "conda",
                    "run",
                    "-n",
                    conda_env,
                    "python",
                    "-m",
                    "ll.local_sessions_runner",
                ]
                + [str(p.absolute()) for p in config_paths]
            )
        if not what_if:
            log.critical(f"Launching session with command: {command}")
            process = subprocess.Popen(command, env=env)
            if wait_for_process:
                _ = process.wait()

        return command

    def local_sessions(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        sessions: list[dict[str, str]],
        config_pickle_save_path: Path | None = None,
        reset_id: bool = True,
        what_if: bool = True,
    ):
        """
        Launches len(sessions) local runs in different environments using `screen`.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to launch.
        sessions : list[dict[str, str]]
            A list of environment variables to use for each session.
        reset_id : bool, optional
            Whether to reset the id of the runs before launching them.

        Returns
        -------
        list[TReturn]
            A list of names for each screen session.
        """

        # This only works in conda environments, so we need to make sure we're in one
        if (current_env := os.environ.get("CONDA_DEFAULT_ENV")) is None:
            raise RuntimeError("This function only works in conda environments.")

        if config_pickle_save_path is None:
            config_pickle_save_path = Path(tempfile.mkdtemp())

        resolved_runs = self._resolve_runs(runs, reset_id=reset_id)
        self._validate_runs(resolved_runs)

        # Save all configs to pickle files
        config_paths: list[Path] = []
        for i, config in enumerate(resolved_runs):
            config_path = config_pickle_save_path / f"ll_{i:03d}.pkl"
            config_paths.append(config_path)
            config = tuple([config[0], *config[1]])
            with config_path.open("wb") as f:
                pickle.dump((self._run, self._init_kwargs, config), f)

        # Launch all sessions
        names: list[str] = []
        commands: list[str] = []
        n_sessions = len(sessions)
        for i, session in enumerate(sessions):
            session_env = {**self.DEFAULT_ENV, **session}
            # Get the shared project name
            project_names = set([config.project for config, _ in resolved_runs])
            if len(project_names) == 1:
                project = project_names.pop()
            else:
                project = "session"
            session_name = f"ll_{project}_{i:03d}"
            command = self._launch_session(
                config_paths,
                current_env,
                session_name,
                session_env,
                what_if=what_if,
            )
            names.append(session_name)
            if what_if:
                command_str = " ".join(command)
                log.critical(f"Sesssion {i+1}/{n_sessions} command: {command_str}")
                commands.append(command_str)
            else:
                log.critical(f"Launched session {i+1}/{n_sessions}")

        if what_if:
            # Print the full command so the user can copy-paste it
            print(
                "The sessions were not launched because `what_if` was set. Please copy-paste the following command to launch the sessions."
            )
            for command in commands:
                print(command)

        return names

    def local_session_per_gpu(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        config_pickle_save_path: Path | None = None,
        reset_id: bool = True,
        what_if: bool = True,
    ):
        # Get the number of GPUs
        import torch

        n_gpus = torch.cuda.device_count()
        log.critical(f"Detected {n_gpus} GPUs. Launching one session per GPU.")

        # Create a session for each GPU
        sessions = [{"CUDA_VISIBLE_DEVICES": str(i)} for i in range(n_gpus)]

        # Launch the sessions
        return self.local_sessions(
            runs,
            sessions,
            config_pickle_save_path=config_pickle_save_path,
            reset_id=reset_id,
            what_if=what_if,
        )

    def fast_dev_run(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        env: dict[str, str] | None = None,
        n_batches: int = 1,
        devices: str | int | None = 1,
        stop_on_error: bool = True,
    ):
        """
        Runs a list of configs locally w/ `LightningTrainer.fast_dev_run = True`.
        """
        resolved_runs = self._resolve_runs(runs)
        self._validate_runs(resolved_runs)

        return_values: list[TReturn] = []

        for config, args in tqdm(resolved_runs, desc="Fast dev run"):
            run_id = config.id
            run_name = config.name
            try:
                config.trainer.fast_dev_run = n_batches
                if devices is not None:
                    config.trainer.devices = devices
                return_value = self.local((config, *args), env=env, reset_id=True)
                return_values.append(return_value)
            except BaseException as e:
                # print full traceback
                log.critical(f"Error in run with {run_id=} ({run_name=}): {e}")
                traceback.print_exc()
                if stop_on_error:
                    raise

        return return_values

    @staticmethod
    def _validate_runs(runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]]):
        if not runs:
            raise ValueError("No run configs provided.")

        id_counter = Counter(config.id for config, _ in runs if config.id is not None)
        for id, count in id_counter.items():
            if count > 1:
                raise ValueError(f"Duplicate id {id=}")

    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def submit(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        *,
        gpus: int,
        nodes: int,
        partition: str,
        cpus_per_task: int,
        snapshot: bool | Path,
        constraint: str | None = None,
        timeout: timedelta | None = None,
        memory: int | None = None,
        email: str | None = None,
        slurm_additional_parameters: dict[str, str] | None = None,
        slurm_setup: list[str] | None = None,
        snapshot_base: Path | None = None,
        env: dict[str, str] | None = None,
    ):
        resolved_runs = self._resolve_runs(runs)
        self._validate_runs(resolved_runs)

        if snapshot_base is None:
            current_user = getpass.getuser()
            snapshot_base = Path(f"/checkpoint/{current_user}/ll_snapshots/")

        if snapshot is True:
            snapshot = snapshot_modules(
                snapshot_base, ["st", "ll", "submitit"]
            ).absolute()

        env = {**self.DEFAULT_ENV, **(env or {})}

        base_path = Path(".") / "slurm_logs"
        base_path.mkdir(exist_ok=True, parents=True)

        additional_parameters = {}
        if email:
            additional_parameters.update({"mail_user": email, "mail_type": "FAIL"})
        if constraint:
            additional_parameters.update({"constraint": constraint})
        if slurm_additional_parameters:
            additional_parameters.update(slurm_additional_parameters)

        setup = []
        if env:
            setup.extend(f"export {k}={v}" for k, v in env.items())
        if slurm_setup:
            setup.extend(slurm_setup)
        if snapshot:
            snapshot_str = str(snapshot.resolve().absolute())
            setup.append(f"export {self.SNAPSHOT_ENV_NAME}={snapshot_str}")
            setup.append(f"export PYTHONPATH={snapshot_str}:$PYTHONPATH")

        parameters_kwargs = dict(
            name=self.slurm_job_name,
            mem_gb=memory,
            cpus_per_task=cpus_per_task,
            tasks_per_node=gpus,
            gpus_per_node=gpus,
            nodes=nodes,
            slurm_partition=partition,
            slurm_additional_parameters=additional_parameters,
            slurm_setup=setup,
        )
        if timeout:
            parameters_kwargs["timeout_min"] = int(timeout.total_seconds() / 60)

        executor = AutoExecutor(folder=base_path / "%j")
        executor.update_parameters(**parameters_kwargs)

        map_array_args = list(zip(*[(c, *args) for c, args in resolved_runs]))
        log.critical(f"Submitting {len(resolved_runs)} jobs to {partition}.")
        jobs = executor.map_array(self.run, *map_array_args)
        for job, (config, _) in zip(jobs, resolved_runs):
            log.critical(f"[id={config.id}] Submitted job: {job.job_id} to {partition}")
        return jobs
