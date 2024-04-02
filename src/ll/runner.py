import copy
import os
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Generic, Protocol, TypeAlias, TypedDict, cast, runtime_checkable

import cloudpickle as pickle
from tqdm.auto import tqdm
from typing_extensions import TypeVar, TypeVarTuple, Unpack, override

from ll.submitit import AutoExecutor

from .model.config import BaseConfig
from .trainer import Trainer
from .util.environment import (
    remove_slurm_environment_variables,
    remove_wandb_environment_variables,
)
from .util.snapshot import snapshot_modules

log = getLogger(__name__)


class SnapshotConfig(TypedDict, total=False):
    dir: Path
    """The directory to save snapshots to. Default: `{cwd}/ll-{id}/snapshot`."""

    snapshot_ll: bool
    """Whether to snapshot the `ll` module. Default: `True`."""

    snapshot_config_cls_module: bool
    """Whether to snapshot the module of the config class. Default: `True`."""

    modules: list[str]
    """Additional modules to snapshot. Default: `[]`."""


SNAPSHOT_CONFIG_DEFAULT = SnapshotConfig(
    snapshot_ll=True,
    snapshot_config_cls_module=True,
)


TConfig = TypeVar("TConfig", bound=BaseConfig, infer_variance=True)
TReturn = TypeVar("TReturn", default=None, infer_variance=True)
TArguments = TypeVarTuple("TArguments", default=Unpack[tuple[()]])


@runtime_checkable
class RunProtocol(Protocol[TConfig, TReturn, Unpack[TArguments]]):
    def __call__(self, config: TConfig, *args: Unpack[TArguments]) -> TReturn: ...


@dataclass
class RunnerSession:
    env: Mapping[str, str]
    """The environment variables to use for the session."""

    name: str | None = None
    """The name of the session."""


SessionGPUIndex: TypeAlias = int | tuple[int, ...]


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
        env: Mapping[str, str] | None = None,
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
        self.env = {
            **self.DEFAULT_ENV,
            **(env or {}),
        }

    @property
    def _run_fn(self) -> RunProtocol[TConfig, TReturn, Unpack[TArguments]]:
        run = self._run

        @wraps(run)
        def wrapped_run(config: TConfig, *args: Unpack[TArguments]) -> TReturn:
            nonlocal self

            with ExitStack() as stack:
                nonlocal run

                # If `auto_call_trainer_init_from_runner`, we call `Trainer.runner_init` before running the program.
                if config.runner.auto_call_trainer_init_from_runner:
                    stack.enter_context(Trainer.runner_init(config))

                # If `validate_config_before_run`, we validate the configuration before running the program.
                if self.validate_config_before_run:
                    config = config.model_deep_validate(strict=self.validate_strict)

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

    def local(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        env: Mapping[str, str] | None = None,
        reset_id: bool = True,
    ):
        """
        Runs a list of configs locally.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        env : Mapping[str, str], optional
            Additional environment variables to set.
        reset_id : bool, optional
            Whether to reset the id of the runs before launching them.
        """
        return_values: list[TReturn] = []
        for run in runs:
            config, args = self._resolve_run(run)
            if reset_id:
                config.id = BaseConfig.generate_id(ignore_rng=True)

            env = {**self.env, **(env or {})}
            env_old = {k: os.environ.get(k, None) for k in env}
            os.environ.update(env)
            try:
                return_value = self._run_fn(config, *args)
                return_values.append(return_value)
            finally:
                for k, v in env_old.items():
                    if v is None:
                        _ = os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        return return_values

    def _launch_session(
        self,
        config_paths: list[Path],
        config_base_path: Path,
        session_name: str,
    ):
        # All we need to do here is launch `python -m ll.local_sessions_runner`
        # with the config paths as arguments. The `local_sessions_runner` will take care of the rest.
        # Obviously, the command above needs to be run in a screen session, so we can come back to it later.
        return (
            [
                "screen",
                "-dmS",
                session_name,
                # Save the logs to a file
                "-L",
                "-Logfile",
                str((config_base_path / f"{session_name}.log").absolute()),
                # Enable UTF-8 encoding
                "-U",
            ]
            + ["python", "-m", "ll.local_sessions_runner"]
            + [str(p.absolute()) for p in config_paths]
        )

    def local_sessions(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        sessions: int | list[Mapping[str, str]] | list[RunnerSession],
        name: str = "ll",
        config_pickle_save_path: Path | None = None,
        reset_id: bool = True,
        snapshot: bool | SnapshotConfig = False,
        delete_run_script_after_launch: bool = False,
        prologue: list[str] | None = None,
        env: Mapping[str, str] | None = None,
    ):
        """
        Launches len(sessions) local runs in different environments using `screen`.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to launch.
        sessions : list[Mapping[str, str]]
            A list of environment variables to use for each session.
        name : str, optional
            The name of this job. This name is pre-pended to the `screen` session names.
        config_pickle_save_path : Path, optional
            The path to save the config pickles to. If `None`, a temporary directory will be created.
        reset_id : bool, optional
            Whether to reset the id of the runs before launching them.
        snapshot : bool | SnapshotConfig
            Whether to snapshot the environment before launching the sessions.
        delete_run_script_after_launch : bool, optional
            Whether to delete the run shell script after launching the sessions.
        prologue : list[str], optional
            A list of commands to run at the beginning of the shell script.
        env : Mapping[str, str], optional
            Additional environment variables to set.

        Returns
        -------
        list[TReturn]
            A list of names for each screen session.
        """

        # Generate a random ID for the job.
        # We'll use this ID for snapshotting, as well as for
        #   defining the name of the shell script that will launch the sessions.
        id = str(uuid.uuid4())
        local_data_path = self._local_data_path(id)

        # If `env` is set, just add it to the prologues
        if env:
            if prologue is None:
                prologue = []
            # Prepend so env takes precedence
            prologue = [f"export {k}={v}" for k, v in env.items()] + prologue

        if isinstance(sessions, int):
            sessions = [{} for _ in range(sessions)]

        # This only works in conda environments, so we need to make sure we're in one
        if (current_env := os.environ.get("CONDA_DEFAULT_ENV")) is None:
            raise RuntimeError("This function only works in conda environments.")

        if config_pickle_save_path is None:
            config_pickle_save_path = local_data_path / "sessions"
            config_pickle_save_path.mkdir(exist_ok=True)

        resolved_runs = self._resolve_runs(runs, reset_id=reset_id)
        self._validate_runs(resolved_runs)

        # Take a snapshot of the environment
        snapshot_path = self._snapshot(snapshot, resolved_runs, local_data_path)

        # Save all configs to pickle files
        config_paths: list[Path] = []
        for i, config in enumerate(resolved_runs):
            config_path = config_pickle_save_path / f"ll_{i:03d}.pkl"
            config_paths.append(config_path)
            config = tuple([config[0], *config[1]])
            with config_path.open("wb") as f:
                pickle.dump((self._run, self._init_kwargs, config), f)

        # Resolve all session names
        session_names: list[str] = []
        for i, session in enumerate(sessions):
            match session:
                case RunnerSession() if session.name:
                    session_name = session.name
                case _:
                    session_name = f"{i:03d}"

            # If the session name is already in use, add a number to it
            if session_name in session_names:
                j = 1
                while (new_name := f"{session_name}_{j}") in session_names:
                    j += 1
                session_name = new_name

            session_names.append(session_name)
        # Prepend the job name to the session names
        session_names = [f"{name}{n}" for n in session_names]

        # Resolve all session envs
        session_envs: list[Mapping[str, str]] = []
        for i, session in enumerate(sessions):
            match session:
                case RunnerSession():
                    session_env = session.env
                case _:
                    session_env = session

            session_envs.append({**self.env, **session_env})

        # Launch all sessions
        commands: list[str] = []

        for i, (session_env, session_name) in enumerate(
            zip(session_envs, session_names)
        ):
            # Get the projects assigned to this session
            session_config_paths = config_paths[i :: len(sessions)]

            # If this session has no configs, skip it
            if not session_config_paths:
                continue

            command = self._launch_session(
                session_config_paths,
                config_pickle_save_path,
                session_name,
            )

            # log.critical(f"Sesssion {i+1}/{n_sessions} command: {command_str}")
            command_prefix = " ".join(f'{k}="{v}"' for k, v in session_env.items())
            command_str = " ".join(command)
            commands.append(f"{command_prefix} {command_str}")

        # Create a shell script to launch all sessions
        script_path = config_pickle_save_path / "launch.sh"
        with script_path.open("w") as f:
            f.write("#!/bin/bash\n\n")

            # Enable error checking
            f.write("set -e\n\n")

            # If a prologue is provided, run it
            if prologue:
                f.write("# Prologue\n")
                for command in prologue:
                    f.write(f"{command}\n")
                f.write("\n")

            # If we're in a snapshot, we need to activate the snapshot before launching the sessions
            if snapshot_path:
                snapshot_str = str(snapshot_path.resolve().absolute())
                f.write('echo "Activating snapshot"\n')
                f.write(f"export PYTHONPATH={snapshot_str}:$PYTHONPATH\n")
                f.write(f"export {self.SNAPSHOT_ENV_NAME}={snapshot_str}\n\n")

            # Activate the conda environment
            f.write('eval "$(conda shell.bash hook)"\n')
            f.write(f'echo "Activating conda environment {current_env}"\n')
            f.write(f"conda activate {current_env}\n\n")

            # Launch the sessions
            for command in commands:
                f.write(f"{command}\n")

            # Delete the script after launching the sessions
            if delete_run_script_after_launch:
                f.write(f"\nrm {script_path}\n")

        # Make the script executable
        script_path.chmod(0o755)

        # Print the full command so the user can copy-paste it
        print(
            f"Run the following command to launch the sessions:\n\n{script_path.resolve()}\n\n"
        )

        return session_names

    @staticmethod
    def _available_gpus():
        # If `CUDA_VISIBLE_DEVICES` is set, we can just return those.
        try:
            if (env := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
                return [int(i) for i in env.split(",")]
        except ValueError:
            pass

        # Otherwise, get all available GPUs
        import torch

        return list(range(torch.cuda.device_count()))

    def local_session_per_gpu(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        name: str = "ll",
        gpus: Sequence[SessionGPUIndex] | Mapping[SessionGPUIndex, int] | None = None,
        num_jobs_per_gpu: int | None = None,
        config_pickle_save_path: Path | None = None,
        reset_id: bool = True,
        snapshot: bool | SnapshotConfig = False,
        prologue: list[str] | None = None,
        env: Mapping[str, str] | None = None,
    ):
        """
        Launches len(sessions) local runs in different environments using `screen`.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to launch.
        num_jobs_per_gpu : int, optional
            The number of jobs to launch per GPU. (default: 1)
        gpus : list[SessionGPUIndex] | dict[SessionGPUIndex, int] | None, optional
            The GPUs to use.
            - If a dictionary is provided, it should map GPU indices to the number of jobs to launch on that GPU.
            - If a list is provided, it should be a list of GPU indices to use. In this case, `num_jobs_per_gpu` will be used to determine the number of jobs to launch on each GPU.
        config_pickle_save_path : Path, optional
            The path to save the config pickles to. If `None`, a temporary directory will be created.
        reset_id : bool, optional
            Whether to reset the id of the runs before launching them.
        snapshot : bool | SnapshotConfig
            Whether to snapshot the environment before launching the sessions.
        name : str, optional
            The name of this job. This name is pre-pended to the `screen` session names.
        prologue : list[str], optional
            A list of commands to run at the beginning of the shell script.
        env : Mapping[str, str], optional
            Additional environment variables to set.

        Returns
        -------
        list[TReturn]
            A list of names for each screen session.
        """
        # Get available GPUs
        all_gpus = self._available_gpus()
        if gpus is None:
            gpus = all_gpus

        # Convert gpus to a dictionary which
        # maps gpu index to the number of jobs to launch on that gpu.
        if isinstance(gpus, Sequence):
            # If `num_jobs_per_gpu` is not provided, default to 1.
            n = num_jobs_per_gpu or 1
            gpus = {gpu: n for gpu in gpus}

        gpus_dict = {
            ((gpu_idx,) if not isinstance(gpu_idx, tuple) else gpu_idx): num_jobs
            for gpu_idx, num_jobs in gpus.items()
        }
        del gpus

        # Make sure all the requested GPUs are available
        for gpu_idxs in gpus_dict:
            for gpu_idx in gpu_idxs:
                if gpu_idx not in all_gpus:
                    raise ValueError(f"GPU {gpu_idx} is not available.")

        log.critical(f"Detected {len(gpus_dict)} GPUs; {gpus_dict=}.")

        # Create a session for each GPU
        sessions: list[RunnerSession] = []
        for gpu_idxs, num_jobs_per_gpu in gpus_dict.items():
            gpu_idxs_str = ",".join(str(i) for i in gpu_idxs)
            session_env = {"CUDA_VISIBLE_DEVICES": gpu_idxs_str}
            session_name_gpu = f"gpu{gpu_idxs_str}"
            for job_idx in range(num_jobs_per_gpu):
                session_name = session_name_gpu
                if num_jobs_per_gpu > 1:
                    session_name = f"{session_name}_job{job_idx}"
                sessions.append(RunnerSession(session_env, session_name))

        # Launch the sessions
        return self.local_sessions(
            runs,
            sessions,
            name=name,
            config_pickle_save_path=config_pickle_save_path,
            reset_id=reset_id,
            snapshot=snapshot,
            prologue=prologue,
            env=env,
        )

    def fast_dev_run(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        env: Mapping[str, str] | None = None,
        n_batches: int = 1,
        stop_on_error: bool = True,
        reset_memory_caches: bool = True,
    ):
        """
        Runs a list of configs locally with `LightningTrainer.fast_dev_run = True`.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        env : Mapping[str, str], optional
            Additional environment variables to set.
        n_batches : int, optional
            The number of batches to run for `fast_dev_run`.
        stop_on_error : bool, optional
            Whether to stop on error.
        reset_memory_caches : bool, optional
            Whether to reset memory caches after each run.
        """
        resolved_runs = self._resolve_runs(runs, copy_config=True)
        self._validate_runs(resolved_runs)

        return_values: list[TReturn] = []
        for config, args in tqdm(resolved_runs, desc="Fast dev run"):
            run_id = config.id
            run_name = config.name
            try:
                config.trainer.fast_dev_run = n_batches
                return_value = self.local([(config, *args)], env=env, reset_id=True)
                return_values.extend(return_value)
            except BaseException as e:
                # print full traceback
                log.critical(f"Error in run with {run_id=} ({run_name=}): {e}")
                traceback.print_exc()
                if stop_on_error:
                    raise
            finally:
                # After each run, we should reset memory/caches
                if reset_memory_caches:
                    self._reset_memory_caches()

        return return_values

    def _reset_memory_caches(self):
        import gc

        import torch

        # Clear the memory caches
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
        gc.collect()

    @staticmethod
    def _validate_runs(runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]]):
        if not runs:
            raise ValueError("No run configs provided.")

        # Make sure there are no duplicate ids
        id_counter = Counter(config.id for config, _ in runs if config.id is not None)
        for id, count in id_counter.items():
            if count > 1:
                raise ValueError(f"Duplicate id {id=}")

    def _local_data_path(self, id: str):
        local_data_path = Path.cwd() / f"ll_{id}"
        local_data_path.mkdir(exist_ok=True)

        # Add a gitignore file to the directory so that the entire directory is ignored by git
        with (local_data_path / ".gitignore").open("w") as f:
            f.write("*\n")

        return local_data_path

    def _snapshot(
        self,
        snapshot: bool | SnapshotConfig,
        resolved_runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]],
        local_data_path: Path,
    ):
        # Handle snapshot
        snapshot_config: SnapshotConfig | None = None
        if snapshot is True:
            snapshot_config = {**SNAPSHOT_CONFIG_DEFAULT}
        elif snapshot is False:
            snapshot_config = None
        elif isinstance(snapshot, Mapping):
            snapshot_config = {**SNAPSHOT_CONFIG_DEFAULT, **snapshot}

        del snapshot
        if snapshot_config is None:
            return None

        # Set the snapshot base to the user's home directory
        snapshot_dir = snapshot_config.get("dir", local_data_path / "ll_snapshot")
        snapshot_dir.mkdir(exist_ok=True, parents=True)

        snapshot_modules_set: set[str] = set()
        snapshot_modules_set.update(snapshot_config.get("modules", []))
        if snapshot_config.get("snapshot_ll", True):
            # Resolve ll by taking the module of the runner class
            ll_module = self.__class__.__module__.split(".", 1)[0]
            if ll_module != "ll":
                log.warning(
                    f"Runner class {self.__class__.__name__} is not in the 'll' module.\n"
                    "This is unexpected and may lead to issues with snapshotting."
                )
            snapshot_modules_set.add(ll_module)
        if snapshot_config.get("snapshot_config_cls_module", True):
            for config, _ in resolved_runs:
                # Resolve the root module of the config class
                # NOTE: We also must handle the case where the config
                #   class's module is "__main__" (i.e. the config class
                #   is defined in the main script).
                module = config.__class__.__module__
                if module == "__main__":
                    log.warning(
                        f"Config class {config.__class__.__name__} is defined in the main script.\n"
                        "Snapshotting the main script is not supported.\n"
                        "Skipping snapshotting of the config class's module."
                    )
                    continue

                # Make sure to get the root module
                module = module.split(".", 1)[0]
                snapshot_modules_set.add(module)

        snapshot_path = snapshot_modules(snapshot_dir, list(snapshot_modules_set))
        return snapshot_path.absolute()

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
        snapshot: bool | SnapshotConfig = False,
        constraint: str | None = None,
        timeout: timedelta | None = None,
        memory: int | None = None,
        email: str | None = None,
        slurm_additional_parameters: Mapping[str, str] | None = None,
        slurm_setup: list[str] | None = None,
        env: Mapping[str, str] | None = None,
    ):
        """
        Submits a list of configs to a SLURM cluster.

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        gpus : int
            The number of GPUs per node.
        nodes : int
            The number of nodes.
        partition : str
            The name of the partition to submit to.
        cpus_per_task : int
            The number of CPUs per task.
        snapshot : bool | Path
            The base path to save snapshots to.
                - If `True`, the default path will be used (`{cwd}/ll-{id}/snapshot`).
                - If `False`, no snapshots will be used.
        constraint : str, optional
            The name of the constraint to use.
        timeout : timedelta, optional
            The maximum time to run the job for.
        memory : int, optional
            The amount of memory to use.
        email : str, optional
            The email to send notifications to.
        slurm_additional_parameters : Mapping[str, str], optional
            Additional parameters to pass to the SLUR
        """
        id = str(uuid.uuid4())
        local_data_path = self._local_data_path(id)

        resolved_runs = self._resolve_runs(runs)
        self._validate_runs(resolved_runs)

        # Handle snapshot
        snapshot_path = self._snapshot(snapshot, resolved_runs, local_data_path)

        env = {**self.env, **(env or {})}

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
        if snapshot_path:
            snapshot_str = str(snapshot_path.resolve().absolute())
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
        jobs = executor.map_array(self._run_fn, *map_array_args)
        for job, (config, _) in zip(jobs, resolved_runs):
            log.critical(f"[id={config.id}] Submitted job: {job.job_id} to {partition}")
        return jobs
