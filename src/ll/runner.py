import contextlib
import copy
import os
import sys
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeAlias, cast, runtime_checkable

from tqdm.auto import tqdm
from typing_extensions import TypedDict, TypeVar, TypeVarTuple, Unpack, override

from ._submit.session import unified
from .model.config import BaseConfig
from .trainer import Trainer
from .util.environment import (
    remove_lsf_environment_variables,
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
    snapshot_ll=False,
    snapshot_config_cls_module=True,
)


TConfig = TypeVar("TConfig", bound=BaseConfig, infer_variance=True)
TReturn = TypeVar("TReturn", default=None, infer_variance=True)
TArguments = TypeVarTuple("TArguments", default=Unpack[tuple[()]])


def _resolve_run(
    run: TConfig | tuple[TConfig, Unpack[TArguments]],
    copy_config: bool = True,
    reset_id: bool = False,
) -> tuple[TConfig, tuple[Unpack[TArguments]]]:
    if isinstance(run, tuple):
        (config, *args) = run
    else:
        config = cast(TConfig, run)
        args = ()
    args = cast(tuple[Unpack[TArguments]], args)
    if copy_config:
        config = copy.deepcopy(config)
    if reset_id:
        config.id = BaseConfig.generate_id(ignore_rng=True)
    return (config, args)


def _resolve_runs(
    runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
    copy_config: bool = True,
    reset_id: bool = False,
):
    resolved: list[tuple[TConfig, tuple[Unpack[TArguments]]]] = []
    for run in runs:
        resolved.append(_resolve_run(run, copy_config=copy_config, reset_id=reset_id))

    return resolved


def _validate_runs(runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]]):
    if not runs:
        raise ValueError("No run configs provided.")

    # Make sure there are no duplicate ids
    id_counter = Counter(config.id for config, _ in runs if config.id is not None)
    for id, count in id_counter.items():
        if count > 1:
            raise ValueError(f"Duplicate id {id=}")


@runtime_checkable
class RunProtocol(Protocol[TConfig, TReturn, Unpack[TArguments]]):
    def __call__(self, config: TConfig, *args: Unpack[TArguments]) -> TReturn: ...


@dataclass
class RunnerSession:
    env: Mapping[str, str]
    """The environment variables to use for the session."""

    name: str | None = None
    """The name of the session."""

    per_rank_env: Sequence[Mapping[str, str]] | None = None
    """The environment variables to set for each rank."""


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
        savedir: str | Path | os.PathLike | None = None,
        job_name: str = "ll",
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
        savedir : Path, optional
            The `savedir` parameter is a string that represents the directory where the program will save its execution files and logs.
            This is used when submitting the program to a SLURM/LSF cluster or when using the `local_sessions` method.
            If `None`, this will default to the current working directory / `llrunner`.
        job_name : str, optional
            The `job_name` parameter is a string that represents the name of the job when submitting it to a cluster.
        validate_config_before_run : bool, optional
            The `validate_config_before_run` parameter is a boolean that represents whether or not to validate the configuration before running the program.
        validate_strict: bool, optional
            Should `validate_config_before_run` be strict? If `True`, the configuration will be validated strictly. If `False`, the configuration will be validated non-strictly.
        """

        super().__init__()

        self._run = run
        self._savedir = savedir
        self.job_name = job_name
        self.validate_config_before_run = validate_config_before_run
        self.validate_strict = validate_strict
        self._init_kwargs = {
            "savedir": savedir,
            "job_name": job_name,
            "validate_config_before_run": validate_config_before_run,
            "validate_strict": validate_strict,
        }
        self.env = {
            **self.DEFAULT_ENV,
            **(env or {}),
        }

    def _get_base_path(
        self,
        runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]] | None,
    ):
        # If the user has provided a `savedir`, use that as the base path.
        if self._savedir is not None:
            base_path = Path(self._savedir)
            base_path.mkdir(exist_ok=True, parents=True)
            return base_path

        # If all configs have the same `project_root` config, use that instead.
        project_root_paths = set(
            str(project_root.absolute())
            if (project_root := config.directory.project_root) is not None
            else None
            for config, _ in (runs or [])
        )
        if (
            project_root_paths
            and len(project_root_paths) == 1
            and (project_root_path := project_root_paths.pop()) is not None
        ):
            project_root_path = Path(project_root_path)
        else:
            project_root_path = Path.cwd()

        base_path = project_root_path / "llrunner"
        base_path.mkdir(exist_ok=True, parents=True)

        return base_path

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

    @contextlib.contextmanager
    def _with_env(self, env: Mapping[str, str]):
        env_old = {k: os.environ.get(k, None) for k in env}
        os.environ.update(env)
        try:
            yield
        finally:
            for new_env_key in env.keys():
                # If we didn't have the key before, remove it
                if (old_value := env_old.get(new_env_key)) is None:
                    _ = os.environ.pop(new_env_key, None)
                else:
                    os.environ[new_env_key] = old_value

    def local(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        env: Mapping[str, str] | None = None,
        reset_id: bool = False,
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
            config, args = _resolve_run(run, reset_id=reset_id)
            with self._with_env(env or {}):
                return_value = self._run_fn(config, *args)
                return_values.append(return_value)

        return return_values

    def _launch_session(
        self,
        session_command: list[str],
        config_base_path: Path,
        session_name: str,
    ):
        return [
            "screen",
            "-dmS",
            session_name,
            # Save the logs to a file
            "-L",
            "-Logfile",
            str((config_base_path / f"{session_name}.log").absolute()),
            # Enable UTF-8 encoding
            "-U",
            *session_command,
        ]

    def local_sessions(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        sessions: int | Sequence[Mapping[str, str]] | Sequence[RunnerSession],
        name: str = "ll",
        config_pickle_save_path: Path | None = None,
        reset_id: bool = False,
        snapshot: bool | SnapshotConfig = False,
        delete_run_script_after_launch: bool = False,
        prologue: Sequence[str] | None = None,
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

        # If `env` is set, just add it to the prologues
        if env:
            if prologue is None:
                prologue = []

            # Prepend so env takes precedence
            prologue = list(prologue)
            prologue = [f"export {k}={v}" for k, v in env.items()] + prologue

        if isinstance(sessions, int):
            sessions = [{} for _ in range(sessions)]

        resolved_runs = _resolve_runs(runs, reset_id=reset_id)
        _validate_runs(resolved_runs)
        local_data_path = self._local_data_path(id, resolved_runs)

        # Take a snapshot of the environment
        snapshot_path = self._snapshot(snapshot, resolved_runs, local_data_path)

        # Save all configs to pickle files
        from .picklerunner import serialize_many

        if config_pickle_save_path is None:
            config_pickle_save_path = local_data_path / "sessions"
            config_pickle_save_path.mkdir(exist_ok=True)
        serialized = serialize_many(
            config_pickle_save_path,
            _runner_main,
            [
                ((self._run, self._init_kwargs, c, args), {})
                for c, args in resolved_runs
            ],
        )

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

        # Get the world size for each session
        session_per_rank_envs: list[Sequence[Mapping[str, str]] | None] = []
        for session in sessions:
            if not isinstance(session, RunnerSession):
                session_per_rank_envs.append(None)
            else:
                session_per_rank_envs.append(session.per_rank_env)

        # Get the session commands
        session_commands = serialized.to_bash_command_sequential_workers(
            num_workers=len(session_names)
        )

        # Launch all sessions
        commands: list[str] = []

        def _session_world_size_to_env(
            session_name: str,
            per_rank_env: Sequence[Mapping[str, str]] | None,
        ) -> list[tuple[str, Mapping[str, str]]]:
            if not per_rank_env:
                return [(session_name, {})]
            return [
                (f"{session_name}_rank{i}", {**env})
                for i, env in enumerate(per_rank_env)
            ]

        for i, (
            session_env,
            session_name,
            session_command,
            session_per_rank_env,
        ) in enumerate(
            zip(session_envs, session_names, session_commands, session_per_rank_envs)
        ):
            for rank_session_name, additional_env in _session_world_size_to_env(
                session_name,
                session_per_rank_env,
            ):
                command = self._launch_session(
                    session_command,
                    config_pickle_save_path,
                    rank_session_name,
                )
                # log.critical(f"Sesssion {i+1}/{n_sessions} command: {command_str}")
                command_prefix = " ".join(
                    f'{k}="{v}"' for k, v in {**session_env, **additional_env}.items()
                )
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

            # Activate the environment
            # Let's detect the environment: If we're in a pixi environment,
            #   use pixi's shell hook instead.
            if "/.pixi/" in sys.prefix:
                f.write('eval "$(pixi shell-hook)"\n')
                f.write(f'echo "Activating pixi environment {sys.prefix}"\n\n')
            else:
                # Otherwise, assume we're in a conda environment.
                f.write('eval "$(conda shell.bash hook)"\n')
                f.write(f'echo "Activating conda environment {sys.prefix}"\n')
                f.write(f"conda activate {sys.prefix}\n\n")

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
        reset_id: bool = False,
        snapshot: bool | SnapshotConfig = False,
        prologue: list[str] | None = None,
        env: Mapping[str, str] | None = None,
        separate_session_per_task: bool = True,
        throw_on_gpu_index_error: bool = True,
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
        separate_session_per_task : bool, optional
            If `True`, each GPU task will be run in a separate screen session with the `LOCAL_RANK` environment variable set to the task index.
        throw_on_gpu_index_error: bool, optional
            If `True`, an error will be raised if an invalid GPU index is provided.

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
        if throw_on_gpu_index_error:
            for gpu_idxs in gpus_dict:
                for gpu_idx in gpu_idxs:
                    if gpu_idx not in all_gpus:
                        raise ValueError(
                            f"GPU {gpu_idx} is not available. Available GPUs: {all_gpus}"
                        )

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

                per_rank_env = None
                if separate_session_per_task:
                    world_size = len(gpu_idxs)
                    per_rank_env = [
                        self._gpu_env(i, world_size) for i in range(world_size)
                    ]
                sessions.append(
                    RunnerSession(session_env, session_name, per_rank_env=per_rank_env)
                )

        if separate_session_per_task:
            # Now that we know the total number of concurrent sessions, we can find
            #   that many open ports for the sessions to use.
            ports = list(self._find_n_free_ports(len(sessions)))

            # Update the sessions with the port information.
            sessions = [
                replace(
                    session,
                    env={
                        **session.env,
                        "MASTER_ADDR": "127.0.0.1",
                        "MASTER_PORT": str(port),
                    },
                )
                for session, port in zip(sessions, ports)
            ]

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

    @staticmethod
    def _find_n_free_ports(n: int):
        import socket

        with contextlib.ExitStack() as stack:
            for _ in range(n):
                s = stack.enter_context(
                    contextlib.closing(
                        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    )
                )
                s.bind(("", 0))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                yield s.getsockname()[1]

    @staticmethod
    def _gpu_env(local_rank: int, world_size: int):
        return {
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
        }

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
        resolved_runs = _resolve_runs(runs, copy_config=True)
        _validate_runs(resolved_runs)

        return_values: list[TReturn] = []
        for config, args in tqdm(resolved_runs, desc="Fast dev run"):
            run_id = config.id
            run_name = config.run_name
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

    def _local_data_path(
        self,
        id: str,
        runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]] | None = None,
    ) -> Path:
        # First, resolve the base path.
        base_path = self._get_base_path(runs)
        base_path.mkdir(parents=True, exist_ok=True)

        # Add a gitignore file to the directory so that the entire directory is ignored by git
        gitignore_path = base_path / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.touch()
            gitignore_path.write_text("*\n")

        local_data_path = base_path / id
        local_data_path.mkdir(exist_ok=True)

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

    @remove_lsf_environment_variables()
    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def submit(
        self,
        runs: Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]],
        *,
        scheduler: unified.Scheduler | Literal["auto"] = "auto",
        snapshot: bool | SnapshotConfig = False,
        reset_id: bool = False,
        activate_conda: bool = True,
        env: Mapping[str, str] | None = None,
        **job_kwargs: Unpack[unified.GenericJobKwargs],
    ):
        """
        Submits a list of runs to a cluster (SLURM or LSF).

        Parameters
        ----------
        runs : Sequence[TConfig] | Sequence[tuple[TConfig, Unpack[TArguments]]]
            A sequence of runs to submit.
        scheduler : str, optional
            The scheduler to use. If `auto`, the scheduler will be inferred.
        snapshot : bool | Path
            The base path to save snapshots to. If `True`, a default path will be used.
        reset_id : bool, optional
            Whether to reset the id of the runs before launching them.
        activate_conda : bool, optional
            Whether to activate the conda environment before running the jobs.
        env : Mapping[str, str], optional
            Additional environment variables to set.
        job_kwargs : dict
            Additional keyword arguments to pass to the job submission script.
        """
        if scheduler == "auto":
            scheduler = unified.infer_current_scheduler()
            log.critical(f"Inferred current scheduler as {scheduler}")

        id = str(uuid.uuid4())

        resolved_runs = _resolve_runs(runs, reset_id=reset_id)
        _validate_runs(resolved_runs)
        local_data_path = self._local_data_path(id, resolved_runs)

        setup_commands = list(job_kwargs.get("setup_commands", []))

        # Handle snapshot
        snapshot_path = self._snapshot(snapshot, resolved_runs, local_data_path)
        if snapshot_path:
            snapshot_str = str(snapshot_path.resolve().absolute())
            setup_commands.append(f"export {self.SNAPSHOT_ENV_NAME}={snapshot_str}")
            setup_commands.append(f"export PYTHONPATH={snapshot_str}:$PYTHONPATH")

        # Conda environment
        if activate_conda:
            # Activate the conda environment
            setup_commands.append('eval "$(conda shell.bash hook)"')
            setup_commands.append(f"echo 'Activating conda environment {sys.prefix}'")
            setup_commands.append(f"conda activate {sys.prefix}")

        job_kwargs["environment"] = {
            **self.env,
            **job_kwargs.get("environment", {}),
            **(env or {}),
        }
        job_kwargs["setup_commands"] = setup_commands

        base_path = local_data_path / "submit"
        base_path.mkdir(exist_ok=True, parents=True)

        # Update submission kwargs based on the configs
        job_kwargs = unified.update_kwargs_from_configs(
            job_kwargs, [config for config, _ in resolved_runs]
        )

        # Serialize the runs
        map_array_args: list[
            tuple[
                RunProtocol[TConfig, TReturn, Unpack[TArguments]],
                Mapping[str, Any],
                TConfig,
                tuple[Unpack[TArguments]],
            ]
        ] = [(self._run, self._init_kwargs, c, args) for c, args in resolved_runs]
        submission = unified.to_array_batch_script(
            scheduler,
            base_path,
            _runner_main,
            map_array_args,
            **job_kwargs,
        )
        print("Please run the following command to submit the jobs:")
        print(submission.submission_command_str)


# First, let's create the function that's going to be run on the cluster.
def _runner_main(
    run_fn: RunProtocol[TConfig, TReturn, Unpack[TArguments]],
    runner_kwargs: Mapping[str, Any],
    config: TConfig,
    args: tuple[Unpack[TArguments]],
):
    # Create the runner
    runner = Runner(run_fn, **runner_kwargs)

    # Run the function and return the result
    return_values = runner.local([(config, *args)])
    assert len(return_values) == 1
    return return_values[0]
