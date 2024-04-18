import os
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import Any, Protocol, overload

from typing_extensions import TypeAlias, TypedDict, TypeVarTuple, Unpack

from ...picklerunner import serialize_many, serialize_single

log = getLogger(__name__)

DEFAULT_JOB_NAME = "sl"
DEFAULT_NODES = 1
DEFAULT_WALLTIME = timedelta(hours=2)

TArgs = TypeVarTuple("TArgs")

_Path: TypeAlias = str | Path | os.PathLike


class CommandPrefixFnProtocol(Protocol):
    def __call__(self, num_nodes: int) -> str: ...


class SLURMJobKwargs(TypedDict, total=False):
    name: str
    """
    The name of the job.

    This corresponds to the "--job-name" option in sbatch.
    """

    partition: str
    """
    The partition to submit the job to.

    This corresponds to the "--partition" option in sbatch. If not specified, the default partition will be used.
    """

    output_file: _Path
    """
    The file to write the job output to.

    This corresponds to the "--output" option in sbatch. If not specified, the output will be written to the default output file.
    """

    error_file: _Path
    """
    The file to write the job errors to.

    This corresponds to the "--error" option in sbatch. If not specified, the errors will be written to the default error file.
    """

    walltime: timedelta
    """
    The maximum walltime for the job.

    This corresponds to the "--time" option in sbatch. The format is "days-hours:minutes:seconds". If not specified, the default walltime will be used.
    """

    memory_mb: int
    """
    The maximum memory for the job in MB.

    This corresponds to the "--mem" option in sbatch. If not specified, the default memory limit will be used.
    """

    cpus_per_task: int
    """
    The number of CPUs per task.

    This corresponds to the "--cpus-per-task" option in sbatch. If not specified, the default value will be used.
    """

    requeue: bool
    """
    Whether the job should be requeued if it fails.

    This corresponds to the "--requeue" option in sbatch. If set to True, the job will be requeued if it fails.
    """

    dependency_conditions: Sequence[str]
    """
    The dependency conditions for the job.

    This corresponds to the "--dependency" option in sbatch. Each condition is a string that specifies the dependency condition.
    Multiple conditions can be specified, and they will be combined using logical AND.
    """

    email: str
    """
    The email address to send notifications to.

    This corresponds to the "--mail-user" option in sbatch. If specified, job notifications will be sent to this email address.
    """

    notify_begin: bool
    """
    Whether to send a notification when the job begins.

    This corresponds to the "--mail-type=BEGIN" option in sbatch. If set to True, a notification will be sent when the job begins.
    """

    notify_end: bool
    """
    Whether to send a notification when the job ends.

    This corresponds to the "--mail-type=END" option in sbatch. If set to True, a notification will be sent when the job ends.
    """

    setup_commands: Sequence[str]
    """
    The setup commands to run before the job.

    These commands will be executed prior to everything else in the job script.
    """

    environment: Mapping[str, str]
    """
    The environment variables to set for the job.

    These variables will be set prior to executing any commands in the job script.
    """

    account: str
    """
    The account to charge the job to.

    This corresponds to the "--account" option in sbatch. If specified, the job will be charged to this account.
    """

    nodes: int
    """
    The number of nodes to use for the job.

    This corresponds to the "--nodes" option in sbatch. The default is 1 node.
    """

    gpus: int
    """
    The number of GPUs to use for the job.

    This corresponds to the "--gpus" option in sbatch. If specified, the job will request this number of GPUs.
    """

    constraint: str
    """
    The constraint to apply to the job.

    This corresponds to the "--constraint" option in sbatch. If specified, the job will be allocated nodes that satisfy this constraint.
    """

    command_prefix: str | CommandPrefixFnProtocol
    """
    A command to prefix the job command with.

    This is used to add commands like `srun` to the job command.
    """


def _write_batch_script_to_file(
    path: Path,
    kwargs: SLURMJobKwargs,
    command: str,
    job_array_n_jobs: int | None = None,
):
    logs_base = path.parent / "logs"
    logs_base.mkdir(exist_ok=True)

    if kwargs.get("output_file") is None:
        kwargs["output_file"] = logs_base / "output_%j.out"

    if kwargs.get("error_file") is None:
        kwargs["error_file"] = logs_base / "error_%j.err"

    with path.open("w") as f:
        f.write("#!/bin/bash\n")

        name = kwargs.get("name", DEFAULT_JOB_NAME)
        if job_array_n_jobs is not None:
            name += "[1-" + str(job_array_n_jobs) + "]"
        f.write(f"#SBATCH --job-name={name}\n")

        if (account := kwargs.get("account")) is not None:
            f.write(f"#SBATCH --account={account}\n")

        if (walltime := kwargs.get("walltime", DEFAULT_WALLTIME)) is not None:
            # Convert the walltime to the format expected by SLURM:
            # days-hours:minutes:seconds
            total_seconds = walltime.total_seconds()
            days = int(total_seconds // (24 * 3600))
            total_seconds %= 24 * 3600
            hours = int(total_seconds // 3600)
            total_seconds %= 3600
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            walltime = f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
            f.write(f"#SBATCH --time={walltime}\n")

        if (nodes := kwargs.get("nodes", DEFAULT_NODES)) is not None:
            f.write(f"#SBATCH --nodes={nodes}\n")

        if (output_file := kwargs.get("output_file")) is not None:
            output_file = Path(output_file).absolute()
            output_file = str(output_file)
            f.write(f"#SBATCH --output={output_file}\n")

        if (error_file := kwargs.get("error_file")) is not None:
            error_file = Path(error_file).absolute()
            error_file = str(error_file)
            f.write(f"#SBATCH --error={error_file}\n")

        if (partition := kwargs.get("partition")) is not None:
            f.write(f"#SBATCH --partition={partition}\n")

        if (memory_mb := kwargs.get("memory_mb")) is not None:
            f.write(f"#SBATCH --mem={memory_mb}\n")

        if (cpus_per_task := kwargs.get("cpus_per_task")) is not None:
            f.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")

        if (requeue := kwargs.get("requeue")) is not None:
            f.write(f"#SBATCH --requeue={'yes' if requeue else 'no'}\n")

        for dependency_condition in kwargs.get("dependency_conditions", []):
            f.write(f"#SBATCH --dependency={dependency_condition}\n")

        if (email := kwargs.get("email")) is not None:
            f.write(f"#SBATCH --mail-user={email}\n")

        mail_type = ""
        if (notify_begin := kwargs.get("notify_begin")) is not None and notify_begin:
            mail_type += "BEGIN,"
        if (notify_end := kwargs.get("notify_end")) is not None and notify_end:
            mail_type += "END"
        if mail_type:
            f.write(f"#SBATCH --mail-type={mail_type}\n")

        if (gpus := kwargs.get("gpus")) is not None:
            f.write(f"#SBATCH --gpus={gpus}\n")

        if (constraint := kwargs.get("constraint")) is not None:
            f.write(f"#SBATCH --constraint={constraint}\n")

        f.write("\n")

        for key, value in kwargs.get("environment", {}).items():
            f.write(f"export {key}={value}\n")

        f.write("\n")

        setup_commands = kwargs.get("setup_commands", [])
        for setup_command in setup_commands:
            f.write(f"{setup_command}\n")

        f.write("\n")

        if (command_prefix := kwargs.get("command_prefix")) is not None:
            if callable(command_prefix):
                command_prefix = command_prefix(nodes)

            command = " ".join(
                x_stripped
                for x in (command_prefix, command)
                if (x_stripped := x.strip())
            )
        f.write(f"{command}\n")

    return path


def _update_kwargs(kwargs: SLURMJobKwargs):
    # Update the kwargs with the default values
    kwargs = {**kwargs}

    return kwargs


@overload
def to_batch_script(
    dest: Path, command: str, /, **kwargs: Unpack[SLURMJobKwargs]
) -> Path: ...


@overload
def to_batch_script(
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args: tuple[Unpack[TArgs]],
    /,
    **kwargs: Unpack[SLURMJobKwargs],
) -> Path: ...


def to_batch_script(
    dest: Path,
    command_or_callable,
    args=None,
    /,
    **kwargs: Unpack[SLURMJobKwargs],
):
    """
    Create the batch script for the job.
    """

    kwargs = _update_kwargs(kwargs)

    # Convert the command/callable to a string for the command
    command: str
    if isinstance(command_or_callable, str):
        command = command_or_callable
    elif callable(command_or_callable):
        assert args is not None, "Expected args to be provided for callable"
        command = serialize_single(
            dest / "fn.pkl",
            command_or_callable,
            args,
            {},
        ).to_command_str()
    else:
        raise TypeError(f"Expected str or callable, got {type(command_or_callable)}")

    return _write_batch_script_to_file(dest, kwargs, command)


@overload
def to_array_batch_script(
    dest: Path,
    command: str,
    num_jobs: int,
    /,
    **kwargs: Unpack[SLURMJobKwargs],
) -> Path: ...


@overload
def to_array_batch_script(
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args_list: Sequence[tuple[Unpack[TArgs]]],
    /,
    job_index_variable: str = "SLURM_ARRAY_TASK_ID",
    **kwargs: Unpack[SLURMJobKwargs],
) -> Path: ...


def to_array_batch_script(
    dest: Path,
    command_or_callable,
    args_list_or_num_jobs,
    /,
    job_index_variable: str = "SLURM_ARRAY_TASK_ID",
    **kwargs: Unpack[SLURMJobKwargs],
):
    """
    Create the batch script for the job.
    """

    kwargs = _update_kwargs(kwargs)

    # Convert the command/callable to a string for the command
    command: str
    num_jobs: int
    if isinstance(command_or_callable, str):
        command = command_or_callable
        assert isinstance(
            args_list_or_num_jobs, int
        ), "Expected num_jobs to be provided for command"
        num_jobs = args_list_or_num_jobs
    elif callable(command_or_callable):
        assert isinstance(
            args_list_or_num_jobs, Sequence
        ), "Expected args_list to be provided for callable"
        args_list = args_list_or_num_jobs
        num_jobs = len(args_list)

        destdir = dest / "fns"
        destdir.mkdir(exist_ok=True)

        command = serialize_many(
            destdir,
            command_or_callable,
            [(args, {}) for args in args_list],
            start_idx=1,  # SLURM job indices are 1-based
        ).to_bash_command(job_index_variable)
    else:
        raise TypeError(f"Expected str or callable, got {type(command_or_callable)}")

    return _write_batch_script_to_file(
        dest / "launch.sh",
        kwargs,
        command,
        job_array_n_jobs=num_jobs,
    )
