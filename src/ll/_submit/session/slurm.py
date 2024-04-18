import copy
import os
from collections.abc import Callable, Sequence
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, overload

from typing_extensions import TypeAlias, TypedDict, TypeVarTuple, Unpack

from ...picklerunner import serialize_many, serialize_single

log = getLogger(__name__)

DEFAULT_JOB_NAME = "ll"
DEFAULT_NODES = 1
DEFAULT_TASKS = 1
DEFAULT_WALLTIME = timedelta(hours=2)
DEFAULT_SUMMIT = False

TArgs = TypeVarTuple("TArgs")

_Path: TypeAlias = str | Path | os.PathLike


class SlurmJobKwargs(TypedDict, total=False):
    name: str
    """
    The name of the job.

    This corresponds to the "-J" option in sbatch.
    """

    account: str
    """
    The account to charge resources used by this job to.

    This corresponds to the "-A" option in sbatch.
    """

    partition: str | Sequence[str]
    """
    The partition to submit the job to.

    This corresponds to the "-p" option in sbatch. If not specified, the default partition will be used.
    Multiple partitions can be specified, and they will be combined using logical OR.
    """

    output_file: _Path
    """
    The file to write the job output to.

    This corresponds to the "-o" option in sbatch. If not specified, the output will be written to the default output file.
    """

    error_file: _Path
    """
    The file to write the job errors to.

    This corresponds to the "-e" option in sbatch. If not specified, the errors will be written to the default error file.
    """

    walltime: timedelta
    """
    The maximum walltime for the job.

    This corresponds to the "-t" option in sbatch. The format is "HH:MM" or "MM". If not specified, the default walltime will be used.
    """

    memory_mb: int
    """
    The maximum memory for the job in MB.

    This corresponds to the "--mem" option in sbatch. If not specified, the default memory limit will be used.
    """

    memory_per_cpu_mb: int
    """
    The minimum memory required per usable allocated CPU.

    This corresponds to the "--mem-per-cpu" option in sbatch. If not specified, the default memory limit will be used.
    """

    memory_per_gpu_mb: int
    """
    The minimum memory required per allocated GPU.

    This corresponds to the "--mem-per-gpu" option in sbatch. If not specified, the default memory limit will be used.
    """

    cpus_per_task: int
    """
    Advise the Slurm controller that ensuing job steps will require _ncpus_ number of processors per task.

    This corresponds to the "-c" option in sbatch.
    """

    nodes: int
    """
    The number of nodes to use for the job.

    This corresponds to the "-N" option in sbatch. The default is 1 node.
    """

    ntasks: int
    """
    The number of tasks to use for the job.

    This corresponds to the "-n" option in sbatch. The default is 1 task.
    """

    constraint: str | Sequence[str]
    """
    Nodes can have features assigned to them by the Slurm administrator. Users can specify which of these features are required by their job using the constraint option.

    This corresponds to the "-C" option in sbatch.
    """

    gres: str | Sequence[str]
    """
    Specifies a comma-delimited list of generic consumable resources.

    This corresponds to the "--gres" option in sbatch.
    """

    gpus: str
    """
    Specify the total number of GPUs required for the job. An optional GPU type specification can be supplied.

    This corresponds to the "-G" option in sbatch.
    """

    gpus_per_node: str
    """
    Specify the number of GPUs required for the job on each node included in the job's resource allocation. An optional GPU type specification can be supplied.

    This corresponds to the "--gpus-per-node" option in sbatch.
    """

    gpus_per_task: str
    """
    Specify the number of GPUs required for the job on each task to be spawned in the job's resource allocation. An optional GPU type specification can be supplied.

    This corresponds to the "--gpus-per-task" option in sbatch.
    """

    mail_user: str
    """
    User to receive email notification of state changes as defined by mail_type.

    This corresponds to the "--mail-user" option in sbatch.
    """

    mail_type: (
        Literal[
            "NONE",
            "BEGIN",
            "END",
            "FAIL",
            "REQUEUE",
            "ALL",
            "INVALID_DEPEND",
            "STAGE_OUT",
            "TIME_LIMIT",
            "TIME_LIMIT_90",
            "TIME_LIMIT_80",
            "TIME_LIMIT_50",
            "ARRAY_TASKS",
        ]
        | Sequence[
            Literal[
                "NONE",
                "BEGIN",
                "END",
                "FAIL",
                "REQUEUE",
                "ALL",
                "INVALID_DEPEND",
                "STAGE_OUT",
                "TIME_LIMIT",
                "TIME_LIMIT_90",
                "TIME_LIMIT_80",
                "TIME_LIMIT_50",
                "ARRAY_TASKS",
            ]
        ]
    )
    """
    Notify user by email when certain event types occur.

    This corresponds to the "--mail-type" option in sbatch.
    """

    dependency: str
    """
    Defer the start of this job until the specified dependencies have been satisfied.

    This corresponds to the "-d" option in sbatch.
    """

    exclusive: bool
    """
    The job allocation can not share nodes with other running jobs.

    This corresponds to the "--exclusive" option in sbatch.
    """

    command_prefix: str
    """
    A command to prefix the job command with.

    This is used to add commands like `jsrun` to the job command.
    """

    # Our own custom options
    update_kwargs_fn: "Callable[[SlurmJobKwargs], SlurmJobKwargs]"
    """
    A function to update the kwargs with the defaults.

    This is useful for setting the command prefix to be dependent on num nodes/gpus/etc.
    """


def _append_job_index_to_path(path: Path) -> Path:
    # If job array, append the job index to the output file
    # E.g., if `output_file` is "output_%J.out", we want "output_%J_%I.out"
    stem = path.stem
    suffix = path.suffix
    new_stem = f"{stem}_%a"
    new_path = path.with_name(new_stem + suffix)
    return new_path


def _write_batch_script_to_file(
    path: Path,
    kwargs: SlurmJobKwargs,
    command: str,
    job_array_n_jobs: int | None = None,
):
    logs_base = path.parent / "logs"
    logs_base.mkdir(exist_ok=True)

    if kwargs.get("output_file") is None:
        kwargs["output_file"] = logs_base / "output_%A_%a.out"

    if kwargs.get("error_file") is None:
        kwargs["error_file"] = logs_base / "error_%A_%a.err"

    with path.open("w") as f:
        f.write("#!/bin/bash\n")

        name = kwargs.get("name", DEFAULT_JOB_NAME)
        if job_array_n_jobs is not None:
            f.write(f"#SBATCH --array=1-{job_array_n_jobs}\n")
        f.write(f"#SBATCH -J {name}\n")

        if (account := kwargs.get("account")) is not None:
            f.write(f"#SBATCH --account={account}\n")

        if (walltime := kwargs.get("walltime", DEFAULT_WALLTIME)) is not None:
            # Convert the walltime to the format expected by LSF:
            # -W [hour:]minute[/host_name | /host_model]
            # E.g., 72 hours -> 72:00
            total_minutes = walltime.total_seconds() // 60
            hours = int(total_minutes // 60)
            minutes = int(total_minutes % 60)
            walltime = f"{hours:02d}:{minutes:02d}"
            f.write(f"#SBATCH --time={walltime}\n")

        if (nodes := kwargs.get("nodes", DEFAULT_NODES)) is not None:
            f.write(f"#SBATCH --nodes={nodes}\n")

        if (ntasks := kwargs.get("ntasks", DEFAULT_TASKS)) is not None:
            f.write(f"#SBATCH --ntasks={ntasks}\n")

        if (output_file := kwargs.get("output_file")) is not None:
            output_file = Path(output_file).absolute()
            if job_array_n_jobs is not None:
                output_file = _append_job_index_to_path(output_file)
            output_file = str(output_file)
            f.write(f"#SBATCH --output={output_file}\n")

        if (error_file := kwargs.get("error_file")) is not None:
            error_file = Path(error_file).absolute()
            if job_array_n_jobs is not None:
                error_file = _append_job_index_to_path(error_file)
            error_file = str(error_file)
            f.write(f"#SBATCH --error={error_file}\n")

        if (partition := kwargs.get("partition")) is not None:
            if isinstance(partition, str):
                partition = [partition]
            f.write(f"#SBATCH --partition={','.join(partition)}\n")

        if (memory_mb := kwargs.get("memory_mb")) is not None:
            f.write(f"#SBATCH --mem={memory_mb}\n")

        if (memory_per_cpu_mb := kwargs.get("memory_per_cpu_mb")) is not None:
            f.write(f"#SBATCH --mem-per-cpu={memory_per_cpu_mb}\n")

        if (memory_per_gpu_mb := kwargs.get("memory_per_gpu_mb")) is not None:
            f.write(f"#SBATCH --mem-per-gpu={memory_per_gpu_mb}\n")

        if (cpus_per_task := kwargs.get("cpus_per_task")) is not None:
            f.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")

        if (gres := kwargs.get("gres")) is not None:
            if isinstance(gres, str):
                gres = [gres]
            f.write(f"#SBATCH --gres={','.join(gres)}\n")

        if (gpus := kwargs.get("gpus")) is not None:
            f.write(f"#SBATCH --gpus={gpus}\n")

        if (gpus_per_node := kwargs.get("gpus_per_node")) is not None:
            f.write(f"#SBATCH --gpus-per-node={gpus_per_node}\n")

        if (gpus_per_task := kwargs.get("gpus_per_task")) is not None:
            f.write(f"#SBATCH --gpus-per-task={gpus_per_task}\n")

        if (mail_user := kwargs.get("mail_user")) is not None:
            f.write(f"#SBATCH --mail-user={mail_user}\n")

        if (mail_type := kwargs.get("mail_type")) is not None:
            if isinstance(mail_type, str):
                mail_type = [mail_type]
            f.write(f"#SBATCH --mail-type={','.join(mail_type)}\n")

        if (dependency := kwargs.get("dependency")) is not None:
            f.write(f"#SBATCH --dependency={dependency}\n")

        if kwargs.get("exclusive"):
            f.write("#SBATCH --exclusive\n")

        if (constraint := kwargs.get("constraint")) is not None:
            if isinstance(constraint, str):
                constraint = [constraint]
            f.write(f"#SBATCH --constraint={','.join(constraint)}\n")

        f.write("\n")

        setup_commands: list[str] = []
        setup_commands.extend(kwargs.get("setup_commands", []))
        for setup_command in setup_commands:
            f.write(f"{setup_command}\n")

        f.write("\n")

        if (command_prefix := kwargs.get("command_prefix")) is not None:
            command = " ".join(
                x_stripped
                for x in (command_prefix, command)
                if (x_stripped := x.strip())
            )
        f.write(f"{command}\n")

    return path


def _update_kwargs(kwargs: SlurmJobKwargs):
    # Update the kwargs with the default values
    kwargs = copy.deepcopy(kwargs)

    if (update_kwargs_fn := kwargs.get("update_kwargs_fn")) is not None:
        kwargs = copy.deepcopy(update_kwargs_fn(kwargs))

    return kwargs


@overload
def to_batch_script(
    dest: Path, command: str, /, **kwargs: Unpack[SlurmJobKwargs]
) -> Path: ...


@overload
def to_batch_script(
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args: tuple[Unpack[TArgs]],
    /,
    **kwargs: Unpack[SlurmJobKwargs],
) -> Path: ...


def to_batch_script(
    dest: Path,
    command_or_callable,
    args=None,
    /,
    **kwargs: Unpack[SlurmJobKwargs],
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
    **kwargs: Unpack[SlurmJobKwargs],
) -> Path: ...


@overload
def to_array_batch_script(
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args_list: Sequence[tuple[Unpack[TArgs]]],
    /,
    job_index_variable: str = "SLURM_ARRAY_TASK_ID",
    **kwargs: Unpack[SlurmJobKwargs],
) -> Path: ...


def to_array_batch_script(
    dest: Path,
    command_or_callable,
    args_list_or_num_jobs,
    /,
    job_index_variable: str = "SLURM_ARRAY_TASK_ID",
    **kwargs: Unpack[SlurmJobKwargs],
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
            start_idx=1,  # Slurm job indices are 1-based
        ).to_bash_command(job_index_variable)
    else:
        raise TypeError(f"Expected str or callable, got {type(command_or_callable)}")

    return _write_batch_script_to_file(
        dest / "launch.sh",
        kwargs,
        command,
        job_array_n_jobs=num_jobs,
    )
