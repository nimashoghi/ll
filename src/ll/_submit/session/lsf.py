import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, TypedDict, overload

from typing_extensions import TypeAlias, TypeVarTuple, Unpack

from ...picklerunner import serialize_many, serialize_single

DEFAULT_JOB_NAME = "ll"
DEFAULT_NODES = 1
DEFAULT_WALLTIME = timedelta(hours=72)
DEFAULT_NTASKS_PER_NODE = 6

TArgs = TypeVarTuple("TArgs")

_Path: TypeAlias = str | Path | os.PathLike


class LSFJobKwargs(TypedDict, total=False):
    name: str
    """
    The name of the job.

    This corresponds to the "-J" option in bsub.
    """

    queue: str
    """
    The queue to submit the job to.

    This corresponds to the "-q" option in bsub. If not specified, the default queue will be used.
    """

    output_file: _Path
    """
    The file to write the job output to.

    This corresponds to the "-o" option in bsub. If not specified, the output will be written to the default output file.
    """

    error_file: _Path
    """
    The file to write the job errors to.

    This corresponds to the "-e" option in bsub. If not specified, the errors will be written to the default error file.
    """

    walltime: timedelta
    """
    The maximum walltime for the job.

    This corresponds to the "-W" option in bsub. The format is "HH:MM" or "MM". If not specified, the default walltime will be used.
    """

    memory_mb: int
    """
    The maximum memory for the job in MB.

    This corresponds to the "-M" option in bsub. If not specified, the default memory limit will be used.
    """

    cpu_limit: int
    """
    The CPU time limit for the job in minutes.

    This corresponds to the "-c" option in bsub. If not specified, the default CPU limit will be used.
    """

    rerunnable: bool
    """
    Whether the job should be rerunnable.

    This corresponds to the "-r" option in bsub. If set to True, the job will be rerun if it fails due to a system failure.
    """

    dependency_conditions: Sequence[str]
    """
    The dependency conditions for the job.

    This corresponds to the "-w" option in bsub. Each condition is a string that specifies the dependency condition.
    Multiple conditions can be specified, and they will be combined using logical AND.
    """

    email: str
    """
    The email address to send notifications to.

    This corresponds to the "-u" option in bsub. If specified, job notifications will be sent to this email address.
    """

    notify_begin: bool
    """
    Whether to send a notification when the job begins.

    This corresponds to the "-B" option in bsub. If set to True, a notification will be sent when the job begins.
    """

    notify_end: bool
    """
    Whether to send a notification when the job ends.

    This corresponds to the "-N" option in bsub. If set to True, a notification will be sent when the job ends.
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

    project: str
    """
    The project to charge the job to.

    This corresponds to the "-P" option in bsub. If specified, the job will be charged to this project.
    """

    nodes: int
    """
    The number of nodes to use for the job.

    This corresponds to the "-nnodes" option in bsub. The default is 1 node.
    """

    alloc_flags: str
    """
    The allocation flags for the job.

    This corresponds to the "-alloc_flags" option in bsub. If specified, the job will be allocated using these flags.
    """

    command_prefix: str
    """
    A command to prefix the job command with.

    This is used to add commands like `jsrun` to the job command.
    """

    # Our own custom options
    # Is this being sent to the Summit cluster?
    summit: bool


SUMMIT_DEFAULTS: LSFJobKwargs = {
    "command_prefix": "jsrun -n6 -c7 -g1 -a7 -dcyclic",
}


def _append_job_index_to_path(path: Path) -> Path:
    # If job array, append the job index to the output file
    # E.g., if `output_file` is "output_%J.out", we want "output_%J_%I.out"
    stem = path.stem
    suffix = path.suffix
    new_stem = f"{stem}_%I"
    new_path = path.with_name(new_stem + suffix)
    return new_path


def _write_batch_script_to_file(
    path: Path,
    kwargs: LSFJobKwargs,
    command: str,
    job_array_n_jobs: int | None = None,
):
    with path.open("w") as f:
        f.write("#!/bin/bash\n")

        name = kwargs.get("name", DEFAULT_JOB_NAME)
        if job_array_n_jobs is not None:
            name += "[1-" + str(job_array_n_jobs) + "]"
        f.write(f"#BSUB -J {name}\n")

        if (project := kwargs.get("project")) is not None:
            f.write(f"#BSUB -P {project}\n")

        if (walltime := kwargs.get("walltime", DEFAULT_WALLTIME)) is not None:
            # Convert the walltime to the format expected by LSF:
            # -W [hour:]minute[/host_name | /host_model]
            # E.g., 72 hours -> 72:00
            total_minutes = walltime.total_seconds() // 60
            hours = int(total_minutes // 60)
            minutes = int(total_minutes % 60)
            walltime = f"{hours:02d}:{minutes:02d}"
            f.write(f"#BSUB -W {walltime}\n")

        if (nodes := kwargs.get("nodes", DEFAULT_NODES)) is not None:
            f.write(f"#BSUB -nnodes {nodes}\n")

        if (output_file := kwargs.get("output_file")) is not None:
            output_file = Path(output_file).absolute()
            if job_array_n_jobs is not None:
                output_file = _append_job_index_to_path(output_file)
            output_file = str(output_file)
            f.write(f"#BSUB -o {output_file}\n")

        if (error_file := kwargs.get("error_file")) is not None:
            error_file = Path(error_file).absolute()
            if job_array_n_jobs is not None:
                error_file = _append_job_index_to_path(error_file)
            error_file = str(error_file)
            f.write(f"#BSUB -e {error_file}\n")

        if (queue := kwargs.get("queue")) is not None:
            f.write(f"#BSUB -q {queue}\n")

        if (memory_mb := kwargs.get("memory_mb")) is not None:
            f.write(f"#BSUB -M {memory_mb}\n")

        if (cpu_limit := kwargs.get("cpu_limit")) is not None:
            f.write(f"#BSUB -c {cpu_limit}\n")

        if (rerunnable := kwargs.get("rerunnable")) is not None:
            f.write(f"#BSUB -r {'y' if rerunnable else 'n'}\n")

        for dependency_condition in kwargs.get("dependency_conditions", []):
            f.write(f"#BSUB -w {dependency_condition}\n")

        if (email := kwargs.get("email")) is not None:
            f.write(f"#BSUB -u {email}\n")

        if (notify_begin := kwargs.get("notify_begin")) is not None:
            f.write(f"#BSUB -B {'y' if notify_begin else 'n'}\n")

        if (notify_end := kwargs.get("notify_end")) is not None:
            f.write(f"#BSUB -N {'y' if notify_end else 'n'}\n")

        if (alloc_flags := kwargs.get("alloc_flags")) is not None:
            f.write(f"#BSUB -alloc_flags {alloc_flags}\n")

        f.write("\n")

        for key, value in kwargs.get("environment", {}).items():
            f.write(f"export {key}={value}\n")

        f.write("\n")

        for setup_command in kwargs.get("setup_commands", []):
            f.write(f"{setup_command}\n")

        f.write("\n")

        f.write(f"{command}\n")

    return path


def _update_kwargs(kwargs: LSFJobKwargs):
    # Update the kwargs with the default values
    kwargs = {**kwargs}

    # If the job is being submitted to Summit, update the kwargs with the Summit defaults
    if kwargs.get("summit"):
        kwargs = {**SUMMIT_DEFAULTS, **kwargs}

    return kwargs


@overload
def to_batch_script(
    dest: Path, command: str, /, **kwargs: Unpack[LSFJobKwargs]
) -> Path: ...


@overload
def to_batch_script(
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args: tuple[Unpack[TArgs]],
    /,
    **kwargs: Unpack[LSFJobKwargs],
) -> Path: ...


def to_batch_script(
    dest: Path,
    command_or_callable,
    args=None,
    /,
    **kwargs: Unpack[LSFJobKwargs],
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
            *args,
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
    **kwargs: Unpack[LSFJobKwargs],
) -> Path: ...


@overload
def to_array_batch_script(
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args_list: Sequence[tuple[Unpack[TArgs]]],
    /,
    job_index_variable: str = "LSB_JOBINDEX",
    **kwargs: Unpack[LSFJobKwargs],
) -> Path: ...


def to_array_batch_script(
    dest: Path,
    command_or_callable,
    args_list_or_num_jobs,
    /,
    job_index_variable: str = "LSB_JOBINDEX",
    **kwargs: Unpack[LSFJobKwargs],
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
        ).to_bash_command(job_index_variable)
    else:
        raise TypeError(f"Expected str or callable, got {type(command_or_callable)}")

    return _write_batch_script_to_file(
        dest / "launch.sh",
        kwargs,
        command,
        job_array_n_jobs=num_jobs,
    )


@dataclass(frozen=True)
class LSFSubmissionOutput:
    script_path: Path
    """
    The path to the batch script that was generated.
    """

    job_ids: Sequence[str]
    """
    The job IDs of the submitted jobs.

    The list size is 1 if the job was submitted as a single job, and the list size is the number of jobs if the job was submitted as an array job.
    """


def submit_job(script_path: Path) -> LSFSubmissionOutput:
    # Submit the job using bsub
    output = subprocess.check_output(["bsub", str(script_path)], text=True)

    # Extract the job IDs from the bsub output
    job_ids = re.findall(r"Job <(\d+)> is submitted", output)

    # Check if it's a job array
    job_array_match = re.search(r"Job <(\d+)\[\d+-\d+\]> is submitted", output)
    if job_array_match:
        # Extract the job array ID
        job_array_id = job_array_match.group(1)

        # Get the individual job IDs in the array
        output = subprocess.check_output(
            ["bjobs", "-A", "-o", "jobid", job_array_id], text=True
        )
        job_ids = output.strip().split("\n")[1:]  # Skip the header line

    return LSFSubmissionOutput(script_path=script_path, job_ids=job_ids)
