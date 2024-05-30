import copy
import os
import signal
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

from typing_extensions import TypeAlias, TypedDict, TypeVarTuple, Unpack

from ._output import SubmitOutput
from ._script import helper_script_to_command, write_helper_script

log = getLogger(__name__)

TArgs = TypeVarTuple("TArgs")

_Path: TypeAlias = str | Path | os.PathLike
MailType: TypeAlias = Literal[
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

    qos: str
    """
    The quality of service to submit the job to.

    This corresponds to the "--qos" option in sbatch.
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

    time: timedelta | Literal[0]
    """
    The maximum time for the job.

    This corresponds to the "-t" option in sbatch. A value of 0 means no time limit.
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

    ntasks_per_node: int
    """
    The number of tasks for each node.

    This corresponds to the "--ntasks-per-node" option in sbatch.
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

    gpus: int | str
    """
    Specify the total number of GPUs required for the job. An optional GPU type specification can be supplied.

    This corresponds to the "-G" option in sbatch.
    """

    gpus_per_node: int | str
    """
    Specify the number of GPUs required for the job on each node included in the job's resource allocation. An optional GPU type specification can be supplied.

    This corresponds to the "--gpus-per-node" option in sbatch.
    """

    gpus_per_task: int | str
    """
    Specify the number of GPUs required for the job on each task to be spawned in the job's resource allocation. An optional GPU type specification can be supplied.

    This corresponds to the "--gpus-per-task" option in sbatch.
    """

    mail_user: str
    """
    User to receive email notification of state changes as defined by mail_type.

    This corresponds to the "--mail-user" option in sbatch.
    """

    mail_type: MailType | Sequence[MailType]
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

    signal: signal.Signals
    """
    The signal to send to the job when the job is being terminated.

    This corresponds to the "--signal" option in sbatch.
    """

    open_mode: str
    """
    The open mode for the output and error files.

    This corresponds to the "--open-mode" option in sbatch.

    Valid values are "append" and "truncate".
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

    command_prefix: str
    """
    A command to prefix the job command with.

    This is used to add commands like `srun` to the job command.
    """

    command_template: str
    """
    The template for the command to execute the helper script.

    Default: `bash {/path/to/helper.sh}`.
    """

    srun_flags: str | Sequence[str]
    """
    The flags to pass to the `srun` command.
    """

    # Our own custom options
    update_kwargs_fn: "Callable[[SlurmJobKwargs, Path], SlurmJobKwargs]"
    """A function to update the kwargs."""


DEFAULT_KWARGS: SlurmJobKwargs = {
    "name": "ll",
    "nodes": 1,
    # "time": timedelta(hours=2),
    "signal": signal.SIGUSR1,
    "open_mode": "append",
}


def _determine_gres(kwargs: SlurmJobKwargs) -> Sequence[str] | None:
    """
    There are many different ways to specify GPU resources, but some are buggy.

    This function normalizes all other ways to specify GPU resources to the `gres` option.
    """

    # If `--gres` is set, just return it
    if (gres := kwargs.get("gres")) is not None:
        if isinstance(gres, str):
            gres = [gres]
        return gres

    # We will only support `--gpus` if `--nodes` is set to 1
    if (gpus := kwargs.get("gpus")) is not None:
        if kwargs.get("nodes") != 1:
            raise ValueError("Cannot specify `gpus` without `nodes` set to 1.")
        if isinstance(gpus, int):
            gpus = [f"gpu:{gpus}"]
        return gpus

    # `--gpus-per-task` is only supported if `--ntasks-per-node` is set (or can be inferred).
    if (gpus_per_task := kwargs.get("gpus_per_task")) is not None:
        if (ntasks_per_node := _determine_ntasks_per_node(kwargs)) is None:
            raise ValueError(
                "Cannot specify `gpus_per_task` without `ntasks_per_node`."
            )

        if isinstance(gpus_per_task, str):
            gpus_per_task = int(gpus_per_task)

        gpus_per_node = ntasks_per_node * gpus_per_task
        return [f"gpu:{gpus_per_node}"]

    # `--gpus-per-node` has no restrictions
    if (gpus_per_node := kwargs.get("gpus_per_node")) is not None:
        if isinstance(gpus_per_node, int):
            gpus_per_node = [f"gpu:{gpus_per_node}"]
        return gpus_per_node

    return None


def _determine_ntasks_per_node(kwargs: SlurmJobKwargs) -> int | None:
    # If `--ntasks-per-node` is set, just return it
    if (ntasks_per_node := kwargs.get("ntasks_per_node")) is not None:
        return ntasks_per_node

    # If `--ntasks` is set, we can infer `--ntasks-per-node`
    if (ntasks := kwargs.get("ntasks")) is not None:
        if (nodes := kwargs.get("nodes")) is None:
            raise ValueError("Cannot infer `ntasks_per_node` without `nodes`.")

        # If nnodes is not divisible by ntasks, raise an error
        if nodes % ntasks != 0:
            raise ValueError(
                "The number of nodes must be divisible by the number of tasks."
            )

        return ntasks // nodes

    return None


def _write_batch_script_to_file(
    path: Path,
    kwargs: SlurmJobKwargs,
    command: str,
    job_array_n_jobs: int | None = None,
):
    with path.open("w") as f:
        f.write("#!/bin/bash\n")

        if job_array_n_jobs is not None:
            f.write(f"#SBATCH --array=1-{job_array_n_jobs}\n")

        if (name := kwargs.get("name")) is not None:
            f.write(f"#SBATCH -J {name}\n")

        if (account := kwargs.get("account")) is not None:
            f.write(f"#SBATCH --account={account}\n")

        if (time := kwargs.get("time")) is not None:
            # A time limit of zero requests that no time limit be imposed. Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
            if time == 0:
                time_str = "0"
            else:
                total_seconds = time.total_seconds()
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours > 24:
                    days, hours = divmod(hours, 24)
                    time_str = f"{int(days)}-{int(hours):02d}:{int(minutes):02d}"
                else:
                    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            f.write(f"#SBATCH --time={time_str}\n")

        if (nodes := kwargs.get("nodes")) is not None:
            f.write(f"#SBATCH --nodes={nodes}\n")

        if (ntasks := kwargs.get("ntasks")) is not None:
            f.write(f"#SBATCH --ntasks={ntasks}\n")

        if (ntasks_per_node := kwargs.get("ntasks_per_node")) is not None:
            f.write(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")

        if (output_file := kwargs.get("output_file")) is not None:
            output_file = str(Path(output_file).absolute())
            f.write(f"#SBATCH --output={output_file}\n")

        if (error_file := kwargs.get("error_file")) is not None:
            error_file = str(Path(error_file).absolute())
            f.write(f"#SBATCH --error={error_file}\n")

        if (partition := kwargs.get("partition")) is not None:
            if isinstance(partition, str):
                partition = [partition]
            f.write(f"#SBATCH --partition={','.join(partition)}\n")

        if (qos := kwargs.get("qos")) is not None:
            f.write(f"#SBATCH --qos={qos}\n")

        if (memory_mb := kwargs.get("memory_mb")) is not None:
            f.write(f"#SBATCH --mem={memory_mb}\n")

        if (memory_per_cpu_mb := kwargs.get("memory_per_cpu_mb")) is not None:
            f.write(f"#SBATCH --mem-per-cpu={memory_per_cpu_mb}\n")

        if (memory_per_gpu_mb := kwargs.get("memory_per_gpu_mb")) is not None:
            f.write(f"#SBATCH --mem-per-gpu={memory_per_gpu_mb}\n")

        if (cpus_per_task := kwargs.get("cpus_per_task")) is not None:
            f.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")

        if gres := _determine_gres(kwargs):
            f.write(f"#SBATCH --gres={','.join(gres)}\n")

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

        if (open_mode := kwargs.get("open_mode")) is not None:
            f.write(f"#SBATCH --open-mode={open_mode}\n")

        if (constraint := kwargs.get("constraint")) is not None:
            if isinstance(constraint, str):
                constraint = [constraint]
            f.write(f"#SBATCH --constraint={','.join(constraint)}\n")

        if (signal := kwargs.get("signal")) is not None:
            f.write(f"#SBATCH --signal={signal.name}\n")

        f.write("\n")

        if (command_prefix := kwargs.get("command_prefix")) is not None:
            command = " ".join(
                x_stripped
                for x in (command_prefix, command)
                if (x_stripped := x.strip())
            )
        f.write(f"{command}\n")

    return path


def _update_kwargs(kwargs: SlurmJobKwargs, base_path: Path):
    # Update the kwargs with the default values
    kwargs = copy.deepcopy(kwargs)
    kwargs = {**DEFAULT_KWARGS, **kwargs}

    # If out/err files are not specified, set them
    logs_base = base_path.parent / "logs"
    logs_base.mkdir(exist_ok=True)

    if kwargs.get("output_file") is None:
        kwargs["output_file"] = logs_base / "output_%j_%a.out"

    if kwargs.get("error_file") is None:
        kwargs["error_file"] = logs_base / "error_%j_%a.err"

    # Update the command_prefix to add srun:
    command_parts: list[str] = ["srun"]
    if (srun_flags := kwargs.get("srun_flags")) is not None:
        if isinstance(srun_flags, str):
            srun_flags = [srun_flags]
        command_parts.extend(srun_flags)
    command_parts.append("--unbuffered")

    # Add the task id to the output filenames
    if (f := kwargs.get("output_file")) is not None:
        f = Path(f).absolute()
        command_parts.extend(
            [
                "--output",
                str(f.with_name(f"{f.stem}-%t{f.suffix}").absolute()),
            ]
        )
    if (f := kwargs.get("error_file")) is not None:
        f = Path(f).absolute()
        command_parts.extend(
            [
                "--error",
                str(f.with_name(f"{f.stem}-%t{f.suffix}").absolute()),
            ]
        )

    # If there is already a command prefix, combine them.
    if (existing_command_prefix := kwargs.get("command_prefix")) is not None:
        command_parts.extend(existing_command_prefix.split())
    # Add the command prefix to the kwargs.
    kwargs["command_prefix"] = " ".join(command_parts)

    if (update_kwargs_fn := kwargs.get("update_kwargs_fn")) is not None:
        kwargs = copy.deepcopy(update_kwargs_fn(kwargs, base_path))

    return kwargs


def to_array_batch_script(
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args_list: Sequence[tuple[Unpack[TArgs]]],
    /,
    job_index_variable: str = "SLURM_ARRAY_TASK_ID",
    print_environment_info: bool = True,
    **kwargs: Unpack[SlurmJobKwargs],
) -> SubmitOutput:
    """
    Create the batch script for the job.
    """

    from ...picklerunner import serialize_many

    kwargs = _update_kwargs(kwargs, dest)

    # Convert the command/callable to a string for the command
    num_jobs = len(args_list)

    destdir = dest / "fns"
    destdir.mkdir(exist_ok=True)

    serialized_command = serialize_many(
        destdir,
        callable,
        [(args, {}) for args in args_list],
        start_idx=1,  # Slurm job indices are 1-based
        print_environment_info=print_environment_info,
    )
    helper_path = write_helper_script(
        destdir,
        serialized_command,
        kwargs.get("environment", {}),
        kwargs.get("setup_commands", []),
        job_index_variable,
        None,
    )
    command = helper_script_to_command(helper_path, kwargs.get("command_template"))

    script_path = _write_batch_script_to_file(
        dest / "launch.sh",
        kwargs,
        command,
        job_array_n_jobs=num_jobs,
    )
    script_path = script_path.resolve().absolute()
    return SubmitOutput(
        submission_command=["sbatch", f"{script_path}"],
        submission_script_path=script_path,
    )
