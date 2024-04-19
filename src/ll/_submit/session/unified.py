import os
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, cast, overload

from typing_extensions import TypeAlias, TypedDict, TypeVarTuple, Unpack

from . import lsf, slurm
from ._output import SubmitOutput

TArgs = TypeVarTuple("TArgs")
_Path: TypeAlias = str | Path | os.PathLike

log = getLogger(__name__)


class GenericJobKwargs(TypedDict, total=False):
    name: str
    """The name of the job."""

    partition: str
    """The partition or queue to submit the job to."""

    account: str
    """The account (or project) to charge the job to. Same as `project`."""

    project: str
    """The project (or account) to charge the job to. Same as `account`."""

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

    nodes: int
    """The number of nodes to request."""

    tasks_per_node: int
    """The number of tasks to request per node."""

    cpus_per_task: int
    """The number of CPUs to request per task."""

    gpus_per_task: int
    """The number of GPUs to request per task."""

    memory_mb: int
    """The maximum memory for the job in MB."""

    walltime: timedelta
    """The maximum walltime for the job."""

    email: str
    """The email address to send notifications to."""

    notifications: set[Literal["begin", "end"]]
    """The notifications to send via email."""

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

    This is used to add commands like `srun` or `jsrun` to the job command.
    """

    constraint: str
    """
    The constraint to request for the job. For SLRUM, this corresponds to the `--constraint` option. For LSF, this is unused.
    """

    additional_slurm_options: slurm.SlurmJobKwargs
    """Additional keyword arguments for Slurm jobs."""

    additional_lsf_options: lsf.LSFJobKwargs
    """Additional keyword arguments for LSF jobs."""


Scheduler: TypeAlias = Literal["slurm", "lsf"]


def _to_slurm(kwargs: GenericJobKwargs) -> slurm.SlurmJobKwargs:
    slurm_kwargs: slurm.SlurmJobKwargs = {}
    if (name := kwargs.get("name")) is not None:
        slurm_kwargs["name"] = name
    if (account := kwargs.get("account")) is not None:
        slurm_kwargs["account"] = account
    if (partition := kwargs.get("partition")) is not None:
        slurm_kwargs["partition"] = partition
    if (output_file := kwargs.get("output_file")) is not None:
        slurm_kwargs["output_file"] = output_file
    if (error_file := kwargs.get("error_file")) is not None:
        slurm_kwargs["error_file"] = error_file
    if (walltime := kwargs.get("walltime")) is not None:
        slurm_kwargs["time"] = walltime
    if (memory_mb := kwargs.get("memory_mb")) is not None:
        slurm_kwargs["memory_mb"] = memory_mb
    if (nodes := kwargs.get("nodes")) is not None:
        slurm_kwargs["nodes"] = nodes
    if (tasks_per_node := kwargs.get("tasks_per_node")) is not None:
        slurm_kwargs["ntasks_per_node"] = tasks_per_node
    if (cpus_per_task := kwargs.get("cpus_per_task")) is not None:
        slurm_kwargs["cpus_per_task"] = cpus_per_task
    if (gpus_per_task := kwargs.get("gpus_per_task")) is not None:
        slurm_kwargs["gpus_per_task"] = gpus_per_task
    if (constraint := kwargs.get("constraint")) is not None:
        slurm_kwargs["constraint"] = constraint
    if (email := kwargs.get("email")) is not None:
        slurm_kwargs["mail_user"] = email
    if (notifications := kwargs.get("notifications")) is not None:
        mail_type: list[slurm.MailType] = []
        for notification in notifications:
            match notification:
                case "begin":
                    mail_type.append("BEGIN")
                case "end":
                    mail_type.append("END")
                case _:
                    raise ValueError(f"Unknown notification type: {notification}")
        slurm_kwargs["mail_type"] = mail_type
    if (setup_commands := kwargs.get("setup_commands")) is not None:
        slurm_kwargs["setup_commands"] = setup_commands
    if (environment := kwargs.get("environment")) is not None:
        slurm_kwargs["environment"] = environment
    if (command_prefix := kwargs.get("command_prefix")) is not None:
        slurm_kwargs["command_prefix"] = command_prefix
    if (additional_kwargs := kwargs.get("additional_slurm_options")) is not None:
        slurm_kwargs.update(additional_kwargs)

    return slurm_kwargs


def _to_lsf(kwargs: GenericJobKwargs) -> lsf.LSFJobKwargs:
    lsf_kwargs: lsf.LSFJobKwargs = {}
    if (name := kwargs.get("name")) is not None:
        lsf_kwargs["name"] = name
    if (account := kwargs.get("account")) is not None:
        lsf_kwargs["project"] = account
    if (queue := kwargs.get("partition")) is not None:
        lsf_kwargs["queue"] = queue
    if (output_file := kwargs.get("output_file")) is not None:
        lsf_kwargs["output_file"] = output_file
    if (error_file := kwargs.get("error_file")) is not None:
        lsf_kwargs["error_file"] = error_file
    if (walltime := kwargs.get("walltime")) is not None:
        lsf_kwargs["walltime"] = walltime
    if (memory_mb := kwargs.get("memory_mb")) is not None:
        lsf_kwargs["memory_mb"] = memory_mb
    if (nodes := kwargs.get("nodes")) is not None:
        lsf_kwargs["nodes"] = nodes
    if (tasks_per_node := kwargs.get("tasks_per_node")) is not None:
        lsf_kwargs["rs_per_node"] = tasks_per_node
    if (cpus_per_task := kwargs.get("cpus_per_task")) is not None:
        lsf_kwargs["cpus_per_rs"] = cpus_per_task
    if (gpus_per_task := kwargs.get("gpus_per_task")) is not None:
        lsf_kwargs["gpus_per_rs"] = gpus_per_task
    if (constraint := kwargs.get("constraint")) is not None:
        log.warning(f'LSF does not support constraints, ignoring "{constraint=}".')
    if (email := kwargs.get("email")) is not None:
        lsf_kwargs["email"] = email
    if (notifications := kwargs.get("notifications")) is not None:
        if "begin" in notifications:
            lsf_kwargs["notify_begin"] = True
        if "end" in notifications:
            lsf_kwargs["notify_end"] = True
    if (setup_commands := kwargs.get("setup_commands")) is not None:
        lsf_kwargs["setup_commands"] = setup_commands
    if (environment := kwargs.get("environment")) is not None:
        lsf_kwargs["environment"] = environment
    if (command_prefix := kwargs.get("command_prefix")) is not None:
        lsf_kwargs["command_prefix"] = command_prefix
    if (additional_kwargs := kwargs.get("additional_lsf_options")) is not None:
        lsf_kwargs.update(additional_kwargs)

    return lsf_kwargs


@overload
def to_batch_script(
    scheduler: Scheduler,
    dest: Path,
    command: str,
    /,
    **kwargs: Unpack[GenericJobKwargs],
) -> SubmitOutput: ...


@overload
def to_batch_script(
    scheduler: Scheduler,
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args: tuple[Unpack[TArgs]],
    /,
    **kwargs: Unpack[GenericJobKwargs],
) -> SubmitOutput: ...


def to_batch_script(
    scheduler: Scheduler,
    dest: Path,
    command_or_callable,
    args=None,
    /,
    **kwargs: Unpack[GenericJobKwargs],
):
    match scheduler:
        case "slurm":
            slurm_kwargs = _to_slurm(kwargs)
            return slurm.to_batch_script(
                dest, command_or_callable, cast(Any, args), **slurm_kwargs
            )
        case "lsf":
            lsf_kwargs = _to_lsf(kwargs)
            return lsf.to_batch_script(
                dest, command_or_callable, cast(Any, args), **lsf_kwargs
            )


@overload
def to_array_batch_script(
    scheduler: Scheduler,
    dest: Path,
    command: str,
    num_jobs: int,
    /,
    **kwargs: Unpack[GenericJobKwargs],
) -> SubmitOutput: ...


@overload
def to_array_batch_script(
    scheduler: Scheduler,
    dest: Path,
    callable: Callable[[Unpack[TArgs]], Any],
    args_list: Sequence[tuple[Unpack[TArgs]]],
    /,
    job_index_variable: str | None = None,
    **kwargs: Unpack[GenericJobKwargs],
) -> SubmitOutput: ...


def to_array_batch_script(
    scheduler: Scheduler,
    dest: Path,
    command_or_callable,
    args_list_or_num_jobs,
    /,
    job_index_variable: str | None = None,
    **kwargs: Unpack[GenericJobKwargs],
):
    job_index_variable_kwargs = {}
    if job_index_variable is not None:
        job_index_variable_kwargs["job_index_variable"] = job_index_variable
    match scheduler:
        case "slurm":
            slurm_kwargs = _to_slurm(kwargs)
            return slurm.to_array_batch_script(
                dest,
                command_or_callable,
                cast(Any, args_list_or_num_jobs),
                **job_index_variable_kwargs,
                **slurm_kwargs,
            )
        case "lsf":
            lsf_kwargs = _to_lsf(kwargs)
            return lsf.to_array_batch_script(
                dest,
                command_or_callable,
                cast(Any, args_list_or_num_jobs),
                **job_index_variable_kwargs,
                **lsf_kwargs,
            )
