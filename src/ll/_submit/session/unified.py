import os
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Literal, TypeAlias

from typing_extensions import TypedDict

from .lsf import LSFJobKwargs
from .slurm import SlurmJobKwargs

_Path: TypeAlias = str | Path | os.PathLike


class GenericJobKwargs(TypedDict, total=False):
    name: str
    """The name of the job."""

    partition: str
    """The partition or queue to submit the job to."""

    account: str
    """The account (or project) to charge the job to."""

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
    The constraint to request for the job. For SLRUM, this corresponds to the `--constraint` option. For LSF, this corresponds to the `-m` option.
    """

    slurm: SlurmJobKwargs
    """Additional keyword arguments for Slurm jobs."""

    lsf: LSFJobKwargs
    """Additional keyword arguments for LSF jobs."""
