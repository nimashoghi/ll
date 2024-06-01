import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from ...picklerunner import SerializedMultiFunction


def launcher_from_command(
    base_dir: Path,
    original_command: str | Iterable[str],
    environment: Mapping[str, str],
    setup_commands: Sequence[str],
    chmod: bool = True,
):
    """
    Creates a helper bash script for running the given function.

    The core idea: The helper script is essentially one additional layer of indirection
    that allows us to encapsulates the environment setup and the actual function call
    in a single bash script (that does not require properly set up Python environment).

    In effect, this allows us to, for example:
    - Easily run the function in the correct environment
        (without having to deal with shell hooks)
        using `conda run -n myenv bash /path/to/helper.sh`.
    - Easily run the function in a Singularity container
        using `singularity exec my_container.sif bash /path/to/helper.sh`.
    """
    with (out_path := (base_dir / "helper.sh")).open("w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -e\n\n")

        if environment:
            for key, value in environment.items():
                f.write(f"export {key}={value}\n")
            f.write("\n")

        if setup_commands:
            for setup_command in setup_commands:
                f.write(f"{setup_command}\n")
            f.write("\n")

        if not isinstance(original_command, str):
            original_command = " ".join(original_command)
        f.write(f"{original_command}\n")

    if chmod:
        # Make the script executable
        out_path.chmod(0o755)

    return out_path


def write_helper_script(
    base_dir: Path,
    function: SerializedMultiFunction,
    environment: Mapping[str, str],
    setup_commands: Sequence[str],
    job_index_variable: str,
    python_executable: str | None = None,
    chmod: bool = True,
):
    """
    Creates a helper bash script for running the given function.

    The core idea: The helper script is essentially one additional layer of indirection
    that allows us to encapsulates the environment setup and the actual function call
    in a single bash script (that does not require properly set up Python environment).

    In effect, this allows us to, for example:
    - Easily run the function in the correct environment
        (without having to deal with shell hooks)
        using `conda run -n myenv bash /path/to/helper.sh`.
    - Easily run the function in a Singularity container
        using `singularity exec my_container.sif bash /path/to/helper.sh`.
    """

    if python_executable is None:
        python_executable = sys.executable

    return launcher_from_command(
        base_dir,
        function.to_bash_command(job_index_variable, python_executable),
        environment,
        setup_commands,
        chmod,
    )


DEFAULT_TEMPLATE = "bash {script}"


def helper_script_to_command(script: Path, template: str | None) -> str:
    if not template:
        template = DEFAULT_TEMPLATE

    # Make sure the template has '{script}' in it
    if "{script}" not in template:
        raise ValueError(f"Template must contain '{{script}}'. Got: {template!r}")

    return template.format(script=str(script.absolute()))
