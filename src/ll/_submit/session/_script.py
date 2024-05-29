import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ...picklerunner import SerializedMultiFunction


def write_helper_script(
    base_dir: Path,
    function: SerializedMultiFunction,
    environment: Mapping[str, str],
    setup_commands: Sequence[str],
    job_index_variable: str,
    python_executable: str | None = None,
    chmod: bool = True,
):
    with (out_path := (base_dir / "helper.sh")).open("w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -e\n\n")

        if python_executable is None:
            python_executable = sys.executable

        if environment:
            for key, value in environment.items():
                f.write(f"export {key}={value}\n")
            f.write("\n")

        if setup_commands:
            for setup_command in setup_commands:
                f.write(f"{setup_command}\n")
            f.write("\n")

        command = " ".join(
            function.to_bash_command(job_index_variable, python_executable)
        )
        f.write(f"{command}\n")

    if chmod:
        # Make the script executable
        out_path.chmod(0o755)

    return out_path


DEFAULT_TEMPLATE = "bash {helper_script}"


def helper_script_to_command(
    helper_script: Path,
    template: str | None,
) -> str:
    if not template:
        template = DEFAULT_TEMPLATE
    return template.format(helper_script=str(helper_script.absolute()))
