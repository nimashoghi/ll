from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubmitOutput:
    submission_command: list[str]
    submission_script_path: Path

    @property
    def submission_command_str(self) -> str:
        return " ".join(self.submission_command)
