import argparse
import contextlib
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

import cloudpickle as pickle
from typing_extensions import TypedDict, override

_Path: TypeAlias = str | Path | PathLike


class SerializedFunctionCallDict(TypedDict):
    fn: Callable
    args: Sequence[Any]
    kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class SerializedFunction(PathLike):
    path: Path

    _additional_command_parts: Sequence[str] = ()

    def to_command_parts(self, python_executable: str | None = None):
        if python_executable is None:
            python_executable = sys.executable

        return [
            python_executable,
            "-m",
            __name__,
            str(self.path),
            *self._additional_command_parts,
        ]

    def to_command_str(self, python_executable: str | None = None) -> str:
        return " ".join(self.to_command_parts(python_executable))

    @override
    def __fspath__(self) -> str:
        return str(self.path)


def serialize_single(
    dest: _Path,
    fn: Callable,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    additional_command_parts: Sequence[str] = (),
):
    serialized: SerializedFunctionCallDict = {"fn": fn, "args": args, "kwargs": kwargs}

    dest = Path(dest)
    with dest.open("wb") as file:
        pickle.dump(serialized, file)

    return SerializedFunction(dest, additional_command_parts)


@dataclass(frozen=True)
class SerializedMultiFunction(PathLike):
    base_dir: Path
    functions: Sequence[SerializedFunction]
    _additional_command_parts: Sequence[str] = ()
    print_environment_info: bool = True

    def to_bash_command(
        self,
        job_index_variable: str,
        environment: Mapping[str, str] | None = None,
        python_executable: str | None = None,
    ) -> list[str]:
        if python_executable is None:
            python_executable = sys.executable

        command: list[str] = []

        # command = f'{python_executable} -m {__name__} "{str(self.base_dir.absolute())}/${{{job_index_variable}}}.pkl"'
        command.append(python_executable)
        command.append("-m")
        command.append(__name__)
        if environment:
            for key, value in environment.items():
                command.append("--env")
                command.append(f"{key}={value}")
        if self.print_environment_info:
            command.append("--print-environment-info")
        else:
            command.append("--no-print-environment-info")
        command.append(
            f'"{str(self.base_dir.absolute())}/${{{job_index_variable}}}.pkl"'
        )

        if self._additional_command_parts:
            # command += " " + " ".join(self._additional_command_parts)
            command.extend(self._additional_command_parts)
        return command

    def _to_bash_command_sequential_worker(
        self,
        num_workers: int,
        worker_id: int,
        python_executable: str,
        environment: Mapping[str, str] | None = None,
    ) -> list[str]:
        assert 0 <= worker_id < num_workers, f"{worker_id=} {num_workers=}"

        all_files = [
            f'"{str(fn.path.absolute())}"'
            for i, fn in enumerate(self.functions)
            if i % num_workers == worker_id
        ]

        command: list[str] = []
        # command = f"{python_executable} -m {__name__} {all_files}"
        command.append(python_executable)
        command.append("-m")
        command.append(__name__)
        if environment:
            for key, value in environment.items():
                command.append("--env")
                command.append(f"{key}={value}")
        command.extend(all_files)

        if self._additional_command_parts:
            # command += " " + " ".join(self._additional_command_parts)
            command.extend(self._additional_command_parts)
        return command

    def to_bash_command_sequential_workers(
        self,
        num_workers: int,
        python_executable: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> list[list[str]]:
        if python_executable is None:
            python_executable = sys.executable

        return [
            self._to_bash_command_sequential_worker(
                num_workers, i, python_executable, environment
            )
            for i in range(num_workers)
        ]

    @override
    def __fspath__(self) -> str:
        return str(self.base_dir)


def serialize_many(
    destdir: _Path,
    fn: Callable,
    args_and_kwargs_list: Sequence[tuple[Sequence[Any], Mapping[str, Any]]],
    start_idx: int = 0,
    additional_command_parts: Sequence[str] = (),
    print_environment_info: bool = True,
):
    serialized_list: list[SerializedFunction] = []

    destdir = Path(destdir)
    for i, (args, kwargs) in enumerate(args_and_kwargs_list):
        dest = destdir / f"{i+start_idx}.pkl"
        serialized = serialize_single(dest, fn, args, kwargs)
        serialized_list.append(serialized)

    return SerializedMultiFunction(
        destdir,
        serialized_list,
        additional_command_parts,
        print_environment_info=print_environment_info,
    )


def execute_single(path: _Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")

    with path.open("rb") as file:
        d = pickle.load(file)

    # Validate the dict.
    assert isinstance(d, Mapping), f"Expected a dict, got {type(d)}"
    # `fn`
    assert (fn := d.get("fn")) is not None, f"Expected a 'fn' key, got {d.keys()}"
    assert callable(fn), f"Expected a callable, got {type(fn)}"
    # `args`
    assert (args := d.get("args")) is not None, f"Expected a 'args' key, got {d.keys()}"
    assert isinstance(args, Sequence), f"Expected a tuple, got {type(args)}"
    # `kwargs`
    assert (
        kwargs := d.get("kwargs")
    ) is not None, f"Expected a 'kwargs' key, got {d.keys()}"
    assert isinstance(kwargs, Mapping), f"Expected a dict, got {type(kwargs)}"

    # Call the function and return the result.
    return fn(*args, **kwargs)


def execute_many(fns: SerializedMultiFunction | Sequence[_Path]):
    if isinstance(fns, SerializedMultiFunction):
        fns = fns.functions

    return [execute_single(path) for path in fns]


def _resolve_paths(paths: Sequence[Path]):
    for path in paths:
        if path.is_file():
            yield path
            continue

        for child in path.iterdir():
            if child.is_file() and child.suffix == ".pkl":
                yield child


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Paths to the sessions to run",
    )
    parser.add_argument(
        "--unset-cuda",
        action=argparse.BooleanOptionalAction,
        help="Unset the CUDA_VISIBLE_DEVICES environment variable",
    )
    parser.add_argument(
        "--print-environment-info",
        action=argparse.BooleanOptionalAction,
        help="Print the environment information before starting the session",
        default=True,
    )
    parser.add_argument(
        "--env",
        "-e",
        help="Set the environment variable. Format: KEY=VALUE",
        action="append",
    )

    args = parser.parse_args()
    return args


@contextlib.contextmanager
def _set_env(key: str, value: str):
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is not None:
            os.environ[key] = original_value
        else:
            del os.environ[key]


def picklerunner_main():
    with contextlib.ExitStack() as stack:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        args = _parse_args()

        # Print the environment information if requested.
        if args.print_environment_info:
            from ._submit.print_environment_info import print_environment_info

            print_environment_info(log)

        # Set the environment variables if requested.
        if args.env:
            for env in args.env:
                key, value = env.split("=", 1)
                log.critical(f"Setting {key}={value}...")
                stack.enter_context(_set_env(key, value))

        # Unset the CUDA_VISIBLE_DEVICES environment variable if requested.
        if args.unset_cuda:
            log.critical("Unsetting CUDA_VISIBLE_DEVICES...")
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            for i in range(40):
                os.environ.pop(f"CUDA_VISIBLE_DEVICES{i}", None)

        paths = list(_resolve_paths(args.paths))
        if not paths:
            raise ValueError("No paths provided")

        # Sort by the job index.
        paths = sorted(paths, key=lambda path: int(path.stem))

        # Make sure all paths exist
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path {path} does not exist")

        for i, path in enumerate(paths):
            log.critical(f"Executing #{i}: {path=}...")
            # The result should be saved to {path_without_extension}.result.pkl
            result = execute_single(path)
            result_path = path.with_suffix(".result.pkl")
            log.critical(f"Saving result to {result_path}...")
            with result_path.open("wb") as file:
                pickle.dump(result, file)

        log.critical("Done!")


if __name__ == "__main__":
    picklerunner_main()
