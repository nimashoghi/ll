import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

import cloudpickle as pickle
from typing_extensions import ParamSpec, override

_Path: TypeAlias = str | Path | PathLike


class SerializedFunctionCallDict(TypedDict):
    fn: Callable
    args: Sequence[Any]
    kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class SerializedFunction(PathLike):
    path: Path

    def to_command_parts(self, python_executable: str | None = None):
        if python_executable is None:
            python_executable = sys.executable

        return [python_executable, "-m", __name__, str(self.path)]

    def to_command_str(self, python_executable: str | None = None) -> str:
        return " ".join(self.to_command_parts(python_executable))

    @override
    def __fspath__(self) -> str:
        return str(self.path)


P = ParamSpec("P")


def serialize_single(
    dest: _Path,
    fn: Callable[P, Any],
    *args: P.args,
    **kwargs: P.kwargs,
):
    serialized: SerializedFunctionCallDict = {"fn": fn, "args": args, "kwargs": kwargs}

    dest = Path(dest)
    with dest.open("wb") as file:
        pickle.dump(serialized, file)

    return SerializedFunction(dest)


@dataclass(frozen=True)
class SerializedMultiFunction(PathLike):
    base_dir: Path
    functions: Sequence[SerializedFunction]

    def to_bash_command(
        self,
        job_index_variable: str,
        python_executable: str | None = None,
    ) -> str:
        if python_executable is None:
            python_executable = sys.executable

        return f'{python_executable} -m {__name__} "{str(self.base_dir.absolute())}/${{{job_index_variable}}}.pkl"'

    @override
    def __fspath__(self) -> str:
        return str(self.base_dir)


def serialize_many(
    destdir: _Path,
    fn: Callable,
    args_and_kwargs: Sequence[tuple[Sequence[Any], Mapping[str, Any]]],
    start_idx: int = 0,
):
    serialized_list: list[SerializedFunction] = []

    destdir = Path(destdir)
    for i, (args, kwargs) in enumerate(args_and_kwargs):
        dest = destdir / f"{i+start_idx}.pkl"
        serialized = serialize_single(dest, fn, *args, **kwargs)
        serialized_list.append(serialized)

    return SerializedMultiFunction(destdir, serialized_list)


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


def picklerunner_main():
    import argparse
    import logging

    def _resolve_paths(paths: Sequence[Path]):
        for path in paths:
            if path.is_file():
                yield path
                continue

            for child in path.iterdir():
                if child.is_file() and child.suffix == ".pkl":
                    yield child

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Paths to the sessions to run",
    )
    args = parser.parse_args()

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
