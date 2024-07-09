import argparse
import ast
import glob
import sys
from pathlib import Path


def get_imports(file_path: Path):
    with open(file_path, "r") as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError:
            print(f"Syntax error in file: {file_path}", file=sys.stderr)
            return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # Absolute import
                imports.add(node.module.split(".")[0])
    return imports


def main():
    parser = argparse.ArgumentParser(
        description="Find unique Python packages used in files."
    )
    parser.add_argument("glob_pattern", help="Glob pattern to match files")
    parser.add_argument(
        "--exclude-std", action="store_true", help="Exclude Python standard libraries"
    )
    args = parser.parse_args()

    all_imports = set()
    for file_path in glob.glob(args.glob_pattern, recursive=True):
        all_imports.update(get_imports(Path(file_path)))

    if args.exclude_std:
        std_libs = set(sys.stdlib_module_names)
        std_libs.update({"pkg_resources"})
        all_imports = all_imports - std_libs

    for package in sorted(all_imports):
        print(package)


if __name__ == "__main__":
    main()
