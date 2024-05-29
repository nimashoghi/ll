import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", default="bash", help="Shell to generate hook for")
    args = parser.parse_args()

    if "/.pixi/" in sys.prefix:
        # If we're in a pixi environment, we use `pixi shell-hook --shell bash`
        # to generate the shell hook.
        subprocess.run(["pixi", "shell-hook", "--shell", args.shell])
    else:
        # Otherwise, we assume it's a conda environment and use `conda shell.bash hook`
        # to generate the shell hook.
        subprocess.run(["conda", f"shell.{args.shell}", "hook"])


if __name__ == "__main__":
    main()
