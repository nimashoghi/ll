import logging
import os
import sys


def print_environment_info(
    log: logging.Logger | None = None,
    use_rich_log_handler: bool = True,
):
    if log is None:
        logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]")
        log = logging.getLogger(__name__)

        # Set up the logging handler if requested.
        if use_rich_log_handler:
            try:
                from rich.logging import RichHandler
            except ImportError:
                pass
            else:
                log.addHandler(RichHandler())

    log.critical("Python executable: " + sys.executable)
    log.critical("Python version: " + sys.version)
    log.critical("Python prefix: " + sys.prefix)
    log.critical("Python path:")
    for path in sys.path:
        log.critical(f"  {path}")

    log.critical("Environment variables:")
    for key, value in os.environ.items():
        log.critical(f"  {key}={value}")

    log.critical("Command line arguments:")
    for i, arg in enumerate(sys.argv):
        log.critical(f"  {i}: {arg}")


if __name__ == "__main__":
    print_environment_info()
