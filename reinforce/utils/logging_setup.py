import sys

from loguru import logger


def setup_logger():
    """
    Configures the Loguru logger for the application.

    Sets up a handler to log messages to stderr with a specific format
    and sets the default logging level to DEBUG.
    """
    logger.remove()  # Remove default handler to avoid duplicate logs if re-configured
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )
    logger.info("Logger configured successfully.")


# Configure logger immediately when this module is imported
# setup_logger() # Option 1: Configure on import (can be problematic if imported multiple times)

# Option 2: Explicit call needed from entry points
# This is generally safer. We will call setup_logger() from experiment_runner.py and hyperparameter_search.py
