import sys

from loguru import logger


def setup_logger():
    """
    Configures the Loguru logger for the application.

    Sets up a handler to log messages to stderr with a specific format
    and sets the default logging level to DEBUG.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )
    logger.info("Logger configured successfully.")
