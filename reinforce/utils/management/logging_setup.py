# -*- coding: utf-8 -*-
"""
Logger configuration module for the module.

This module sets up the Loguru logger with a custom format and logging level.
"""

import sys

from loguru import logger


def setup_logger():
    """
    Configures the Loguru logger for the module.

    Sets up a handler to log messages to stderr with a specific format and sets the default logging level to INFO.
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
