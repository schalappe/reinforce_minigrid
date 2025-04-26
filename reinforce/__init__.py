# -*- coding: utf-8 -*-
"""
REINFORCE Agent Training Module.

This module provides functionalities for training and evaluating a reinforcement learning agent using
the REINFORCE algorithm within the MiniGrid environment. It includes logger setup and potentially
other utilities related to the training process.

Functions
---------
setup_logger()
    Configures the Loguru logger for the module.
"""

import sys

from loguru import logger


def setup_logger():
    """
    Configure Loguru Logger.

    Sets up the Loguru logger with a specific format for console output (stderr).
    The default logging level is set to INFO.

    Notes
    -----
    Removes any existing handlers before adding the new stderr handler to avoid duplicate logs.
    The format includes timestamp, level, name, function, line number, and the message, with colorization enabled.
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
