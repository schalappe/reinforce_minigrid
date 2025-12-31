"""
REINFORCE Agent Training Module.

This module provides functionalities for training and evaluating reinforcement learning agents using
multiple algorithms within the MiniGrid environment.

Supported algorithms:
- PPO (Proximal Policy Optimization) with curriculum learning and exploration enhancements
- Rainbow DQN with six algorithmic improvements

Subpackages
-----------
core
    Shared base classes and utilities (BaseAgent, BaseBuffer, network_utils, schedules)
ppo
    PPO algorithm implementation (agent, buffer, network, rnd, exploration)
dqn
    Rainbow DQN implementation (agent, buffer, network, losses)
config
    Configuration management (Pydantic models, YAML loading)
"""

import sys

from loguru import logger


def setup_logger():
    """
    Configure Loguru Logger.

    Sets up the Loguru logger with a specific format for console output (stderr).
    The default logging level is set to INFO.
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
