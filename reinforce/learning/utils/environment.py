# -*- coding: utf-8 -*-
"""Environment setup utilities for the Maze environment."""

import numpy as np
import tensorflow as tf
from gymnasium import Env
from loguru import logger
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

from maze.envs import Maze
from reinforce import setup_logger

setup_logger()

def setup_environment(seed: int) -> Env:
    """
    Initializes and wraps the Maze environment.

    Sets random seeds for reproducibility and applies necessary wrappers.

    Parameters
    ----------
    seed : int
        The random seed to use for environment initialization and randomization.

    Returns
    -------
    gym.En
        The initialized and wrapped Maze environment instance.
    """
    # ##: Set random seeds for reproducibility.
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = Maze()

    # ##: Apply wrappers.
    env = ImgObsWrapper(RGBImgObsWrapper(env))

    logger.info(f"Environment initialized with seed: {seed}")
    logger.info(f"Observation Shape: {env.observation_space.shape}, Number of Actions: {env.action_space.n}")

    return env
