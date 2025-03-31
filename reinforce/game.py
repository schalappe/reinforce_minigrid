# -*- coding: utf-8 -*-
"""
Environment that implements the rule of 2048.
"""
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from minigrid.wrappers import RGBImgObsWrapper
from maze.envs import Maze

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class MazeGame:
    """The Maze environment."""

    def __init__(self):
        self.environment = RGBImgObsWrapper(Maze())

    def reset(self) -> np.ndarray:
        """
        Reinitialize the environment.

        Returns
        -------
        ndarray
            Initial state of environments
        """
        obs = {}
        reachable = False
        while not reachable:
            obs, _ = self.environment.reset(seed=1335)
            reachable = self.environment.env.check_objs_reachable(raise_exc=False)
        return obs["image"].astype(np.float32)

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns state, reward and done flag given an action.

        Parameters
        ----------
        action: ndarray
            Action to apply to environment

        Returns
        -------
        Tuple[ndarray, ndarray, ndarray]
            Next step, reward, done flag
        """

        state, _, done, _, _ = self.environment.step(action)
        reward = self.environment.env.reward()
        return state["image"].astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32)

    def step(self, action: tf.Tensor) -> List[tf.Tensor]:
        """
        Returns state, reward and done flag given an action.

        Parameters
        ----------
        action: Tensor
            Action to apply to environment

        Returns
        -------
        List[Tensor]
            Next step, reward, done flag
        """
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.int32])
