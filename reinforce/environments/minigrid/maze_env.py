# -*- coding: utf-8 -*-
"""
Maze environment wrapper for MiniGrid.
"""

from typing import Any, Dict, SupportsFloat, Tuple, Union

import tensorflow as tf
from minigrid.wrappers import RGBImgObsWrapper
from numpy import float32, ndarray

from maze.envs import Maze
from reinforce.core.base_environment import BaseEnvironment


class MazeEnvironment(BaseEnvironment):
    """
    Maze environment wrapper for MiniGrid.

    This class implements the ``BaseEnvironment`` interface for the Maze environment,
    providing a standardized API for agent interaction.
    """

    def __init__(self, use_image_obs: bool = True):
        """
        Initialize the maze environment.

        Parameters
        ----------
        use_image_obs : bool, optional
            Whether to return image observations, by default ``True``.
        """
        self.name = "MazeEnvironment"
        self.use_image_obs = use_image_obs

        self._env = Maze()
        if self.use_image_obs:
            self._env = RGBImgObsWrapper(self._env)

    def reset(self) -> ndarray:
        """
        Reset the environment to an initial state.

        Returns
        -------
        np.ndarray
            The initial observation of the environment.
        """
        obs = {}
        reachable = False

        # ##: Keep resetting until we get a reachable configuration.
        while not reachable:
            obs, _ = self._env.reset(seed=1335)
            reachable = self._env.env.check_objs_reachable(raise_exc=False)

        if self.use_image_obs:
            return obs["image"].astype(float32)
        return obs

    def step(self, action: Union[int, ndarray]) -> Tuple[ndarray, SupportsFloat, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Parameters
        ----------
        action : Union[int, np.ndarray]
            The action to take in the environment.

        Returns
        -------
        Tuple[np.ndarray, SupportsFloat, bool, Dict[str, Any]]
            A tuple containing:
                - The next observation of the environment
                - The reward received for taking the action
                - Whether the episode has terminated
                - Additional information about the step
        """
        if isinstance(action, tf.Tensor):
            action = action.numpy()

        if isinstance(action, ndarray) and action.size == 1:
            action = action.item()

        state, reward, done, truncated, info = self._env.step(action)

        if self.use_image_obs:
            return state["image"].astype(float32), reward, done or truncated, info
        return state, reward, done or truncated, info

    @property
    def action_space(self) -> Any:
        """
        Return the action space of the environment.

        Returns
        -------
        Any
            The action space of the environment.
        """
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        """
        Return the observation space of the environment.

        Returns
        -------
        Any
            The observation space of the environment.
        """
        return self._env.observation_space

    @property
    def name(self) -> str:
        """
        Return the name of the environment.

        Returns
        -------
        str
            The name of the environment.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Set the name of the environment.

        Parameters
        ----------
        value : str
            The name to set.
        """
        self._name = value
