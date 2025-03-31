# -*- coding: utf-8 -*-
"""
Base interface for all environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

from numpy import ndarray


class BaseEnvironment(ABC):
    """
    Base interface for all environments.

    This abstract class defines the interface that all environment implementations must adhere to.
    It provides a standard API for agent-environment interactions.
    """

    @abstractmethod
    def reset(self) -> ndarray:
        """
        Reset the environment to an initial state.

        Returns
        -------
        np.ndarray
            The initial observation of the environment.
        """

    @abstractmethod
    def step(self, action: Union[int, ndarray]) -> Tuple[ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Parameters
        ----------
        action : Union[int, np.ndarray]
            The action to take in the environment.

        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict[str, Any]]
            A tuple containing:
                - The next observation of the environment
                - The reward received for taking the action
                - Whether the episode has terminated
                - Additional information about the step
        """

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """
        Return the action space of the environment.

        Returns
        -------
        Any
            The action space specification.
        """

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """
        Return the action space of the environment.

        Returns
        -------
        Any
            The action space specification.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the environment.

        Returns
        -------
        str
            The name of the environment.
        """
