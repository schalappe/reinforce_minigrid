# -*- coding: utf-8 -*-
"""
Base interface for all reinforcement learning agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from numpy import ndarray


class BaseAgent(ABC):
    """
    Base interface for all reinforcement learning agents.

    This abstract class defines the interface that all agent implementations must adhere to.
    It provides a standard API for agent interaction with environments, training, saving, and loading.
    """

    @abstractmethod
    def act(self, observation: ndarray, training: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action based on the current observation.

        Parameters
        ----------
        observation : np.ndarray
            The current observation from the environment.
        training : bool, optional
            Whether the agent is in training mode, by default ``True``.

        Returns
        -------
        Tuple[int, Dict[str, Any]]
           A tuple containing the selected action and additional information.
        """

    @abstractmethod
    def learn(self, experience_batch: Dict[str, ndarray]) -> Dict[str, Any]:
        """
        Update the agent based on a batch of experiences.

        Parameters
        ----------
        experience_batch : Dict[str, np.ndarray]
            A dictionary containing batches of experiences, typically including keys like
            'observations', 'actions', 'rewards', 'next_observations', and 'dones'.

        Returns
        -------
        Dict[str, Any]
            Dictionary of learning metrics.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent to the specified path.

        Parameters
        ----------
        path : str
            Directory path to save the agent.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the agent from the specified path.

        Parameters
        ----------
        path : str
            Directory path to load the agent from.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the agent.

        Returns
        -------
        str
            The name of the agent.
        """

    @property
    @abstractmethod
    def model(self) -> Optional[Any]:
        """
        Return the underlying model used by the agent, if any.

        Returns
        -------
        Any | None
            The model (typically a tf.keras.Model) used by the agent, or ``None`` if not applicable.
        """
