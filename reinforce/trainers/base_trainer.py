# -*- coding: utf-8 -*-
"""
Base interface for all trainers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTrainer(ABC):
    """
    Base interface for all trainers.

    This abstract class defines the interface that all trainer implementations must adhere to.
    It provides a standard API for training agents in environments.
    """

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the agent in the environment.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics.
        """

    def evaluate(self, num_episodes: int = 1) -> Dict[str, Any]:
        """
        Evaluate the agent in the environment.

        Parameters
        ----------
        num_episodes : int, default=1
            Number of episodes to evaluate.

        Returns
        -------
        Dict[str, Any]
            Dictionary of evaluation metrics.
        """

    def save_checkpoint(self, path: str) -> None:
        """
        Save the training state to the specified path.

        Parameters
        ----------
        path : str
            Directory path to save the training state.
        """
