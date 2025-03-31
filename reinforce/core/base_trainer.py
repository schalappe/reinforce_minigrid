# -*- coding: utf-8 -*-
"""
Base interface for all trainers.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from reinforce.core.base_agent import BaseAgent
from reinforce.core.base_environment import BaseEnvironment


class BaseTrainer(ABC):
    """
    Base interface for all trainers.

    This abstract class defines the interface that all trainer implementations must adhere to.
    It provides a standard API for training agents in environments.
    """

    @abstractmethod
    def __init__(
        self,
        agent: BaseAgent,
        environment: BaseEnvironment,
        config: Dict[str, Any],
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        agent : BaseAgent
            The agent to train.
        environment : BaseEnvironment
            The environment to train in.
        config : dict
            Configuration parameters for training.
        callbacks : list of callable, optional
            Callbacks to invoke during training.
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

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load the training state from the specified path.

        Parameters
        ----------
        path : str
            Directory path to load the training state from.
        """
