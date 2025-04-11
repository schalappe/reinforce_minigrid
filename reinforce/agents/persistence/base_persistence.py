# -*- coding: utf-8 -*-
"""
Abstract base class for agent persistence handlers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from keras import Model


class AgentPersistence(ABC):
    """
    Abstract base class for agent persistence handlers.
    """

    @abstractmethod
    def save(self, path: str, agent_name: str, model: Model, hyperparameters: Dict[str, Any]) -> None:
        """
        Save the agent's state to the specified path.

        Parameters
        ----------
        path : str
            Directory path to save the agent.
        agent_name : str
            Name of the agent (used for file naming).
        model : Model
            The Keras model to save.
        hyperparameters : Dict[str, Any]
            The agent's hyperparameters.
        """
        pass

    @abstractmethod
    def load(self, path: str, agent_name: str) -> Tuple[Model, Dict[str, Any]]:
        """
        Load the agent's state from the specified path.

        Parameters
        ----------
        path : str
            Directory path to load the agent from.
        agent_name : str
            Name of the agent (used for file naming).

        Returns
        -------
        Tuple[Model, Dict[str, Any]]
            A tuple containing the loaded model, hyperparameters.
        """
        pass
