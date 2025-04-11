# -*- coding: utf-8 -*-
"""
Base class for Actor-Critic agents like A2C and PPO.
"""

import json
import os
from abc import abstractmethod
from typing import Any, Dict, Tuple, Union

from keras import models, optimizers
from numpy import ndarray

from reinforce.agents.actor_critic.core.model import ResNetA2CModel
from reinforce.agents.base_agent import BaseAgent
from reinforce.configs.models import A2CConfig, PPOConfig

HyperparameterConfig = Union[A2CConfig, PPOConfig]


class ActorCriticAgent(BaseAgent):
    """
    Abstract base class for Actor-Critic agents (A2C, PPO).

    Handles common functionalities like model initialization, saving, loading, and basic properties.
    Subclasses must implement the specific learning and action selection logic.
    """

    def __init__(self, action_space: int, hyperparameters: HyperparameterConfig, agent_name: str):
        """
        Initialize the Base Actor-Critic agent.

        Parameters
        ----------
        action_space : int
            Number of possible actions.
        hyperparameters : HyperparameterConfig
            Pydantic model containing hyperparameters (either A2CConfig or PPOConfig).
        agent_name : str
            The specific name of the agent (e.g., "A2CAgent", "PPOAgent").
        """
        self._name = agent_name
        self.action_space = action_space
        self.hyperparameters: HyperparameterConfig = hyperparameters

        # ##: Common model and optimizer setup.
        # ##: Subclasses might override model creation if needed, but use A2CModel as default.
        self._model = ResNetA2CModel(action_space=action_space, embedding_size=self.hyperparameters.embedding_size)
        self._optimizer = optimizers.Adam(learning_rate=self.hyperparameters.learning_rate)

    @abstractmethod
    def act(self, observation: ndarray, training: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action based on the current observation. Must be implemented by subclasses.

        Parameters
        ----------
        observation : np.ndarray
            The current observation from the environment.
        training : bool, optional
            Whether the agent is in training mode, by default ``True``.

        Returns
        -------
        Tuple[int, Dict[str, Any]]
            The selected action and additional information.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, experience_batch: Any) -> Dict[str, Any]:
        """
        Update the agent based on a batch of experiences. Must be implemented by subclasses.

        Parameters
        ----------
        experience_batch : Any
            The format depends on the specific algorithm (e.g., Tuple for A2C, Dict for PPO).

        Returns
        -------
        Dict[str, Any]
            Dictionary of learning metrics.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        Save the agent's model and hyperparameters to the specified path.

        Parameters
        ----------
        path : str
            Directory path to save the agent.
        """
        os.makedirs(path, exist_ok=True)

        # ##: Save the Keras model. Use the agent's specific name.
        model_path = os.path.join(path, f"{self._name}_model.keras")
        self._model.save(model_path)

        # ##: Save the hyperparameters (Pydantic model) to JSON.
        hyperparams_path = os.path.join(path, "hyperparams.json")
        with open(hyperparams_path, "w", encoding="utf-8") as file:
            file.write(self.hyperparameters.model_dump_json(indent=2))
        print(f"Agent {self._name} saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the agent's model and hyperparameters from the specified path.

        Parameters
        ----------
        path : str
            Directory path to load the agent from.
        """
        self._load_model(path)
        self._load_hyperparameters(path)

    def _load_model(self, path: str):
        """
        Load the Keras model from the specified path.

        Parameters
        ----------
        path : str
            Directory path to load the model from.
        """
        model_path = os.path.join(path, f"{self._name}_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self._model = models.load_model(model_path)

    def _load_hyperparameters(self, path: str):
        """
        Load the hyperparameters from the specified path.

        Parameters
        ----------
        path : str
            Directory path to load the hyperparameters from.
        """
        hyperparams_path = os.path.join(path, "hyperparams.json")
        if not os.path.exists(hyperparams_path):
            raise FileNotFoundError(f"Hyperparameters file not found at {hyperparams_path}")
        with open(hyperparams_path, "r", encoding="utf-8") as f:
            loaded_hyperparams_dict = json.load(f)

        self.hyperparameters = self._load_specific_hyperparameters(loaded_hyperparams_dict)
        self.action_space = self.hyperparameters.action_space
        self._optimizer = optimizers.Adam(learning_rate=self.hyperparameters.learning_rate)

    @property
    def name(self) -> str:
        """
        Return the name of the agent.

        Returns
        -------
        str
            The name of the agent.
        """
        return self._name

    @property
    def model(self) -> models.Model:
        """
        Return the underlying model used by the agent.

        Returns
        -------
        models.Model
            The underlying model.
        """
        return self._model

    @abstractmethod
    def _load_specific_hyperparameters(self, config: Dict[str, Any]) -> HyperparameterConfig:
        """
        Load specific hyperparameters for the agent.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration dictionary containing hyperparameters.

        Returns
        -------
        HyperparameterConfig
            The loaded hyperparameters.
        """
        raise NotImplementedError
