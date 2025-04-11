# -*- coding: utf-8 -*-
"""
Base class for Actor-Critic agents like A2C and PPO.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

from keras import Model, optimizers
from numpy import ndarray

from reinforce.agents.base_agent import BaseAgent
from reinforce.agents.persistence import AgentPersistence, KerasFilePersistence
from reinforce.configs.models.agent import A2CConfig, AgentConfig, PPOConfig

# ##: Type alias for hyperparameters, now using the base AgentConfig
HyperparameterConfig = Union[A2CConfig, PPOConfig, AgentConfig]


class ActorCriticAgent(BaseAgent):
    """
    Abstract base class for Actor-Critic agents (A2C, PPO).

    Handles common functionalities like optimizer setup, persistence delegation, and basic properties.
    Requires the model and persistence handler to be injected. Subclasses must implement the specific
    learning and action selection logic.
    """

    def __init__(
        self,
        model: Model,
        agent_name: str,
        hyperparameters: HyperparameterConfig,
        persistence_handler: Optional[AgentPersistence] = None,
    ):
        """
        Initialize the Base Actor-Critic agent with injected model and persistence handler.

        Parameters
        ----------
        model : Model
            The Keras model instance to be used by the agent.
        agent_name : str
            The specific name of the agent (e.g., "A2CAgent", "PPOAgent").
        hyperparameters : HyperparameterConfig
            Pydantic model containing hyperparameters (e.g., A2CConfig, PPOConfig).
        persistence_handler : AgentPersistence, optional
            An instance of AgentPersistence to handle saving/loading.
            If None, defaults to `KerasFilePersistence`, by default None.
        """
        self._name = agent_name
        self._hyperparameters = hyperparameters

        # ##: Inject model and persistence handler
        self._model = model
        self._persistence_handler = persistence_handler or KerasFilePersistence()

        # ##: Common optimizer setup.
        self._optimizer = optimizers.Adam(learning_rate=self._hyperparameters.learning_rate)

    def save(self, path: str) -> None:
        """
        Save the agent's state (model, hyperparameters) using the persistence handler.

        Parameters
        ----------
        path : str
            Directory path to save the agent's state.
        """
        hyperparams = self._hyperparameters.model_dump()
        self._persistence_handler.save(path=path, agent_name=self.name, model=self._model, hyperparameters=hyperparams)

    def load(self, path: str) -> None:
        """
        Load the agent's state (model, hyperparameters) using the persistence handler.

        Parameters
        ----------
        path : str
            Directory path to load the agent's state from.
        """
        loaded_model, loaded_hyperparams_dict = self._persistence_handler.load(path=path, agent_name=self.name)

        # ##: Assign the loaded model.
        self._model = loaded_model

        # ##: Parse the loaded hyperparameter dict into the specific Pydantic model.
        self._hyperparameters = self._load_specific_hyperparameters(loaded_hyperparams_dict)

        # ##: Re-initialize action space and optimizer based on loaded hyperparameters.
        self._optimizer = optimizers.Adam(learning_rate=self._hyperparameters.learning_rate)

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
    def model(self) -> Model:
        """
        Return the underlying Keras model used by the agent.

        Returns
        -------
        Model
            The Keras model.
        """
        return self._model

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
