# -*- coding: utf-8 -*-
"""
Base class for Actor-Critic agents like A2C and PPO.
"""

from abc import abstractmethod
from typing import Any, Dict, Tuple, Union

import tensorflow as tf
from keras import Model, optimizers
from numpy import ndarray

from reinforce.agents.base_agent import BaseAgent
from reinforce.configs.models.agent import A2CConfig, PPOConfig
from reinforce.utils.persistence import load_model, save_model
from reinforce.utils.preprocessing import preprocess_observation


class ActorCriticAgent(BaseAgent):
    """
    Abstract base class for Actor-Critic agents (A2C, PPO).

    Handles common functionalities like optimizer setup, persistence delegation, and basic properties.
    Requires the model and persistence handler to be injected. Subclasses must implement the specific
    learning and action selection logic.
    """

    def __init__(self, model: Model, agent_name: str, hyperparameters: Union[A2CConfig, PPOConfig]):
        """
        Initialize the Base Actor-Critic agent with injected model and persistence handler.

        Parameters
        ----------
        model : Model
            The Keras model instance to be used by the agent.
        agent_name : str
            The specific name of the agent (e.g., "A2CAgent", "PPOAgent").
        hyperparameters : A2CConfig | PPOConfig
            Pydantic model containing hyperparameters (e.g., A2CConfig, PPOConfig).
        """
        self._name = agent_name
        self._model = model
        self.hyperparameters = hyperparameters

        # ##: Initialize optimizer with optional LR scheduling.
        if self.hyperparameters.lr_schedule_enabled:
            initial_learning_rate = self.hyperparameters.learning_rate
            lr_schedule = optimizers.schedules.PolynomialDecay(
                initial_learning_rate,
                decay_steps=self.hyperparameters.max_total_steps,
                end_learning_rate=initial_learning_rate * self.hyperparameters.lr_decay_factor,
                power=1.0,
            )
            self._optimizer = optimizers.Adam(learning_rate=lr_schedule)
        else:
            self._optimizer = optimizers.Adam(learning_rate=self.hyperparameters.learning_rate)

    def save(self, path: str) -> None:
        """
        Save the agent's state (model, hyperparameters) using the persistence handler.

        Parameters
        ----------
        path : str
            Directory path to save the agent's state.
        """
        hyperparams = self.hyperparameters.model_dump()
        save_model(path=path, agent_name=self.name, model=self._model, hyperparameters=hyperparams)

    def load(self, path: str) -> None:
        """
        Load the agent's state (model, hyperparameters) using the persistence handler.

        Parameters
        ----------
        path : str
            Directory path to load the agent's state from.
        """
        loaded_model, loaded_hyperparams_dict = load_model(path=path, agent_name=self.name)

        self._model = loaded_model
        self.hyperparameters = self._load_specific_hyperparameters(loaded_hyperparams_dict)
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
    def model(self) -> Model:
        """
        Return the underlying Keras model used by the agent.

        Returns
        -------
        Model
            The Keras model.
        """
        return self._model

    @classmethod
    def _preprocess_observation(cls, observation: ndarray) -> tf.Tensor:
        """
        Preprocess the observation before feeding it to the model.

        Parameters
        ----------
        observation : np.ndarray
            The raw observation from the environment.

        Returns
        -------
        tf.Tensor
            The preprocessed observation tensor.
        """
        processed_observation = preprocess_observation(observation)
        if len(processed_observation.shape) == 3:
            processed_observation = tf.expand_dims(processed_observation, axis=0)
        return processed_observation

    def _forward_pass(self, observation: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        observation : tf.Tensor
            The preprocessed observation tensor.
        training : bool, optional
            Whether the model is in training mode, by default ``True``.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Action logits and value estimate from the model.
        """
        return self._model(observation, training=training)

    def _apply_gradients(self, tape: tf.GradientTape, loss: tf.Tensor) -> None:
        """
        Calculate and apply gradients.

        Parameters
        ----------
        tape : tf.GradientTape
            The gradient tape recording the operations.
        loss : tf.Tensor
            The loss tensor to compute gradients for.
        """
        gradients = tape.gradient(loss, self._model.trainable_variables)
        if self.hyperparameters.max_grad_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.hyperparameters.max_grad_norm)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

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
    def _load_specific_hyperparameters(self, config: Dict[str, Any]) -> Union[A2CConfig, PPOConfig]:
        """
        Load specific hyperparameters for the agent.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration dictionary containing hyperparameters.

        Returns
        -------
        A2CConfig | PPOConfig
            The loaded hyperparameters.
        """
        raise NotImplementedError
