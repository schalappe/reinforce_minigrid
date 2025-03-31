# -*- coding: utf-8 -*-
"""
A2C Agent implementation.
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from keras import models, optimizers
from numpy import ndarray

from reinforce.agents.a2c.model import A2CModel
from reinforce.core.base_agent import BaseAgent
from reinforce.utils.preprocessing import preprocess_observation


class A2CAgent(BaseAgent):
    """
    A2C (Advantage Actor-Critic) agent implementation.

    This class implements the BaseAgent interface for the A2C algorithm.
    """

    def __init__(
        self,
        action_space: int,
        embedding_size: int = 128,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        """
        Initialize the A2C agent.

        Parameters
        ----------
        action_space : int
            Number of possible actions.
        embedding_size : int, optional
            Size of the embedding layer, by default 128.
        learning_rate : float, optional
            Learning rate for the optimizer, by default 0.001.
        discount_factor : float, optional
            Discount factor for future rewards, by default 0.99.
        entropy_coef : float, optional
            Entropy regularization coefficient, by default 0.01.
        value_coef : float, optional
            Value loss coefficient, by default 0.5.
        """
        self._name = "A2CAgent"
        self._model = A2CModel(action_space=action_space, embedding_size=embedding_size)
        self._optimizer = optimizers.Adam(learning_rate=learning_rate)

        self.step_counter = 0
        self.action_space = action_space

        # ##: Store hyperparameters.
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

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
            The selected action and additional information about the decision.
        """
        observation = preprocess_observation(observation)

        # ##: Add batch dimension if necessary and pass through model.
        if len(observation.shape) == 3:
            observation = tf.expand_dims(observation, axis=0)
        action_logits, value = self._model(observation)

        if not training:
            action = tf.argmax(action_logits[0]).numpy()
        else:
            action = tf.random.categorical(action_logits, 1)[0, 0].numpy()

        action_probs = tf.nn.softmax(action_logits)

        return action, {
            "value": value[0, 0].numpy(),
            "action_probs": action_probs[0].numpy(),
            "action_logits": action_logits[0].numpy(),
        }

    def learn(
        self, observations: ndarray, actions: ndarray, rewards: ndarray, next_observations: ndarray, dones: ndarray
    ) -> Dict[str, Any]:
        """
        Update the agent based on experiences.

        Parameters
        ----------
        observations : np.ndarray
            Batch of observations.
        actions : np.ndarray
            Batch of actions taken.
        rewards : np.ndarray
            Batch of rewards received.
        next_observations : np.ndarray
            Batch of next observations.
        dones : np.ndarray
            Batch of episode termination flags.

        Returns
        -------
        Dict[str, Any]
            Dictionary of learning metrics.
        """
        # ##: Preprocess observations.
        observations = preprocess_observation(observations)
        next_observations = preprocess_observation(next_observations)

        # ##: Convert numpy arrays to tensors.
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_observations = tf.convert_to_tensor(next_observations, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)  # Convert bool to float

        # ##: Compute returns and advantages.
        returns, advantages = self._compute_returns_and_advantages(rewards, dones, observations, next_observations)

        # ##: Perform one training step.
        metrics = self._train_step(observations, actions, returns, advantages)
        self.step_counter += 1

        return metrics

    def _compute_returns_and_advantages(
        self, rewards: tf.Tensor, dones: tf.Tensor, observations: tf.Tensor, next_observations: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute returns and advantages for the given experiences.

        Parameters
        ----------
        rewards : tf.Tensor
            Batch of rewards.
        dones : tf.Tensor
            Batch of episode termination flags.
        observations : tf.Tensor
            Batch of observations.
        next_observations : tf.Tensor
            Batch of next observations.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Tuple of (returns, advantages).
        """
        # ##: Get value estimates for current and next observations.
        _, values = self._model(observations)
        _, next_values = self._model(next_observations)

        # ##: Reshape values for easier calculations.
        values = tf.squeeze(values)
        next_values = tf.squeeze(next_values)

        # ##: Calculate returns using TD(lambda) with lambda=0 (i.e., TD(0)).
        returns = rewards + self.discount_factor * next_values * (1.0 - dones)

        # ##: Calculate advantages.
        advantages = returns - values

        return returns, advantages

    @tf.function
    def _train_step(
        self, observations: tf.Tensor, actions: tf.Tensor, returns: tf.Tensor, advantages: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Perform one training step.

        Parameters
        ----------
        observations : tf.Tensor
            Batch of observations.
        actions : tf.Tensor
            Batch of actions taken.
        returns : tf.Tensor
            Batch of returns.
        advantages : tf.Tensor
            Batch of advantages.

        Returns
        -------
        Dict[str, tf.Tensor]
            Dictionary of training metrics.
        """
        with tf.GradientTape() as tape:
            # ##: Forward pass.
            action_logits, values = self._model(observations)
            values = tf.squeeze(values)

            # ##: Select the action probabilities for the actions that were taken.
            indices = tf.range(tf.shape(actions)[0])
            action_indices = tf.stack([indices, actions], axis=1)
            action_log_probs = tf.math.log(tf.nn.softmax(action_logits) + 1e-10)
            selected_action_log_probs = tf.gather_nd(action_log_probs, action_indices)

            # ##: Calculate policy loss (negative for gradient ascent).
            policy_loss = -tf.reduce_mean(selected_action_log_probs * advantages)

            # ##: Calculate value loss
            value_loss = self.value_coef * tf.reduce_mean(tf.square(returns - values))

            # ##: Calculate entropy loss (for exploration).
            action_probs = tf.nn.softmax(action_logits)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            entropy_loss = -self.entropy_coef * tf.reduce_mean(entropy)

            # ##: Calculate total loss.
            total_loss = policy_loss + value_loss + entropy_loss

        # ##: Calculate gradients and apply them.
        gradients = tape.gradient(total_loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": tf.reduce_mean(entropy),
        }

    def save(self, path: str) -> None:
        """
        Save the agent to the specified path.

        Parameters
        ----------
        path : str
            Directory path to save the agent.
        """
        # ##: Create directory if it doesn't exist.
        os.makedirs(path, exist_ok=True)

        # ##: Save the model.
        self._model.save(os.path.join(path, f"{self._name}_model.keras"))

        # ##: Save the optimizer state.
        np.save(os.path.join(path, "step_counter.npy"), self.step_counter)

        # ##: Save the hyperparameters.
        hyperparams = {
            "discount_factor": self.discount_factor,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "action_space": self.action_space,
        }
        np.save(os.path.join(path, "hyperparams.npy"), hyperparams)

    def load(self, path: str) -> None:
        """
        Load the agent from the specified path.

        Parameters
        ----------
        path : str
            Directory path to load the agent from.
        """
        # ##: Load the model.
        self._model = models.load_model(os.path.join(path, "model"))

        # ##: Load the optimizer state.
        self.step_counter = np.load(os.path.join(path, "step_counter.npy")).item()

        # ##: Load the hyperparameters.
        hyperparams = np.load(os.path.join(path, "hyperparams.npy"), allow_pickle=True).item()
        self.discount_factor = hyperparams["discount_factor"]
        self.entropy_coef = hyperparams["entropy_coef"]
        self.value_coef = hyperparams["value_coef"]
        self.action_space = hyperparams["action_space"]

    @property
    def name(self) -> str:
        """
        Return the name of the agent.

        Returns
        -------
        str
            Agent name.
        """
        return self._name

    @property
    def model(self) -> A2CModel:
        """
        Return the underlying model used by the agent.

        Returns
        -------
        A2CModel
            The model used by the agent.
        """
        return self._model
