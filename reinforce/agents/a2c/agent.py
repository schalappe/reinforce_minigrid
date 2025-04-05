# -*- coding: utf-8 -*-
"""
A2C Agent implementation.
"""

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from keras import models, optimizers
from numpy import ndarray

from reinforce.agents.a2c.a2c_model import A2CModel
from reinforce.configs.models import A2CConfig
from reinforce.core.base_agent import BaseAgent
from reinforce.utils.preprocessing import preprocess_observation


class A2CAgent(BaseAgent):
    """
    A2C (Advantage Actor-Critic) agent implementation.

    This class implements the BaseAgent interface for the A2C algorithm.
    """

    def __init__(self, action_space: int, hyperparameters: A2CConfig):
        """
        Initialize the A2C agent.

        Parameters
        ----------
        action_space : int
            Number of possible actions. Should match `hyperparameters.action_space`.
        hyperparameters : A2CConfig
            Pydantic model containing hyperparameters.
        """
        self._name = "A2CAgent"

        # ##: Ensure action_space consistency.
        if action_space != hyperparameters.action_space:
            raise ValueError(
                f"action_space ({action_space}) must match hyperparameters.action_space ({hyperparameters.action_space})"
            )
        self.action_space = action_space
        self.hyperparameters: A2CConfig = hyperparameters

        # ##: Access attributes directly from Pydantic model.
        self._model = A2CModel(action_space=action_space, embedding_size=self.hyperparameters.embedding_size)
        self._optimizer = optimizers.Adam(learning_rate=self.hyperparameters.learning_rate)

        self.step_counter = 0

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

    def learn(self, experience_batch: Dict[str, ndarray]) -> Dict[str, Any]:
        """
        Update the agent based on a batch of experiences.

        Parameters
        ----------
        experience_batch : Dict[str, np.ndarray]
            A dictionary containing batches of experiences:
            'observations', 'actions', 'rewards', 'next_observations', 'dones'.

        Returns
        -------
        Dict[str, Any]
            Dictionary of learning metrics.
        """
        # ##: Unpack experience batch and preprocess observations.
        observations = preprocess_observation(experience_batch["observations"])
        actions = experience_batch["actions"]
        rewards = experience_batch["rewards"]
        next_observations = preprocess_observation(experience_batch["next_observations"])
        dones = experience_batch["dones"]

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
        returns = rewards + self.hyperparameters.discount_factor * next_values * (1.0 - dones)

        # ##: Calculate advantages.
        advantages = returns - values

        return returns, advantages

    def _calculate_losses(
        self,
        action_logits: tf.Tensor,
        values: tf.Tensor,
        actions: tf.Tensor,
        returns: tf.Tensor,
        advantages: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculate policy, value, entropy, and total losses."""
        # ##: Select the action probabilities for the actions that were taken.
        indices = tf.range(tf.shape(actions)[0])
        action_indices = tf.stack([indices, actions], axis=1)
        action_log_probs = tf.math.log(tf.nn.softmax(action_logits) + 1e-10)
        selected_action_log_probs = tf.gather_nd(action_log_probs, action_indices)

        # ##: Calculate policy loss (negative for gradient ascent).
        policy_loss = -tf.reduce_mean(selected_action_log_probs * advantages)

        # ##: Calculate value loss.
        value_loss = self.hyperparameters.value_coef * tf.reduce_mean(tf.square(returns - values))

        # ##: Calculate entropy loss (for exploration).
        action_probs = tf.nn.softmax(action_logits)
        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
        entropy_loss = -self.hyperparameters.entropy_coef * tf.reduce_mean(entropy)

        # ##: Calculate total loss.
        total_loss = policy_loss + value_loss + entropy_loss

        return total_loss, policy_loss, value_loss, entropy_loss, tf.reduce_mean(entropy)

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

            # ##: Calculate losses using the helper method.
            total_loss, policy_loss, value_loss, entropy_loss, entropy = self._calculate_losses(
                action_logits, values, actions, returns, advantages
            )

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

        # ##: Save the optimizer state (weights are saved separately).
        # Note: Optimizer state saving/loading might require more complex handling
        # depending on the TF version and optimizer type. For simplicity, we omit it here
        # but save the step counter which might be related.
        np.save(os.path.join(path, "step_counter.npy"), self.step_counter)

        # ##: Save the hyperparameters by dumping the Pydantic model to JSON.
        hyperparams_path = os.path.join(path, "hyperparams.json")
        with open(hyperparams_path, "w", encoding="utf-8") as file:
            file.write(self.hyperparameters.model_dump_json(indent=2))

    def load(self, path: str) -> None:
        """
        Load the agent from the specified path.

        Parameters
        ----------
        path : str
            Directory path to load the agent from.
        """
        # ##: Load the model.
        self._model = models.load_model(os.path.join(path, f"{self._name}_model.keras"))

        # ##: Load the optimizer state (see note in save method).
        self.step_counter = np.load(os.path.join(path, "step_counter.npy")).item()

        # ##: Load the hyperparameters from JSON and parse into Pydantic model.
        hyperparams_path = os.path.join(path, "hyperparams.json")
        with open(hyperparams_path, "r", encoding="utf-8") as f:
            loaded_hyperparams_dict = json.load(f)
        self.hyperparameters = A2CConfig(**loaded_hyperparams_dict)
        self.action_space = self.hyperparameters.action_space

        # ##: Re-initialize optimizer with loaded learning rate.
        self._optimizer = optimizers.Adam(learning_rate=self.hyperparameters.learning_rate)

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
