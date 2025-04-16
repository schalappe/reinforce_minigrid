# -*- coding: utf-8 -*-
"""
Proximal Policy Optimization (PPO) Agent implementation using Keras/TensorFlow.

This module defines the PPOAgent class, which encapsulates the PPO algorithm logic,
including actor and critic networks, training loops, action sampling, and model saving/loading.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import keras
import tensorflow as tf
from keras import Model, models, ops, random
from loguru import logger

from reinforce.ppo.buffer import Buffer
from reinforce.ppo.network import build_actor_critic

# Placeholder for hyperparameters - will be refined later
DEFAULT_HYPERPARAMS = {
    "gamma": 0.99,
    "lam": 0.95,
    "clip_ratio": 0.2,
    "policy_learning_rate": 3e-4,
    "value_function_learning_rate": 1e-3,
    "train_policy_iterations": 80,  # Max policy training epochs per update
    "train_value_iterations": 80,  # Max value function training epochs per update
    "target_kl": 0.01,  # Target KL divergence for early stopping policy training
    "seed": 1337,  # Random seed for reproducibility
    "mini_batch_size": 256,  # Size of mini-batches for training updates
}

# Constant for KL divergence early stopping threshold multiplier
KL_EARLY_STOPPING_MULTIPLIER = 1.5


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.

    Implements the PPO-Clip algorithm. It uses an actor-critic architecture where the actor network learns
    the policy and the critic network learns the value function. Training involves collecting trajectories,
    computing advantages using GAE, and performing multiple epochs of mini-batch updates on both the policy
    and value functions.

    Attributes
    ----------
    observation_space : gym.Space
        The observation space of the environment.
    action_space : gym.Space
        The action space of the environment.
    num_actions : int
        The number of discrete actions.
    hyperparams : Dict[str, Any]
        Dictionary containing hyperparameters for the PPO algorithm.
    mini_batch_size : int
        Size of mini-batches used during training updates.
    network_params : Dict[str, Any]
        Dictionary containing parameters for building the actor-critic networks.
    actor : keras.Model
        The actor network model (policy).
    critic : keras.Model
        The critic network model (value function).
    policy_optimizer : keras.optimizers.Optimizer
        Optimizer for the actor network.
    value_optimizer : keras.optimizers.Optimizer
        Optimizer for the critic network.
    seed_generator : keras.random.SeedGenerator
        Seed generator for reproducible random operations (e.g., action sampling).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hyperparams: Optional[Dict[str, Any]] = None,
        network_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PPO agent.

        Parameters
        ----------
        observation_space : gym.Space
            The environment's observation space (e.g., Box, Discrete).
        action_space : gym.Space
            The environment's action space (must be Discrete for this implementation).
        hyperparams : Optional[Dict[str, Any]], optional
            Dictionary of PPO hyperparameters. If None, uses `DEFAULT_HYPERPARAMS`.
            Defaults to None.
        network_params : Optional[Dict[str, Any]], optional
            Dictionary of parameters to pass to the `build_actor_critic` function.
            If None, uses default network architecture. Defaults to None.

        Raises
        ------
        TypeError
            If the action space is not discrete.
        """
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(f"This PPO implementation requires a Discrete action space, got {action_space}")

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_actions: int = action_space.n

        # ##: Set hyperparameters, using defaults if not provided.
        self.hyperparams = DEFAULT_HYPERPARAMS.copy()
        if hyperparams:
            self.hyperparams.update(hyperparams)
        self.mini_batch_size = self.hyperparams["mini_batch_size"]

        # ##: Set network parameters, using an empty dict if not provided.
        self.network_params = network_params if network_params else {}

        # ##: Build Actor and Critic networks.
        self.actor: Model
        self.critic: Model
        self.actor, self.critic = build_actor_critic(
            self.observation_space.shape, self.num_actions, **self.network_params
        )

        # Initialize optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.hyperparams["policy_learning_rate"])
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.hyperparams["value_function_learning_rate"])

        # Initialize seed generator for reproducibility
        self.seed_generator = keras.random.SeedGenerator(self.hyperparams["seed"])

    def _logprobabilities(self, logits: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        """
        Compute the log probabilities of taking specific actions given logits.

        Parameters
        ----------
        logits : tf.Tensor
            The output logits from the actor network (shape: [batch_size, num_actions]).
        actions : tf.Tensor
            The actions taken (shape: [batch_size]).

        Returns
        -------
        tf.Tensor
            The log probabilities of the given actions under the policy defined by logits
            (shape: [batch_size]).
        """
        logprobabilities_all = ops.log_softmax(logits)
        logprobability = ops.sum(keras.ops.one_hot(actions, self.num_actions) * logprobabilities_all, axis=1)
        return logprobability

    @tf.function
    def sample_action(self, observation: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Sample an action from the policy given an observation.

        Also returns the value estimate from the critic for the observation.

        Parameters
        ----------
        observation : tf.Tensor
            The observation from the environment. Should have the shape defined
            by `observation_space`. If a single observation is passed (less dimensions
            than expected), it will be expanded to include a batch dimension.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            A tuple containing:
            - action: The action sampled from the policy (shape: [1] or [batch_size]).
            - value: The value estimate from the critic (shape: [1] or [batch_size]).
            - logprobability: The log probability of the sampled action (shape: [1] or [batch_size]).
        """
        # ##: Add batch dimension if a single observation is provided.
        if len(observation.shape) == len(self.observation_space.shape):
            observation = tf.expand_dims(observation, axis=0)

        logits = self.actor(observation)
        value = self.critic(observation)

        # ##: Sample action from the categorical distribution defined by logits.
        action = ops.squeeze(random.categorical(logits, 1, seed=self.seed_generator), axis=1)

        # ##: Calculate the log probability of the sampled action.
        logprobability = self._logprobabilities(logits, action)

        # ##: Squeeze value if batch size was 1.
        if value.shape[0] == 1:
            value = ops.squeeze(value, axis=0)
        else:
            value = ops.squeeze(value, axis=-1)

        return action, value, logprobability

    @tf.function
    def train_policy(
        self,
        observation_buffer: tf.Tensor,
        action_buffer: tf.Tensor,
        logprobability_buffer: tf.Tensor,
        advantage_buffer: tf.Tensor,
    ) -> tf.Tensor:
        """
        Perform a single training step for the policy network (actor).

        Uses the PPO-Clip objective function.

        Parameters
        ----------
        observation_buffer : tf.Tensor
            A batch of observations from the buffer.
        action_buffer : tf.Tensor
            A batch of actions corresponding to the observations.
        logprobability_buffer : tf.Tensor
            A batch of log probabilities of the actions taken, computed by the
            policy before the update (pi_old).
        advantage_buffer : tf.Tensor
            A batch of advantage estimates for the state-action pairs.

        Returns
        -------
        tf.Tensor
            The approximate KL divergence between the old policy (before update)
            and the new policy (after update). This is used for early stopping.
        """
        with tf.GradientTape() as tape:
            # ##: Get current policy logits and log probabilities for the actions.
            logits = self.actor(observation_buffer)
            logprobability = self._logprobabilities(logits, action_buffer)

            # ##: Calculate the ratio pi_new / pi_old.
            ratio = ops.exp(logprobability - logprobability_buffer)

            # ##: Calculate the clipped surrogate objective.
            clip_ratio_val = self.hyperparams["clip_ratio"]
            min_advantage = ops.where(
                advantage_buffer > 0,
                (1 + clip_ratio_val) * advantage_buffer,
                (1 - clip_ratio_val) * advantage_buffer,
            )
            # ##: PPO-Clip loss.
            policy_loss = -ops.mean(ops.minimum(ratio * advantage_buffer, min_advantage))

        # ##: Compute and apply gradients.
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        # ##: Calculate approximate KL divergence for early stopping.
        kl_divergence = ops.mean(logprobability_buffer - logprobability)
        return kl_divergence

    @tf.function
    def train_value_function(self, observation_buffer: tf.Tensor, return_buffer: tf.Tensor):
        """
        Perform a single training step for the value network (critic).

        Uses the mean squared error between predicted values and calculated returns.

        Parameters
        ----------
        observation_buffer : tf.Tensor
            A batch of observations from the buffer.
        return_buffer : tf.Tensor
            A batch of calculated returns (rewards-to-go) corresponding to the
            observations.
        """
        with tf.GradientTape() as tape:
            # ##: Get value predictions from the critic.
            values = self.critic(observation_buffer)

            # ##: Ensure value and return buffers have compatible shapes (e.g., [batch, 1]).
            if len(values.shape) == 1:
                values = tf.expand_dims(values, axis=-1)
            if len(return_buffer.shape) == 1:
                return_buffer = tf.expand_dims(return_buffer, axis=-1)

            # ##: Calculate Mean Squared Error loss.
            value_loss = keras.ops.mean((return_buffer - values) ** 2)

        # ##: Compute and apply gradients.
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    @classmethod
    def _prepare_training_data(cls, buffer: Buffer) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Retrieve data from the buffer and convert it to TensorFlow tensors.

        Parameters
        ----------
        buffer : Buffer
            The experience buffer containing trajectories.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
            A tuple containing the training data as tensors:
            (observations, actions, advantages, returns, log_probabilities).
        """
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        # ##: Convert numpy arrays to tensors.
        observation_tensor = tf.convert_to_tensor(observation_buffer, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(action_buffer, dtype=tf.int32)
        advantage_tensor = tf.convert_to_tensor(advantage_buffer, dtype=tf.float32)
        return_tensor = tf.convert_to_tensor(return_buffer, dtype=tf.float32)
        logprobability_tensor = tf.convert_to_tensor(logprobability_buffer, dtype=tf.float32)

        return (
            observation_tensor,
            action_tensor,
            advantage_tensor,
            return_tensor,
            logprobability_tensor,
        )

    def train(self, buffer: Buffer):
        """
        Perform a full PPO training update using data from the buffer.

        This involves multiple epochs of mini-batch updates for both the policy (actor) and value (critic) networks.
        Policy updates use early stopping based on KL divergence.

        Parameters
        ----------
        buffer : Buffer
            The buffer instance containing the trajectories collected during interaction with the environment.
        """
        (
            observation_tensor,
            action_tensor,
            advantage_tensor,
            return_tensor,
            logprobability_tensor,
        ) = self._prepare_training_data(buffer)

        batch_size = tf.shape(observation_tensor)[0]
        indices = tf.range(batch_size)

        # --- Policy (Actor) Training Loop ---
        for i in range(self.hyperparams["train_policy_iterations"]):
            indices = tf.random.shuffle(indices)
            total_kl = tf.constant(0.0, dtype=tf.float32)
            num_processed = tf.constant(0, dtype=tf.int32)

            for start in range(0, batch_size, self.mini_batch_size):
                end = tf.minimum(start + self.mini_batch_size, batch_size)
                minibatch_indices = indices[start:end]
                mb_size = tf.shape(minibatch_indices)[0]

                # ##: Slice data for the mini-batch.
                mb_observation = tf.gather(observation_tensor, minibatch_indices)
                mb_action = tf.gather(action_tensor, minibatch_indices)
                mb_logprobability = tf.gather(logprobability_tensor, minibatch_indices)
                mb_advantage = tf.gather(advantage_tensor, minibatch_indices)

                # ##: Perform one policy update step.
                kl = self.train_policy(
                    mb_observation,
                    mb_action,
                    mb_logprobability,
                    mb_advantage,
                )
                # ##: Accumulate KL divergence weighted by mini-batch size.
                total_kl += kl * tf.cast(mb_size, tf.float32)
                num_processed += mb_size

            # ##: Calculate average KL divergence over the full batch/
            avg_kl = total_kl / tf.cast(num_processed, tf.float32)

            # ##: Early stopping check.
            if avg_kl > KL_EARLY_STOPPING_MULTIPLIER * self.hyperparams["target_kl"]:
                logger.info(
                    f"Policy training epoch {i+1}: KL divergence ({avg_kl:.4f}) exceeded "
                    f"target ({self.hyperparams['target_kl']:.4f}). Stopping early."
                )
                break

        # --- Value Function (Critic) Training Loop ---
        for _ in range(self.hyperparams["train_value_iterations"]):
            indices = tf.random.shuffle(indices)

            for start in range(0, batch_size, self.mini_batch_size):
                end = tf.minimum(start + self.mini_batch_size, batch_size)
                minibatch_indices = indices[start:end]

                # ##: Slice data for the mini-batch.
                mb_observation = tf.gather(observation_tensor, minibatch_indices)
                mb_return = tf.gather(return_tensor, minibatch_indices)

                # ##: Perform one value function update step.
                self.train_value_function(mb_observation, mb_return)

    def save_weights(self, path: Union[str, Path], overwrite: bool = True):
        """
        Save the weights of the actor and critic networks.

        Parameters
        ----------
        path : Union[str, Path]
            The directory path where the weights ('ppo_actor.keras',
            'ppo_critic.keras') will be saved. The directory will be created
            if it doesn't exist.
        overwrite : bool, optional
            Whether to overwrite existing weight files. Defaults to True.
        """
        save_directory = Path(path)
        save_directory.mkdir(parents=True, exist_ok=True)

        actor_path = save_directory / "ppo_actor.keras"
        critic_path = save_directory / "ppo_critic.keras"

        self.actor.save(actor_path, overwrite=overwrite)
        self.critic.save(critic_path, overwrite=overwrite)
        logger.info(f"Agent weights saved to {save_directory}")

    def load_weights(self, path: Union[str, Path]):
        """
        Load the weights for the actor and critic networks.

        Parameters
        ----------
        path : Union[str, Path]
            The directory path from which to load the weights
            ('ppo_actor.keras', 'ppo_critic.keras').

        Raises
        ------
        FileNotFoundError
            If the weight files ('ppo_actor.keras' or 'ppo_critic.keras')
            are not found in the specified directory.
        """
        load_directory = Path(path)

        actor_path = load_directory / "ppo_actor.keras"
        critic_path = load_directory / "ppo_critic.keras"

        if not actor_path.exists():
            raise FileNotFoundError(f"Actor weights not found at {actor_path}")
        if not critic_path.exists():
            raise FileNotFoundError(f"Critic weights not found at {critic_path}")

        self.actor = models.load_model(actor_path, safe_mode=False)
        self.critic = models.load_model(critic_path, safe_mode=False)
        logger.info(f"Agent weights loaded from {load_directory}")
