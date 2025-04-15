# -*- coding: utf-8 -*-
"""PPO Agent implementation."""

from pathlib import Path
from typing import Tuple, Union

import keras
import tensorflow as tf
from keras import models, ops, random

from reinforce.ppo.buffer import Buffer
from reinforce.ppo.network import build_actor_critic

# Placeholder for hyperparameters - will be refined later
DEFAULT_HYPERPARAMS = {
    "gamma": 0.99,
    "lam": 0.95,
    "clip_ratio": 0.2,
    "policy_learning_rate": 3e-4,
    "value_function_learning_rate": 1e-3,
    "train_policy_iterations": 80,
    "train_value_iterations": 80,
    "target_kl": 0.01,
    "seed": 1337,
    "mini_batch_size": 64,  # Added for mini-batch updates
}


class PPOAgent:
    """Proximal Policy Optimization Agent."""

    def __init__(
        self,
        observation_space,
        action_space,
        hyperparams=None,
        network_params=None,
    ):
        """
        Initialize the PPO agent.

        Args:
            observation_space: The environment's observation space.
            action_space: The environment's action space.
            hyperparams (dict, optional): PPO hyperparameters. Defaults to DEFAULT_HYPERPARAMS.
            network_params (dict, optional): Parameters for the neural network architecture.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_actions = action_space.n

        # ##: Use default hyperparameters if none provided.
        self.hyperparams = DEFAULT_HYPERPARAMS.copy()
        if hyperparams:
            self.hyperparams.update(hyperparams)
        self.mini_batch_size = self.hyperparams["mini_batch_size"]

        # ##: Use default network parameters if none provided.
        self.network_params = network_params if network_params else {}

        # ##: Initialize Actor and Critic networks.
        self.actor, self.critic = build_actor_critic(
            self.observation_space.shape, self.num_actions, **self.network_params
        )

        # ##: Initialize optimizers.
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.hyperparams["policy_learning_rate"])
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.hyperparams["value_function_learning_rate"])

        # ##: Seed generator for reproducibility.
        self.seed_generator = keras.random.SeedGenerator(self.hyperparams["seed"])

    def _logprobabilities(self, logits: tf.Tensor, actions: tf.Tensor):
        """
        Compute the log-probabilities of taking actions given logits.

        Parameters
        ----------
        logits : tf.Tensor
            Logits output from the actor network.
        actions : tf.Tensor
            Actions taken by the agent.

        Returns
        -------
        tf.Tensor
            Log-probabilities of the actions.
        """
        logprobabilities_all = ops.log_softmax(logits)
        logprobability = ops.sum(keras.ops.one_hot(actions, self.num_actions) * logprobabilities_all, axis=1)
        return logprobability

    @tf.function
    def sample_action(self, observation: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample an action from the policy given an observation.

        Parameters
        ----------
        observation : tf.Tensor
            The observation from the environment.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Logits and sampled action.
        """
        if len(observation.shape) == len(self.observation_space.shape):
            observation = tf.expand_dims(observation, axis=0)

        logits = self.actor(observation)
        action = ops.squeeze(random.categorical(logits, 1, seed=self.seed_generator), axis=1)
        return logits, action

    @tf.function
    def train_policy(
        self,
        observation_buffer: tf.Tensor,
        action_buffer: tf.Tensor,
        logprobability_buffer: tf.Tensor,
        advantage_buffer: tf.Tensor,
    ):
        """
        Train the policy network.

        Parameters
        ----------
        observation_buffer : tf.Tensor
            Buffer of observations.
        action_buffer : tf.Tensor
           Buffer of actions.
        logprobability_buffer : tf.Tensor
            Buffer of log-probabilities.
        advantage_buffer : tf.Tensor
            Buffer of advantages.

        Returns
        -------
        tf.Tensor
            KL divergence between old and new policies.
        """
        with tf.GradientTape() as tape:
            logits = self.actor(observation_buffer)
            logprobability = self._logprobabilities(logits, action_buffer)
            ratio = ops.exp(logprobability - logprobability_buffer)

            min_advantage = ops.where(
                advantage_buffer > 0,
                (1 + self.hyperparams["clip_ratio"]) * advantage_buffer,
                (1 - self.hyperparams["clip_ratio"]) * advantage_buffer,
            )

            policy_loss = -ops.mean(ops.minimum(ratio * advantage_buffer, min_advantage))

        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        # ##: Calculate KL divergence for early stopping.
        return ops.sum(ops.mean(logprobability_buffer - logprobability))

    @tf.function
    def train_value_function(self, observation_buffer: tf.Tensor, return_buffer: tf.Tensor):
        """
        Train the value network.

        Parameters
        ----------
        observation_buffer : tf.Tensor
            Buffer of observations.
        return_buffer : tf.Tensor
            Buffer of returns (rewards-to-go).
        """
        with tf.GradientTape() as tape:
            values = self.critic(observation_buffer)
            if len(values.shape) == 1:
                values = tf.expand_dims(values, axis=-1)
            if len(return_buffer.shape) == 1:
                return_buffer = tf.expand_dims(return_buffer, axis=-1)

            value_loss = keras.ops.mean((return_buffer - values) ** 2)

        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def train(self, buffer: Buffer):
        """
        Perform one training update using data from the buffer.

        Parameters
        ----------
        buffer : Buffer
            The buffer containing the collected trajectories.
        """
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        # ##: Convert numpy arrays to tensors.
        observation_buffer = tf.convert_to_tensor(observation_buffer, dtype=tf.float32)
        action_buffer = tf.convert_to_tensor(action_buffer, dtype=tf.int32)
        advantage_buffer = tf.convert_to_tensor(advantage_buffer, dtype=tf.float32)
        return_buffer = tf.convert_to_tensor(return_buffer, dtype=tf.float32)
        logprobability_buffer = tf.convert_to_tensor(logprobability_buffer, dtype=tf.float32)

        batch_size = observation_buffer.shape[0]
        indices = tf.range(batch_size)

        # ##: Update policy with early stopping, using mini-batches.
        for i in range(self.hyperparams["train_policy_iterations"]):
            indices = tf.random.shuffle(indices)
            total_kl = 0.0
            num_processed = 0
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                minibatch_indices = indices[start:end]
                mb_size = tf.shape(minibatch_indices)[0]

                # ##: Slice data for the mini-batch.
                mb_observation = tf.gather(observation_buffer, minibatch_indices)
                mb_action = tf.gather(action_buffer, minibatch_indices)
                mb_logprobability = tf.gather(logprobability_buffer, minibatch_indices)
                mb_advantage = tf.gather(advantage_buffer, minibatch_indices)

                kl = self.train_policy(
                    mb_observation,
                    mb_action,
                    mb_logprobability,
                    mb_advantage,
                )
                total_kl += kl * tf.cast(mb_size, tf.float32)
                num_processed += mb_size

            # ##: Average KL divergence over the full batch.
            avg_kl = total_kl / tf.cast(num_processed, tf.float32)

            if avg_kl > 1.5 * self.hyperparams["target_kl"]:
                print(
                    f"  Policy training epoch {i+1}: KL divergence ({avg_kl:.4f}) exceeded target ({self.hyperparams['target_kl']:.4f}). Stopping early."
                )
                break

        # ##: Update value function using mini-batches.
        for _ in range(self.hyperparams["train_value_iterations"]):
            indices = tf.random.shuffle(indices)  # Re-shuffle for value function updates
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                minibatch_indices = indices[start:end]

                # ##: Slice data for the mini-batch.
                mb_observation = tf.gather(observation_buffer, minibatch_indices)
                mb_return = tf.gather(return_buffer, minibatch_indices)

                self.train_value_function(mb_observation, mb_return)

    def save_weights(self, path: Union[str, Path]):
        """
        Save the actor and critic network weights.

        Parameters
        ----------
        path : Path | str
            The directory path where the weights will be saved.
        """
        save_directory = Path(path)
        save_directory.mkdir(parents=True, exist_ok=True)

        self.actor.save(save_directory / "ppo_actor.keras")
        self.critic.save(save_directory / "ppo_critic.keras")
        print(f"Agent weights saved to {save_directory}")

    def load_weights(self, path: Union[str, Path]):
        """
        Load the actor and critic network weights.

        Parameters
        ----------
        path : Path | str
            The directory path where the weights are saved.
        """
        load_directory = Path(path)

        # ##: Load the Keras model.
        actor_path = load_directory / "ppo_actor.keras"
        critic_path = load_directory / "ppo_critic.keras"
        if not actor_path.exists() or not critic_path.exists():
            raise FileNotFoundError(f"Model files not found at {load_directory}")

        self.actor = models.load_model(actor_path, safe_mode=False)
        self.critic = models.load_model(critic_path, safe_mode=False)
        print(f"Agent weights loaded from {load_directory}")
