# -*- coding: utf-8 -*-
"""
Rollout buffer for on-policy algorithms like PPO.
"""

from typing import Dict, Generator, Tuple

import numpy as np
import tensorflow as tf
from loguru import logger

from reinforce.utils.management import setup_logger
from reinforce.utils.preprocessing import preprocess_observation

setup_logger()


class RolloutBuffer:
    """
    Rollout buffer for storing trajectories and computing GAE for PPO.

    Stores experiences (observations, actions, rewards, dones, values, log_probs) collected during
    rollouts and computes advantages and returns using GAE. Provides an iterator for sampling
    mini-batches during the PPO update phase.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize the Rollout Buffer.

        Parameters
        ----------
        buffer_size : int
            Maximum number of steps to store in the buffer (rollout length).
        observation_shape : tuple
            Shape of a single observation.
        action_shape : tuple
            Shape of a single action.
        gamma : float, optional
            Discount factor, by default 0.99.
        gae_lambda : float, optional
            Factor for Generalized Advantage Estimation (GAE), by default 0.95.
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # ##: Buffer storage (using lists for flexibility, convert to numpy/tf later).
        # ##: Store raw observations first, preprocess when retrieving batch.
        self.observations = [None] * self.buffer_size
        self.actions = np.zeros((self.buffer_size, *self.action_shape), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)  # Use float for calculations
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)

        # ##: Computed values.
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        Add a single step of experience to the buffer.

        Parameters
        ----------
        obs : np.ndarray
            Observation from the environment (raw).
        action : int
            Action taken by the agent.
        reward : float
            Reward received.
        done : bool
            Whether the episode terminated.
        value : float
            Value estimate V(s) from the critic.
        log_prob : float
            Log probability log(pi(a|s)) of the action taken.
        """
        if self.ptr >= self.buffer_size:
            logger.warning("Rollout buffer overflow. Check trainer logic.")
            self.ptr = 0

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).

        Must be called after a rollout is complete (buffer is full or episode ended early).

        Parameters
        ----------
        last_value : float
            Value estimate V(s') for the state after the last step in the buffer.
            Used for bootstrapping if the episode didn't end.
        last_done : bool
            Whether the episode terminated after the last step in the buffer.
        """
        if not self.full:
            logger.warning("Computing GAE on a partially filled buffer. Ensure this is intended.")

        last_value = last_value * (1.0 - float(last_done))
        last_gae_lam = 0.0

        # ##: Iterate backwards through the buffer.
        for t in reversed(range(self.buffer_size)):
            # ##: Get value estimate for current and next step.
            value_t = self.values[t]

            # ##: Use bootstrapped value for step after buffer end, else use stored value.
            value_t_plus_1 = self.values[t + 1] if t < self.buffer_size - 1 else last_value

            # ##: If step t led to a terminal state, V(s_{t+1}) is 0.
            value_t_plus_1 = value_t_plus_1 * (1.0 - self.dones[t])

            # ##: Calculate TD error (delta).
            delta = self.rewards[t] + self.gamma * value_t_plus_1 - value_t

            # ##: Calculate GAE(gamma, lambda).
            last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * last_gae_lam
            self.advantages[t] = last_gae_lam

        # ##: Compute returns by adding advantages to values.
        self.returns = self.advantages + self.values

        # ##: Normalize advantages (optional but recommended).
        # ##: Check for zero std deviation before normalizing.
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        if adv_std > 1e-8:
            self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

    def get_batch(self) -> Dict[str, tf.Tensor]:
        """
        Get the entire buffer's data as TensorFlow tensors.

        Preprocesses observations before converting to tensor.
        Assumes GAE has already been computed.

        Returns
        -------
        Dict[str, tf.Tensor]
            Dictionary containing tensors for:
            'observations', 'actions', 'advantages', 'returns', 'log_probs_old', 'values_old'.
        """
        if not self.full:
            logger.warning("Retrieving batch from a partially filled buffer.")

        # ##: Ensure advantages and returns have been computed.
        if np.all(self.advantages == 0) or np.all(self.returns == 0):
            logger.info("Advantages/Returns seem uncomputed. Call compute_returns_and_advantages first.")

        # ##: Preprocess observations before creating tensor.
        valid_observations = [obs for obs in self.observations if obs is not None]
        if len(valid_observations) != self.buffer_size:
            logger.warning(
                f"Mismatch in observation count ({len(valid_observations)} vs {self.buffer_size}). Check buffer filling."
            )
            indices = np.arange(len(valid_observations))
        else:
            indices = np.arange(self.buffer_size)

        processed_obs = tf.stack([preprocess_observation(obs) for obs in valid_observations])

        # ##: Convert numpy arrays to tensors.
        batch = {
            "observations": processed_obs,
            "actions": tf.convert_to_tensor(self.actions[indices], dtype=tf.int32),
            "advantages": tf.convert_to_tensor(self.advantages[indices], dtype=tf.float32),
            "returns": tf.convert_to_tensor(self.returns[indices], dtype=tf.float32),
            "log_probs_old": tf.convert_to_tensor(self.log_probs[indices], dtype=tf.float32),
            "values_old": tf.convert_to_tensor(self.values[indices], dtype=tf.float32),
        }
        return batch

    def sample_mini_batches(self, batch_size: int, n_epochs: int) -> Generator[Dict[str, tf.Tensor], None, None]:
        """
        Generator that yields mini-batches sampled from the buffer for PPO updates.

        Shuffles the data at the start of each epoch.

        Parameters
        ----------
        batch_size : int
            Size of each minibatch.
        n_epochs : int
            Number of epochs to iterate over the data.

        Yields
        ------
        Generator[Dict[str, tf.Tensor], None, None]
            A generator yielding dictionaries, each containing a minibatch of tensors.
        """
        full_batch = self.get_batch()
        buffer_size = tf.shape(full_batch["observations"])[0].numpy()
        indices = np.arange(buffer_size)

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, buffer_size, batch_size):
                end_idx = min(start_idx + batch_size, buffer_size)
                minibatch_indices = indices[start_idx:end_idx]

                minibatch = {key: tf.gather(value, minibatch_indices) for key, value in full_batch.items()}
                yield minibatch

    def clear(self) -> None:
        """Reset the buffer pointer and clear computed values."""
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    def is_full(self) -> bool:
        """
        Check if the buffer has reached its capacity.

        Returns
        -------
        bool
            True if the buffer is full, False otherwise.
        """
        return self.full

    def size(self) -> int:
        """
        Return the current number of elements in the buffer.

        Returns
        -------
        int
            The number of elements in the buffer.
        """
        return self.ptr
