"""
Experience replay buffer for PPO, optimized for vectorized environments.

Stores trajectories from multiple parallel environments and calculates advantages and returns using
Generalized Advantage Estimation (GAE).
"""

from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger

from reinforce.core.base_buffer import BaseBuffer


class Buffer(BaseBuffer):
    """
    A buffer for storing trajectories from parallel environments and calculating GAE.

    This buffer uses pre-allocated NumPy arrays for efficiency when working with vectorized
    environments. It collects experiences (states, actions, rewards, etc.) from `num_envs`
    environments over `steps_per_env` steps and computes the Generalized Advantage Estimation (GAE)
    and returns for PPO training.
    """

    def __init__(self, obs_shape: tuple, num_envs: int, steps_per_env: int, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize the buffer with pre-allocated arrays.

        Parameters
        ----------
        obs_shape : tuple
            Shape of a single observation from the environment.
        num_envs : int
            Number of parallel environments.
        steps_per_env : int
            Number of steps collected from each environment per update cycle.
        gamma : float, optional
            Discount factor. Default is 0.99.
        lam : float, optional
            GAE lambda parameter. Default is 0.95.
        """
        capacity = num_envs * steps_per_env
        super().__init__(obs_shape, capacity)

        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.gamma = gamma
        self.lam = lam
        self.buffer_size = capacity

        # ##: Pre-allocate NumPy arrays for efficiency.
        self.states = np.zeros((self.steps_per_env, self.num_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.steps_per_env, self.num_envs), dtype=np.int32)
        self.rewards = np.zeros((self.steps_per_env, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.steps_per_env, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.steps_per_env, self.num_envs), dtype=np.float32)
        self.action_probs = np.zeros((self.steps_per_env, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.steps_per_env, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.steps_per_env, self.num_envs), dtype=np.float32)

        self.ptr = 0
        self.trajectory_ready = False

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: np.ndarray,
        action_prob: np.ndarray,
    ):
        """
        Store a batch of transitions from parallel environments at the current pointer position.

        Parameters
        ----------
        state : np.ndarray
            Batch of states observed. Shape: (num_envs, *obs_shape)
        action : np.ndarray
            Batch of actions taken. Shape: (num_envs,)
        reward : np.ndarray
            Batch of rewards received. Shape: (num_envs,)
        value : np.ndarray
            Batch of value estimates of the states. Shape: (num_envs,)
        done : np.ndarray
            Batch of done flags (boolean or int). Shape: (num_envs,)
        action_prob : np.ndarray
            Batch of log probabilities of the actions taken. Shape: (num_envs,)

        Raises
        ------
        IndexError
            If trying to store beyond the buffer capacity (steps_per_env).
        """
        if self.ptr >= self.steps_per_env:
            raise IndexError(f"Buffer full. Tried to store at index {self.ptr} with capacity {self.steps_per_env}.")

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done.astype(np.float32)
        self.action_probs[self.ptr] = action_prob

        self.ptr += 1
        self.trajectory_ready = False

    def compute_advantages_and_returns(self, last_values: np.ndarray):
        """
        Computes advantages and returns for the stored trajectories using GAE.

        Operates across all parallel environments simultaneously.

        Parameters
        ----------
        last_values : np.ndarray
            The value estimates of the final state in each parallel trajectory.
            Shape: (num_envs,).
        """
        if self.ptr != self.steps_per_env:
            logger.warning(
                f"Computing advantages with incomplete buffer ({self.ptr}/{self.steps_per_env} steps). "
                "This might happen at the end of training."
            )

        values_with_last = np.vstack((self.values[: self.ptr], last_values.reshape(1, self.num_envs)))

        gae = 0.0
        # ##: Calculate advantages working backwards.
        for t in reversed(range(self.ptr)):
            # ##: Delta = R_t + gamma * V(S_{t+1}) * (1 - Done_t) - V(S_t)
            delta = (
                self.rewards[t] + self.gamma * values_with_last[t + 1] * (1.0 - self.dones[t]) - values_with_last[t]
            )
            # ##: GAE = Delta_t + gamma * lambda * (1 - Done_t) * GAE_{t+1}
            gae = delta + self.gamma * self.lam * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae

        # ##: Calculate returns (target for value function) = Advantages + Values
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

        # ##: Normalize advantages across the entire batch of collected data.
        flat_advantages = self.advantages[: self.ptr].flatten()
        mean_adv = np.mean(flat_advantages)
        std_adv = np.std(flat_advantages)
        self.advantages[: self.ptr] = (self.advantages[: self.ptr] - mean_adv) / (std_adv + 1e-8)

        self.trajectory_ready = True

    def _flatten_buffer(self) -> tuple:
        """
        Flattens the buffer arrays from (steps, envs, ...) to (total_steps, ...).

        Returns
        -------
        tuple
            Tuple containing flattened arrays of states, actions, action_probs, returns, advantages, and values.
        """
        total_steps = self.ptr * self.num_envs

        states_flat = self.states[: self.ptr].swapaxes(0, 1).reshape((total_steps,) + self.obs_shape)
        actions_flat = self.actions[: self.ptr].swapaxes(0, 1).reshape(total_steps)
        action_probs_flat = self.action_probs[: self.ptr].swapaxes(0, 1).reshape(total_steps)
        returns_flat = self.returns[: self.ptr].swapaxes(0, 1).reshape(total_steps)
        advantages_flat = self.advantages[: self.ptr].swapaxes(0, 1).reshape(total_steps)
        values_flat = self.values[: self.ptr].swapaxes(0, 1).reshape(total_steps)

        return states_flat, actions_flat, action_probs_flat, returns_flat, advantages_flat, values_flat

    def get_batches(self, batch_size: int) -> tf.data.Dataset:
        """
        Creates a TensorFlow Dataset for iterating over mini-batches from the flattened buffer.

        Parameters
        ----------
        batch_size : int
            The size of each mini-batch.

        Returns
        -------
        tf.data.Dataset
            A TensorFlow Dataset yielding batches of (states, actions, action_probs, returns, advantages, values).

        Raises
        ------
        ValueError
            If advantages and returns have not been computed yet.
        """
        if not self.trajectory_ready:
            raise ValueError("Advantages and returns must be computed before getting batches.")
        if self.ptr == 0:
            logger.warning("Attempting to get batches from an empty buffer.")
            return tf.data.Dataset.from_tensor_slices((
                tf.zeros((0,) + self.obs_shape),
                tf.zeros(0, dtype=tf.int32),
                tf.zeros(0),
                tf.zeros(0),
                tf.zeros(0),
                tf.zeros(0),
            )).batch(batch_size)

        # ##>: Flatten data across environments and steps.
        states_f, actions_f, action_probs_f, returns_f, advantages_f, values_f = self._flatten_buffer()

        # ##>: Convert flattened numpy arrays to tensors.
        states_tensor = tf.convert_to_tensor(states_f, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions_f, dtype=tf.int32)
        action_probs_tensor = tf.convert_to_tensor(action_probs_f, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns_f, dtype=tf.float32)
        advantages_tensor = tf.convert_to_tensor(advantages_f, dtype=tf.float32)
        values_tensor = tf.convert_to_tensor(values_f, dtype=tf.float32)

        # ##>: Create dataset from flattened tensors.
        dataset = tf.data.Dataset.from_tensor_slices(
            (states_tensor, actions_tensor, action_probs_tensor, returns_tensor, advantages_tensor, values_tensor)
        )

        # ##>: Shuffle, batch, and prefetch for GPU efficiency.
        total_samples = self.ptr * self.num_envs
        dataset = dataset.shuffle(buffer_size=total_samples).batch(batch_size)
        # ##>: Prefetch allows loading next batch while GPU trains on current batch.
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def clear(self) -> None:
        """Resets the buffer pointer and trajectory ready flag."""
        self.ptr = 0
        self.trajectory_ready = False

    def sample(self, batch_size: int) -> tuple[Any, ...]:
        """
        Sample a batch from the buffer.

        For PPO, use get_batches() instead since PPO uses all data with shuffling.
        This method is provided for interface compatibility.

        Parameters
        ----------
        batch_size : int
            Not used in PPO buffer.

        Returns
        -------
        tuple
            Flattened buffer contents.
        """
        return self._flatten_buffer()

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self.ptr * self.num_envs
