# -*- coding: utf-8 -*-
"""
Rollout buffer for on-policy algorithms like PPO, with GAE computation.
"""

from typing import Dict, Generator, List, Tuple, Union

import numpy as np
import tensorflow as tf
from loguru import logger

from reinforce.utils.management import setup_logger
from reinforce.utils.preprocessing import preprocess_observation

setup_logger()


class RolloutBuffer:
    """
    Rollout buffer optimized for on-policy algorithms like PPO.
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

        # ##: Use dictionary-based storage with pre-allocated numpy arrays.
        self.data = {
            "observations": np.zeros((self.buffer_size, *self.observation_shape), dtype=np.float32),
            "actions": np.zeros((self.buffer_size, *self.action_shape), dtype=np.int32),
            "rewards": np.zeros((self.buffer_size,), dtype=np.float32),
            "dones": np.zeros((self.buffer_size,), dtype=np.float32),
            "values": np.zeros((self.buffer_size,), dtype=np.float32),
            "log_probs": np.zeros((self.buffer_size,), dtype=np.float32),
            "advantages": np.zeros((self.buffer_size,), dtype=np.float32),
            "returns": np.zeros((self.buffer_size,), dtype=np.float32),
        }

        self.ptr = 0
        self.path_start_idx = 0
        self.full = False
        self.episode_ends: List[int] = []

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        Add a single step of experience (transition) to the buffer.

        Handles buffer overflow by clearing the buffer before adding new data.

        Parameters
        ----------
        obs : np.ndarray
            Observation from the environment (raw). Should match observation_shape.
        action : Union[int, np.ndarray]
            Action taken by the agent.
        reward : float
            Reward received from the environment.
        done : bool
            Whether the episode terminated after this step.
        value : float
            Value estimate V(s) from the critic for the current state `obs`.
        log_prob : float
            Log probability log(pi(a|s)) of the action taken.
        """
        if self.ptr >= self.buffer_size:
            logger.warning("Rollout buffer overflow. Clearing buffer before adding new data.")
            self.clear()

        # ##: Store data in the dictionary.
        self.data["observations"][self.ptr] = obs
        self.data["actions"][self.ptr] = action
        self.data["rewards"][self.ptr] = reward
        self.data["dones"][self.ptr] = float(done)
        self.data["values"][self.ptr] = value
        self.data["log_probs"][self.ptr] = log_prob

        # ##: Track episode ends.
        if done:
            self.episode_ends.append(self.ptr)

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
            logger.debug("Rollout buffer is now full.")

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).

        Parameters
        ----------
        last_value : float
            Value estimate V(s') for the state immediately following the last state in the buffer.
        last_done : bool
            Indicates whether the episode terminated immediately after the last step in the buffer.
            If True, the `last_value` is effectively zero.
        """
        num_steps = self.buffer_size if self.full else self.ptr
        if num_steps < self.buffer_size:
            logger.warning(
                f"Computing GAE on a partially filled buffer ({num_steps}/{self.buffer_size} steps). "
                "Ensure this is intended and the logic handles it correctly."
            )
        elif not self.full and self.ptr == self.buffer_size:
            self.full = True

        # ##: Adjust last_value based on last_done for bootstrapping.
        last_value = last_value * (1.0 - float(last_done))
        last_gae_lam = 0.0

        # ##: Iterate backwards through the buffer steps.
        for t in reversed(range(num_steps)):
            # ##: Get value estimate for current and next step from the data dictionary.
            value_t = self.data["values"][t]

            # ##: Determine the value of the next state V(s_{t+1}).
            # ##: If t is the last step in the buffer, use the bootstrapped last_value.
            # ##: Otherwise, use the stored value for step t+1.
            value_t_plus_1 = self.data["values"][t + 1] if t < num_steps - 1 else last_value

            # ##: If step t led to a terminal state (done=True), the value of the next state is 0.
            # ##: We use 1.0 - dones[t] as a mask.
            value_t_plus_1 = value_t_plus_1 * (1.0 - self.data["dones"][t])

            # ##: Calculate the Temporal Difference (TD) error (delta).
            # ##: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.data["rewards"][t] + self.gamma * value_t_plus_1 - value_t

            # ##: Calculate the GAE advantage A_t using the recursive formula:
            # ##: A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            # ##: where A_{t+1} is represented by last_gae_lam in the loop.
            last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - self.data["dones"][t]) * last_gae_lam
            self.data["advantages"][t] = last_gae_lam

        # ##: Compute returns R_t by adding the computed advantages A_t to the value estimates V(s_t).
        # ##: R_t = A_t + V(s_t)
        self.data["returns"][:num_steps] = self.data["advantages"][:num_steps] + self.data["values"][:num_steps]

        # ##: Normalize advantages across the processed steps to reduce variance.
        adv_slice = self.data["advantages"][:num_steps]
        adv_mean = np.mean(adv_slice)
        adv_std = np.std(adv_slice)

        # ##: Add epsilon (1e-8) to std dev to prevent division by zero.
        if adv_std > 1e-8:
            self.data["advantages"][:num_steps] = (adv_slice - adv_mean) / (adv_std + 1e-8)
        else:
            self.data["advantages"][:num_steps] = adv_slice - adv_mean

    def get_batch(self) -> Dict[str, tf.Tensor]:
        """
        Retrieve the entire buffer's data as TensorFlow tensors.

        Preprocesses observations before converting them to a tensor. Assumes that `compute_returns_and_advantages`
        has already been called to populate advantages and returns. Warns if the buffer is not full or
        if advantages/returns appear uncomputed.

        Returns
        -------
        Dict[str, tf.Tensor]
            A dictionary containing TensorFlow tensors for the data stored in the buffer.
        """
        # ##: Determine the number of valid steps (handles partially filled buffer).
        num_steps = self.buffer_size if self.full else self.ptr

        if num_steps == 0:
            logger.error("Attempting to get batch from an empty buffer.")
            return {}

        if not self.full:
            logger.warning(f"Retrieving batch from a partially filled buffer ({num_steps}/{self.buffer_size} steps).")

        # ##: Check if advantages/returns seem computed (basic check).
        if np.all(self.data["advantages"][:num_steps] == 0) or np.all(self.data["returns"][:num_steps] == 0):
            logger.info(
                "Advantages/Returns are all zero. Ensure compute_returns_and_advantages was called "
                "and this is expected."
            )

        # ##: Preprocess observations before creating the tensor.
        # ##: Assumes observations up to num_steps are valid.
        processed_obs = tf.stack([preprocess_observation(obs) for obs in self.data["observations"][:num_steps]])

        # ##: Convert relevant numpy arrays from the data dictionary to TensorFlow tensors.
        batch = {
            "observations": processed_obs,
            "actions": tf.convert_to_tensor(self.data["actions"][:num_steps], dtype=tf.int32),
            "advantages": tf.convert_to_tensor(self.data["advantages"][:num_steps], dtype=tf.float32),
            "returns": tf.convert_to_tensor(self.data["returns"][:num_steps], dtype=tf.float32),
            "log_probs_old": tf.convert_to_tensor(self.data["log_probs"][:num_steps], dtype=tf.float32),
            "values_old": tf.convert_to_tensor(self.data["values"][:num_steps], dtype=tf.float32),
        }
        return batch

    def sample_mini_batches(self, batch_size: int, n_epochs: int) -> Generator[Dict[str, tf.Tensor], None, None]:
        """
        Generator that yields mini-batches sampled from the buffer for PPO updates.

        Retrieves the full batch data, shuffles the indices at the start of each epoch,
        and then yields mini-batches of the specified size.

        Parameters
        ----------
        batch_size : int
            The desired size of each mini-batch.
        n_epochs : int
            The number of times to iterate over the entire buffer data.

        Yields
        ------
        Generator[Dict[str, tf.Tensor], None, None]
            A generator where each item is a dictionary containing a mini-batch of data as TensorFlow tensors.
        """
        full_batch = self.get_batch()
        if not full_batch:
            logger.warning("Cannot sample mini-batches from an empty or invalid buffer.")
            return

        # ##: Get the actual size of the retrieved batch (handles partial fills).
        buffer_size_actual = tf.shape(full_batch["observations"])[0].numpy()
        if buffer_size_actual == 0:
            logger.warning("Buffer size is 0 after get_batch, cannot sample.")
            return

        indices = np.arange(buffer_size_actual)

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, buffer_size_actual, batch_size):
                end_idx = min(start_idx + batch_size, buffer_size_actual)
                minibatch_indices = indices[start_idx:end_idx]

                # ##: Create the minibatch by gathering data using the shuffled indices.
                minibatch = {key: tf.gather(value, minibatch_indices) for key, value in full_batch.items()}
                yield minibatch

    def clear(self) -> None:
        """
        Reset the buffer state.

        Sets the pointer (`ptr`) and `path_start_idx` to 0, marks the buffer as not full (`full=False`),
        clears the list of episode end indices (`episode_ends`), and resets computed values (advantages and returns)
        to zero. Does not clear the stored experience data itself, as it will be overwritten.
        """
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False
        self.episode_ends = []
        self.data["advantages"].fill(0)
        self.data["returns"].fill(0)

        logger.debug("Rollout buffer cleared.")

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
