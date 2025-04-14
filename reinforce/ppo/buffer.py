# -*- coding: utf-8 -*-
"""Experience Buffer for PPO."""

from typing import Tuple

from numpy import append, float32, int32, mean, ndarray, std, zeros
from scipy import signal


def discounted_cumulative_sums(vector: ndarray, discount: float) -> ndarray:
    """
    Compute discounted cumulative sums of vectors.

    Parameters
    ----------
    vector : np.ndarray
        Input array.
    discount : float
        Discount factor.

    Returns
    -------
    np.ndarray
        Array of discounted cumulative sums.
    """
    return signal.lfilter([1], [1, float(-discount)], vector[::-1], axis=0)[::-1]


class Buffer:
    """
    Buffer for storing trajectories experienced by a PPO agent interacting with the environment.

    Uses Generalized Advantage Estimation (GAE) for calculating advantages.
    """

    def __init__(self, observation_shape: Tuple[int, int, int], size: int, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize the buffer.

        Parameters
        ----------
        observation_shape : Tuple[int, int, int]
            Shape of the observations.
        size : int
            Maximum size of the buffer (number of steps).
        gamma : float
            Discount factor for rewards.
        lam : float
            Lambda factor for GAE computation.
        """
        self.observation_buffer = zeros((size,) + observation_shape, dtype=float32)
        self.action_buffer = zeros(size, dtype=int32)
        self.advantage_buffer = zeros(size, dtype=float32)
        self.reward_buffer = zeros(size, dtype=float32)
        self.return_buffer = zeros(size, dtype=float32)
        self.value_buffer = zeros(size, dtype=float32)
        self.logprobability_buffer = zeros(size, dtype=float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index, self.max_size = 0, 0, size

    def store(self, observation, action, reward, value, logprobability):
        """
        Append one step of agent-environment interaction to the buffer.

        Args:
            observation: The observation received from the environment.
            action: The action taken by the agent.
            reward: The reward received from the environment.
            value: The value estimate for the observation from the critic.
            logprobability: The log-probability of the action taken.
        """
        if self.pointer >= self.max_size:
            print("Warning: Buffer overflow. Increase buffer size or decrease steps_per_epoch.")
            self.pointer = 0

        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value: int = 0):
        """
        Finalize the trajectory.

        Called at the end of an episode or when the buffer is full. Calculates
        advantage estimates (GAE-Lambda) and rewards-to-go.

        Parameters
        ----------
        last_value : int, optional
            Value estimate for the final observation in the trajectory.
            Used for bootstrapping if the trajectory didn't end terminally.
        """
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = append(self.reward_buffer[path_slice], last_value)
        values = append(self.value_buffer[path_slice], last_value)

        # ##: GAE-Lambda advantage calculation.
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lam)

        # ##: Rewards-to-go calculation.
        self.return_buffer[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        """
        Retrieve all data from the buffer.

        Also normalizes the advantages (standard score).
        Resets the buffer pointer and trajectory start index.

        Returns:
            tuple: Contains observation, action, advantage, return, and log-probability buffers.
        """
        if self.pointer != self.max_size:
            print(
                f"Warning: Buffer not full ({self.pointer}/{self.max_size}). Ensure steps_per_epoch matches buffer size."
            )

        if self.trajectory_start_index < self.pointer:
            print("Warning: finish_trajectory likely not called for the last path before get().")

        self.pointer, self.trajectory_start_index = 0, 0

        # ##: Normalize advantages.
        advantage_mean = mean(self.advantage_buffer)
        advantage_std = std(self.advantage_buffer)

        # ##: Add a small epsilon to prevent division by zero.
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / (advantage_std + 1e-8)

        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )
