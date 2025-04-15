# -*- coding: utf-8 -*-
"""
Experience Buffer for Proximal Policy Optimization (PPO).

This module defines the Buffer class used to store trajectories experienced by a PPO agent interacting
with an environment. It utilizes Generalized Advantage Estimation (GAE) for calculating advantages.
"""

from typing import Tuple

import numpy as np
from scipy import signal


def discounted_cumulative_sums(vector: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute discounted cumulative sums of vectors.

    Parameters
    ----------
    vector : np.ndarray
        The vector for which to compute discounted cumulative sums.
        For example, rewards or GAE deltas.
    discount : float
        The discount factor (e.g., gamma or gamma * lambda).

    Returns
    -------
    np.ndarray
        An array containing the discounted cumulative sums of the input vector.
        output[i] = sum_{j=i}^{N-1} (discount**(j-i) * vector[j])

    Examples
    --------
    >>> rewards = np.array([1, 1, 1, 1])
    >>> gamma = 0.9
    >>> discounted_cumulative_sums(rewards, gamma)
    array([3.439, 2.71 , 1.9  , 1.   ]) # 1 + 0.9*1 + 0.9^2*1 + 0.9^3*1, etc.
    """
    return signal.lfilter([1], [1, float(-discount)], vector[::-1], axis=0)[::-1]


class Buffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment. Uses Generalized Advantage Estimation (GAE) for
    calculating the advantages of state-action pairs.

    Attributes
    ----------
    observation_buffer : np.ndarray
        Stores observations from the environment.
    action_buffer : np.ndarray
        Stores actions taken by the agent.
    advantage_buffer : np.ndarray
        Stores the calculated advantages for each step.
    reward_buffer : np.ndarray
        Stores rewards received from the environment.
    return_buffer : np.ndarray
        Stores the rewards-to-go (discounted cumulative rewards).
    value_buffer : np.ndarray
        Stores value estimates from the critic network.
    logprobability_buffer : np.ndarray
        Stores the log probabilities of the actions taken.
    gamma : float
        Discount factor for future rewards.
    lam : float
        Lambda factor for GAE calculation.
    pointer : int
        Current position in the buffer.
    trajectory_start_index : int
        Index marking the start of the current trajectory.
    max_size : int
        Maximum capacity of the buffer.
    """

    def __init__(self, observation_shape: Tuple[int, ...], size: int, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize the PPO buffer.

        Parameters
        ----------
        observation_shape : Tuple[int, ...]
            The shape of the observations from the environment.
        size : int
            The maximum number of steps (transitions) the buffer can hold.
        gamma : float, optional
            The discount factor for reward calculation (default is 0.99).
        lam : float, optional
            The lambda factor for Generalized Advantage Estimation (GAE) (default is 0.95).
        """
        self.observation_buffer = np.zeros((size,) + observation_shape, dtype=np.float32)
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index, self.max_size = 0, 0, size

    def store(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        value: float,
        logprobability: float,
    ):
        """Append one step of agent-environment interaction to the buffer.

        If the buffer is full, it overwrites the oldest data.

        Parameters
        ----------
        observation : np.ndarray
            The observation received from the environment.
        action : int
            The action taken by the agent.
        reward : float
            The reward received from the environment.
        value : float
            The value estimate for the observation from the critic network.
        logprobability : float
            The log-probability of the action taken under the policy.
        """
        if self.pointer >= self.max_size:
            print("Warning: Buffer overflow. Overwriting oldest data.")
            self.pointer = 0

        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value: float = 0.0):
        """
        Finalize the current trajectory and compute advantages and returns.

        This method should be called at the end of an episode or when the buffer reaches its maximum size.
        It calculates advantage estimates using GAE-Lambda and computes the rewards-to-go for each step
        in the trajectory.

        Parameters
        ----------
        last_value : float, optional
            The value estimate for the final state in the trajectory. Defaults to 0.0.
        """
        path_slice = slice(self.trajectory_start_index, self.pointer)

        # ##: Ensure np.append is used.
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        # ##: GAE-Lambda advantage calculation.
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lam)

        # ##: Rewards-to-go calculation (often called returns).
        self.return_buffer[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1]

        # ##: Mark the start of the next trajectory.
        self.trajectory_start_index = self.pointer

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve all data stored in the buffer and prepare it for training.

        This method retrieves the collected observations, actions, advantages, returns (rewards-to-go),
        and log probabilities. It also normalizes the advantages (subtract mean, divide by standard deviation) to
        stabilize training. After calling `get()`, the buffer is reset.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - observation_buffer: Observations collected.
            - action_buffer: Actions taken.
            - advantage_buffer: Normalized advantages.
            - return_buffer: Rewards-to-go (returns).
            - logprobability_buffer: Log probabilities of the actions taken.

        Raises
        ------
        RuntimeWarning
            If the buffer is not full when `get()` is called, or if `finish_trajectory` was likely
            not called for the last path.
        """
        buffer_full = self.pointer == self.max_size
        last_trajectory_finished = self.trajectory_start_index == self.pointer or self.trajectory_start_index == 0

        if not buffer_full:
            print(
                f"Warning: Buffer not full ({self.pointer}/{self.max_size}). "
                "Data retrieved might be incomplete or from partial trajectories."
            )
        if not last_trajectory_finished and buffer_full:
            print(
                "Warning: Buffer is full, but finish_trajectory() was likely not "
                "called for the last path. Final trajectory data might be missing "
                "advantages and returns."
            )

        # ##: Normalize advantages to zero mean and unit variance.
        adv_mean = np.mean(self.advantage_buffer[: self.pointer])
        adv_std = np.std(self.advantage_buffer[: self.pointer])
        self.advantage_buffer[: self.pointer] = (self.advantage_buffer[: self.pointer] - adv_mean) / (adv_std + 1e-8)

        # ##: Retrieve data up to the current pointer.
        obs_buf = self.observation_buffer[: self.pointer]
        act_buf = self.action_buffer[: self.pointer]
        adv_buf = self.advantage_buffer[: self.pointer]
        ret_buf = self.return_buffer[: self.pointer]
        logp_buf = self.logprobability_buffer[: self.pointer]

        # ##: Reset buffer pointers.
        self.pointer, self.trajectory_start_index = 0, 0

        return obs_buf, act_buf, adv_buf, ret_buf, logp_buf
