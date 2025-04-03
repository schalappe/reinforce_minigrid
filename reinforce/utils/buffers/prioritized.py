# -*- coding: utf-8 -*-
"""
Prioritized experience replay buffer for reinforcement learning agents.
"""

from typing import Dict, Tuple, Union

import numpy as np
from numpy import ndarray

from reinforce.utils.buffers.base import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer for reinforcement learning agents.

    This class extends the basic replay buffer with prioritized sampling based on TD errors,
    as described in the paper "Prioritized Experience Replay" by Schaul et al.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...] = (),
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ):
        """
        Initialize the prioritized replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of experiences to store.
        observation_shape : tuple of int
            Shape of observations.
        action_shape : tuple of int, optional
            Shape of actions, default is empty tuple.
        alpha : float, optional
            Prioritization exponent (0 = uniform sampling), default is 0.6.
        beta : float, optional
            Importance sampling exponent (1 = no correction), default is 0.4.
        epsilon : float, optional
            Small value to add to priorities to ensure non-zero sampling probability, default is 1e-6.
        """
        super().__init__(capacity, observation_shape, action_shape)

        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        # ##: Initialize priorities with ones.
        self._priorities = np.ones((capacity,), dtype=np.float32)

    def add(
        self, observation: ndarray, action: Union[int, ndarray], reward: float, next_observation: ndarray, done: bool
    ) -> None:
        """
        Add an experience to the buffer with maximum priority.

        Parameters
        ----------
        observation : ndarray
            Current observation.
        action : int or ndarray
            Action taken.
        reward : float
            Reward received.
        next_observation : ndarray
            Next observation.
        done : bool
            Whether the episode is done.
        """
        # ##: Set the priority of the new experience to the maximum priority
        # in the buffer, or 1 if the buffer is empty.
        max_priority = np.max(self._priorities) if self.size > 0 else 1.0
        self._priorities[self._next_idx] = max_priority

        # ##: Add the experience to the buffer.
        super().add(observation, action, reward, next_observation, done)

    def sample(self, batch_size: int) -> Dict[str, ndarray]:
        """
        Sample a batch of experiences from the buffer based on priorities.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing batches of observations, actions, rewards, next_observations,
            dones, indices, and weights.

        Raises
        ------
        ValueError
            If the buffer is empty or batch_size is larger than the number of experiences in the buffer.
        """
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} experiences from a buffer with {self.size} experiences")

        # ##: Calculate sampling probabilities.
        priorities = self._priorities[: self.size] ** self.alpha
        probabilities = priorities / np.sum(priorities)

        # ##: Sample indices based on priorities.
        indices = np.random.choice(self.size, size=batch_size, p=probabilities)

        # ##: Calculate importance sampling weights.
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": weights,
        }

    def update_priorities(self, indices: ndarray, priorities: ndarray) -> None:
        """
        Update the priorities of experiences.

        Parameters
        ----------
        indices : ndarray
            Indices of experiences to update.
        priorities : ndarray
            New priorities for the experiences.

        Raises
        ------
        ValueError
            If indices or priorities are invalid.
        """
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have the same length")

        if np.any(indices >= self.size):
            raise ValueError(f"indices must be less than buffer size ({self.size})")

        # ##: Add epsilon to ensure non-zero sampling probability.
        self._priorities[indices] = priorities + self.epsilon

    def update_beta(self, beta: float) -> None:
        """
        Update the importance sampling exponent.

        Parameters
        ----------
        beta : float
            New importance sampling exponent.
        """
        self.beta = beta
