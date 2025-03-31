# -*- coding: utf-8 -*-
"""
Experience replay buffer for reinforcement learning agents.
"""

from typing import Dict, Tuple, Union

import numpy as np
from numpy import ndarray

# ##: TODO: Split this into a separate file.


class ReplayBuffer:
    """
    Simple experience replay buffer for reinforcement learning agents.

    This class implements a circular buffer to store and sample experiences
    for off-policy reinforcement learning algorithms.
    """

    def __init__(self, capacity: int, observation_shape: Tuple[int, ...], action_shape: Tuple[int, ...] = ()):
        """
        Initialize the replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of experiences to store.
        observation_shape : tuple of int
            Shape of observations.
        action_shape : tuple of int, optional
            Shape of actions, default is empty tuple.
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        # ##: Initialize buffer arrays.
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.int32 if not action_shape else np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

        self._buffer_full = False
        self._next_idx = 0

    def add(
        self, observation: ndarray, action: Union[int, ndarray], reward: float, next_observation: ndarray, done: bool
    ) -> None:
        """
        Add an experience to the buffer.

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
        self.observations[self._next_idx] = observation
        self.actions[self._next_idx] = action
        self.rewards[self._next_idx] = reward
        self.next_observations[self._next_idx] = next_observation
        self.dones[self._next_idx] = done

        self._next_idx = (self._next_idx + 1) % self.capacity
        if self._next_idx == 0:
            self._buffer_full = True

    def sample(self, batch_size: int) -> Dict[str, ndarray]:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing batches of observations, actions, rewards, next_observations, and dones.

        Raises
        ------
        ValueError
            If the buffer is empty or batch_size is larger than the number of experiences in the buffer.
        """
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} experiences from a buffer with {self.size} experiences")

        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def can_sample(self, batch_size: int) -> bool:
        """
        Check if the buffer can sample a batch of given size.

        Parameters
        ----------
        batch_size : int
            The batch size to check.

        Returns
        -------
        bool
            Whether the buffer can sample a batch of given size.
        """
        return self.size >= batch_size

    @property
    def size(self) -> int:
        """
        Get the current size of the buffer.

        Returns
        -------
        int
            Number of experiences in the buffer.
        """
        if self._buffer_full:
            return self.capacity
        return self._next_idx

    def clear(self) -> None:
        """Clear the buffer."""
        self._next_idx = 0
        self._buffer_full = False


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
