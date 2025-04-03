# -*- coding: utf-8 -*-
"""
Base experience replay buffer for reinforcement learning agents.
"""

from typing import Dict, Tuple, Union

import numpy as np
from numpy import ndarray


class ReplayBuffer:
    """
    Simple experience replay buffer for reinforcement learning agents.

    This class implements a circular buffer to store and sample experiences for off-policy
    reinforcement learning algorithms.
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
