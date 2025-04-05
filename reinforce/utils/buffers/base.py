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
        observation_shape : Tuple[int, ...]
            Shape of observations.
        action_shape : Tuple[int, ...], optional
            Shape of actions, default is empty tuple.
        """
        self.capacity = capacity

        # ##: Store shapes for potential use later (e.g., validation).
        self._observation_shape = observation_shape
        self._action_shape = action_shape

        # ##: Initialize buffer arrays within a dictionary.
        self.data = {
            "observations": np.zeros((capacity, *observation_shape), dtype=np.float32),
            "actions": np.zeros((capacity, *action_shape), dtype=np.int32 if not action_shape else np.float32),
            "rewards": np.zeros((capacity,), dtype=np.float32),
            "next_observations": np.zeros((capacity, *observation_shape), dtype=np.float32),
            "dones": np.zeros((capacity,), dtype=np.bool_),
        }

        self._buffer_full = False
        self._next_idx = 0

    def add(self, experience: Dict[str, Union[ndarray, int, float, bool]]) -> None:
        """
        Add an experience to the buffer.

        Parameters
        ----------
        experience : Dict[str, Union[ndarray, int, float, bool]]
            Dictionary containing the experience tuple:
            - observation: Current observation (ndarray)
            - action: Action taken (int or ndarray)
            - reward: Reward received (float)
            - next_observation: Next observation (ndarray)
            - done: Whether the episode is done (bool)
        """
        # ##: Store experience components in their respective arrays.
        idx = self._next_idx
        self.data["observations"][idx] = experience["observation"]
        self.data["actions"][idx] = experience["action"]
        self.data["rewards"][idx] = experience["reward"]
        self.data["next_observations"][idx] = experience["next_observation"]
        self.data["dones"][idx] = experience["done"]

        self._next_idx = (idx + 1) % self.capacity
        if self._next_idx == 0 and not self._buffer_full:
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
        if not self.can_sample(batch_size):
            raise ValueError(f"Cannot sample {batch_size} experiences from a buffer with {self.size} experiences")

        indices = np.random.randint(0, self.size, size=batch_size)
        return self._get_batch(indices)

    def _get_batch(self, indices: ndarray) -> Dict[str, ndarray]:
        """
        Retrieve a batch of data corresponding to the given indices.

        Parameters
        ----------
        indices : ndarray
            Indices of the experiences to retrieve.

        Returns
        -------
        Dict[str, ndarray]
            Dictionary containing batches for each data type.
        """
        return {key: buffer[indices] for key, buffer in self.data.items()}

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
