# -*- coding: utf-8 -*-
"""
Prioritized experience replay buffer for reinforcement learning agents.
"""

from typing import Dict, Tuple, Union

from numpy import any as np_any
from numpy import float32
from numpy import max as np_max
from numpy import ndarray, ones
from numpy import sum as np_sum
from numpy.random import choice

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
        prioritization_params: Dict[str, float] = None,
    ):
        """
        Initialize the prioritized replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of experiences to store.
        observation_shape : Tuple[int, ...]
            Shape of observations.
        action_shape : Tuple[int, ...], optional
            Shape of actions, default is empty tuple.
        prioritization_params : Dict[str, float], optional
            Dictionary containing prioritization parameters:
            - alpha: Prioritization exponent (default: 0.6)
            - beta: Importance sampling exponent (default: 0.4)
            - epsilon: Small value added to priorities (default: 1e-6)
        """
        super().__init__(capacity, observation_shape, action_shape)

        # ##: Default prioritization parameters.
        default_params = {"alpha": 0.6, "beta": 0.4, "epsilon": 1e-6}
        if prioritization_params:
            default_params.update(prioritization_params)

        self.alpha = default_params["alpha"]
        self.beta = default_params["beta"]
        self.epsilon = default_params["epsilon"]

        # ##: Initialize priorities with ones.
        self._priorities = ones((capacity,), dtype=float32)

    def add(self, experience: Dict[str, Union[ndarray, int, float, bool]]) -> None:
        """
        Add an experience to the buffer with maximum priority.

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
        # ##: Set the priority of the new experience to the maximum priority
        # in the buffer, or 1 if the buffer is empty.
        max_priority = np_max(self._priorities) if self.size > 0 else 1.0
        self._priorities[self._next_idx] = max_priority

        # ##: Add the experience to the buffer.
        super().add(experience)

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
        probabilities = priorities / np_sum(priorities)

        # ##: Sample indices based on priorities.
        indices = choice(self.size, size=batch_size, p=probabilities)

        # ##: Calculate importance sampling weights.
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # ##: Retrieve the batch data using the base class helper method.
        batch_data = super()._get_batch(indices)

        # ##: Add prioritized buffer specific items.
        batch_data["indices"] = indices
        batch_data["weights"] = weights

        return batch_data

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

        if np_any(indices >= self.size):
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
