"""Abstract base class for experience buffers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseBuffer(ABC):
    """
    Abstract base class for experience replay buffers.

    Different algorithms require different buffer implementations:
    - PPO: On-policy buffer with GAE computation
    - DQN: Off-policy replay buffer with priority sampling
    """

    def __init__(self, obs_shape: tuple[int, ...], capacity: int):
        """
        Initialize the buffer.

        Parameters
        ----------
        obs_shape : tuple[int, ...]
            Shape of a single observation.
        capacity : int
            Maximum number of transitions the buffer can hold.
        """
        self.obs_shape = obs_shape
        self.capacity = capacity

    @abstractmethod
    def store(self, *args: Any, **kwargs: Any) -> None:
        """
        Store a transition in the buffer.

        Parameters vary by buffer type (PPO stores value/log_prob, DQN stores SARS tuples).
        """

    @abstractmethod
    def sample(self, batch_size: int) -> tuple[Any, ...]:
        """
        Sample a batch of experiences.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        tuple
            Batch of experiences (format varies by buffer type).
        """

    @abstractmethod
    def clear(self) -> None:
        """Reset the buffer, removing all stored transitions."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the current number of stored transitions."""
