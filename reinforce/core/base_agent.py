"""Abstract base class for reinforcement learning agents."""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class defining the interface for all RL agents.

    All concrete agent implementations (PPO, DQN, etc.) must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Initialize the base agent.

        Parameters
        ----------
        observation_space : gym.Space
            The observation space of the environment.
        action_space : gym.Space
            The action space of the environment.
        """
        self.observation_space = observation_space
        self.action_space = action_space

        if observation_space.shape is None:
            raise ValueError("Observation space must have a defined shape.")
        self.obs_shape: tuple[int, ...] = observation_space.shape

    @abstractmethod
    def get_action(self, state: np.ndarray, training: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Select action(s) given state(s).

        Parameters
        ----------
        state : np.ndarray
            Current observation(s) from the environment.
        training : bool, optional
            Whether the agent is in training mode. Default is True.

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            Tuple of (actions, info_dict) where info_dict contains algorithm-specific data
            such as values, log probabilities, Q-values, etc.
        """

    @abstractmethod
    def store_transition(self, *args: Any, **kwargs: Any) -> None:
        """
        Store experience in the agent's buffer.

        Parameters vary by algorithm (e.g., PPO stores values/log_probs, DQN stores transitions).
        """

    @abstractmethod
    def learn(self, **kwargs: Any) -> dict[str, float]:
        """
        Perform a learning update.

        Returns
        -------
        dict[str, float]
            Dictionary of training metrics (loss values, entropy, etc.).
        """

    @abstractmethod
    def save_models(self, path_prefix: str) -> None:
        """
        Save model weights to disk.

        Parameters
        ----------
        path_prefix : str
            Path prefix for model files (e.g., 'models/agent' -> 'models/agent_policy.keras').
        """

    @abstractmethod
    def load_models(self, path_prefix: str) -> None:
        """
        Load model weights from disk.

        Parameters
        ----------
        path_prefix : str
            Path prefix for model files.
        """

    def on_episode_end(self) -> None:  # noqa: B027
        """
        Handle episode end events.

        Override this method if your agent needs to perform cleanup
        or flush buffers when an episode ends.
        """
