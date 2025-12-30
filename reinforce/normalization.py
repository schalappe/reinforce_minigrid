"""
Observation and reward normalization utilities for reinforcement learning.

Implements running statistics normalization as recommended in the "37 Implementation Details of PPO"
paper for improved training stability.
"""

import numpy as np


class RunningMeanStd:
    """
    Maintains running mean and standard deviation using Welford's algorithm.

    This is a numerically stable method for computing running statistics without storing all samples.

    Attributes
    ----------
    mean : np.ndarray
        Running mean of observed values.
    var : np.ndarray
        Running variance of observed values.
    count : float
        Number of samples observed.
    """

    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        """
        Initialize running statistics tracker.

        Parameters
        ----------
        shape : tuple, optional
            Shape of the values to track. Default is scalar ().
        epsilon : float, optional
            Small constant for numerical stability. Default is 1e-4.
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        """
        Update running statistics with a new batch of values.

        Uses Welford's parallel algorithm for combining statistics from two sets of samples.
        This is numerically stable and allows efficient batch updates.

        Parameters
        ----------
        batch : np.ndarray
            Batch of values to incorporate. First dimension is batch size.
        """
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """
        Update running statistics using batch moments.

        Implements parallel Welford algorithm:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        Parameters
        ----------
        batch_mean : np.ndarray
            Mean of the new batch.
        batch_var : np.ndarray
            Variance of the new batch.
        batch_count : int
            Number of samples in the new batch.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        # ##>: Parallel variance combination formula (Welford's algorithm).
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """
        Normalize values using running statistics.

        Parameters
        ----------
        x : np.ndarray
            Values to normalize.
        clip : float, optional
            Clip normalized values to [-clip, clip]. Default is 10.0.

        Returns
        -------
        np.ndarray
            Normalized and clipped values.
        """
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip)


class RewardNormalizer:
    """
    Normalizes rewards using a running estimate of reward standard deviation.

    Instead of normalizing by mean (which would shift the reward signal), this only scales by
    standard deviation to maintain reward sign semantics.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        """
        Initialize reward normalizer.

        Parameters
        ----------
        gamma : float, optional
            Discount factor for computing returns. Default is 0.99.
        epsilon : float, optional
            Small constant for numerical stability. Default is 1e-8.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(shape=())
        self.returns = None

    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Normalize rewards by running standard deviation of returns.

        Per the "37 Implementation Details of PPO", rewards are scaled by the standard deviation
        of the discounted return sum, not normalized.

        Parameters
        ----------
        rewards : np.ndarray
            Raw rewards from the environment. Shape: (num_envs,)
        dones : np.ndarray
            Done flags for each environment. Shape: (num_envs,)

        Returns
        -------
        np.ndarray
            Normalized rewards.
        """
        if self.returns is None:
            self.returns = np.zeros_like(rewards, dtype=np.float64)

        # ##>: Update discounted returns estimate.
        self.returns = self.returns * self.gamma * (1 - dones) + rewards
        self.return_rms.update(self.returns)

        # ##>: Scale rewards by return std (not mean, to preserve sign).
        return rewards / np.sqrt(self.return_rms.var + self.epsilon)

    def reset(self, env_idx: int | None = None) -> None:
        """
        Reset return tracking for specific environment(s).

        Parameters
        ----------
        env_idx : int, optional
            Index of environment to reset. If None, resets all.
        """
        if self.returns is not None:
            if env_idx is not None:
                self.returns[env_idx] = 0.0
            else:
                self.returns.fill(0.0)
