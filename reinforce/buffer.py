# -*- coding: utf-8 -*-
"""
Experience replay buffer for PPO.

Stores trajectories and calculates advantages and returns using Generalized Advantage Estimation (GAE).
"""

import numpy as np
import tensorflow as tf


class Buffer:
    """
    A buffer for storing trajectories and calculating GAE.

    This buffer collects agent experiences (states, actions, rewards, etc.) over a period and computes
    the Generalized Advantage Estimation (GAE) and returns, which are crucial for training policy
    gradient algorithms like PPO.

    Attributes
    ----------
    gamma : float
        Discount factor for future rewards.
    lam : float
        Lambda parameter for GAE calculation. Controls the bias-variance
        trade-off in advantage estimation.
    states : list[np.ndarray]
        List of states encountered during trajectory collection.
    actions : list[int]
        List of actions taken during trajectory collection.
    rewards : list[float]
        List of rewards received during trajectory collection.
    values : list[float]
        List of value estimates from the critic network for each state.
    dones : list[bool]
        List of boolean flags indicating episode termination after each step.
    action_probs : list[float]
        List of log probabilities of the actions taken.
    advantages : tf.Tensor | None
        Calculated Generalized Advantage Estimation (GAE) for each step.
        Computed after calling `compute_advantages_and_returns`. Initially None.
    returns : tf.Tensor | None
        Calculated returns (target values for the critic) for each step.
        Computed after calling `compute_advantages_and_returns`. Initially None.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize the buffer.

        Parameters
        ----------
        gamma : float, optional
            Discount factor. Default is 0.99.
        lam : float, optional
            GAE lambda parameter. Default is 0.95.
        """
        self.gamma = gamma
        self.lam = lam
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_probs = []
        self.advantages = None
        self.returns = None

    def store(self, state: np.ndarray, action: int, reward: float, value: float, done: bool, action_prob: float):
        """
        Store a single step transition in the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state observation.
        action : int
            The action taken.
        reward : float
            The reward received.
        value : float
            The value estimate for the state.
        done : bool
            Whether the episode terminated after this step.
        action_prob : float
            The log probability of the action taken.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_probs.append(action_prob)

    def compute_advantages_and_returns(self, last_value: float = 0.0):
        """
        Computes advantages and returns for the stored trajectory using GAE.

        Parameters
        ----------
        last_value : float, optional
            The value estimate of the final state in the trajectory. Default is 0.0.
        """
        # ##: Convert lists to numpy arrays for vectorized operations.
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0

        # ##: Calculate advantages working backwards from the end of the trajectory.
        for traj in reversed(range(len(rewards))):
            delta = rewards[traj] + self.gamma * values[traj + 1] * (1 - dones[traj]) - values[traj]
            gae = delta + self.gamma * self.lam * (1 - dones[traj]) * gae
            advantages[traj] = gae

        # ##: Calculate returns (target for value function).
        returns = advantages + values[:-1]

        self.advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        self.returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        self.advantages = (self.advantages - tf.reduce_mean(self.advantages)) / (
            tf.math.reduce_std(self.advantages) + 1e-8
        )

    def get_batches(self, batch_size: int) -> tf.data.Dataset:
        """
        Creates a TensorFlow Dataset for iterating over mini-batches.

        Parameters
        ----------
        batch_size : int
            The size of each mini-batch.

        Returns
        -------
        tf.data.Dataset
            A TensorFlow Dataset yielding batches of (states, actions, action_probs, returns, advantages).

        Raises
        ------
        ValueError
            If advantages and returns have not been computed yet.
        """
        if self.advantages is None or self.returns is None:
            raise ValueError("Advantages and returns must be computed before getting batches.")

        # ##: Convert remaining lists to tensors.
        states_tensor = tf.convert_to_tensor(np.array(self.states), dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(np.array(self.actions), dtype=tf.int32)
        action_probs_tensor = tf.convert_to_tensor(np.array(self.action_probs), dtype=tf.float32)

        # ##: Create dataset.
        dataset = tf.data.Dataset.from_tensor_slices(
            (states_tensor, actions_tensor, action_probs_tensor, self.returns, self.advantages)
        )

        # ##: Shuffle and batch the dataset.
        dataset = dataset.shuffle(buffer_size=len(self.states)).batch(batch_size)

        return dataset

    def clear(self):
        """Clears all stored trajectories from the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_probs.clear()
        self.advantages = None
        self.returns = None
