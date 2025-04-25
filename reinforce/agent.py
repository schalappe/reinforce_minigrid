# -*- coding: utf-8 -*-
"""
PPO Agent implementation using TensorFlow.

Combines the policy/value networks, buffer, and PPO training logic.
"""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import tensorflow as tf
from loguru import logger
from pathlib import Path

from . import setup_logger
from .buffer import Buffer
from .network import build_actor_critic_networks
from .ppo import get_action_distribution, train_step

setup_logger()


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.

    This agent implements the PPO algorithm, managing the policy and value networks, an experience buffer,
    and the training process.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_param: float = 0.2,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        epochs: int = 4,
        batch_size: int = 64,
    ):
        """Initializes the PPO Agent.

        Sets up networks, optimizers, and hyperparameters.

        Parameters
        ----------
        observation_space : gym.Space
            The observation space of the environment.
        action_space : gym.Space
            The action space of the environment. Must be Discrete.
        learning_rate : float, optional
            Learning rate for the policy and value network optimizers.
            Default is 3e-4.
        gamma : float, optional
            Discount factor for reward calculation. Default is 0.99.
        lam : float, optional
            Lambda parameter for Generalized Advantage Estimation (GAE).
            Default is 0.95.
        clip_param : float, optional
            Clipping parameter (epsilon) for the PPO objective function.
            Default is 0.2.
        entropy_coef : float, optional
            Coefficient for the entropy bonus term in the loss, encouraging
            exploration. Default is 0.01.
        vf_coef : float, optional
            Coefficient for the value function loss term in the total loss.
            Default is 0.5.
        epochs : int, optional
            Number of training epochs to run on the collected batch of
            experiences. Default is 4.
        batch_size : int, optional
            Size of the mini-batches used during training epochs. Default is 64.
        """
        self.input_shape = observation_space.shape

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("PPOAgent currently only supports Discrete action spaces.")
        self.num_actions = action_space.n

        # ##: Store hyperparameters.
        self.gamma = gamma
        self.lam = lam
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.epochs = epochs
        self.batch_size = batch_size

        # ##: Build networks.
        self.policy_network, self.value_network = build_actor_critic_networks(self.input_shape, self.num_actions)

        # ##: Setup optimizers.
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # ##: Initialize buffer.
        self.buffer = Buffer(gamma=self.gamma, lam=self.lam)

    def _preprocess_state(self, state: np.ndarray) -> tf.Tensor:
        """
        Preprocesses the environment state for network input.

        Ensures the state has a batch dimension and converts it to a TensorFlow tensor.

        Parameters
        ----------
        state : np.ndarray
            The environment observation.

        Returns
        -------
        tf.Tensor
            The processed state tensor ready for network input.
        """
        if len(state.shape) == len(self.input_shape):
            state = np.expand_dims(state, 0)
        return tf.convert_to_tensor(state, dtype=tf.float32)

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Selects an action based on the current policy and state.

        Uses the policy network to sample an action from the distribution for the
        given state and the value network to estimate the state's value.

        Parameters
        ----------
        state : np.ndarray
            The current environment observation.

        Returns
        -------
        action : int
            The action selected by the policy.
        value : float
            The value estimate for the state from the critic.
        action_log_prob : float
            The log probability of the selected action under the current policy.
        """
        processed_state = self._preprocess_state(state)

        # ##: Get action distribution logits from policy network.
        action_logits = self.policy_network(processed_state, training=False)
        dist = get_action_distribution(action_logits)

        # ##: Sample an action from the distribution.
        action = dist.sample()

        # ##: Get log probability of the sampled action.
        action_prob = dist.log_prob(action)

        # ##: Get value estimate from value network.
        value = self.value_network(processed_state, training=False)

        return action.numpy()[0], value.numpy()[0, 0], action_prob.numpy()[0]

    def store_transition(
        self, state: np.ndarray, action: int, reward: float, value: float, done: bool, action_log_prob: float
    ):
        """
        Stores a single transition in the experience buffer.

        Parameters
        ----------
        state : np.ndarray
            The state observed.
        action : int
            The action taken.
        reward : float
            The reward received.
        value : float
            The value estimate of the state.
        done : bool
            Whether the episode terminated after this transition.
        action_log_prob : float
            The log probability of the action taken.
        """
        self.buffer.store(state, action, reward, value, done, action_log_prob)

    def learn(self, last_state: Optional[np.ndarray] = None):
        """
        Performs the PPO learning update step.

        Computes advantages and returns for the collected trajectory, then updates the policy
        and value networks using mini-batch gradient descent over multiple epochs based on
        the PPO objective. Clears the buffer afterwards.

        Parameters
        ----------
        last_state : Optional[np.ndarray], optional
            The final state observed after the last step of the trajector.
            Default is None.
        """
        # ##: Estimate the value of the last state for GAE calculation.
        last_value = 0.0
        if last_state is not None:
            processed_last_state = self._preprocess_state(last_state)
            last_value = self.value_network(processed_last_state, training=False).numpy()[0, 0]

        # ##: Compute advantages and returns for the collected trajectory.
        self.buffer.compute_advantages_and_returns(last_value)

        # ##: Get batched data.
        dataset = self.buffer.get_batches(self.batch_size)

        # ##: Perform optimization over multiple epochs.
        for _ in range(self.epochs):
            for batch in dataset:
                states, actions, old_action_probs, returns, advantages = batch
                train_step(
                    states,
                    actions,
                    old_action_probs,
                    returns,
                    advantages,
                    self.policy_network,
                    self.value_network,
                    self.policy_optimizer,
                    self.value_optimizer,
                    self.clip_param,
                    self.vf_coef,
                    self.entropy_coef,
                )

        # ##: Clear the buffer for the next trajectory collection phase.
        self.buffer.clear()

    def save_models(self, path_prefix: str):
        """
        Saves the policy and value networks to files using the .keras format.

        Parameters
        ----------
        path_prefix : str
            The prefix for the filenames. Models will be saved to
            `{path_prefix}_policy.keras` and `{path_prefix}_value.keras`.
        """
        policy_path = f"{path_prefix}_policy.keras"
        value_path = f"{path_prefix}_value.keras"
        self.policy_network.save(policy_path)
        self.value_network.save(value_path)
        logger.info(f"Models saved to {policy_path} and {value_path}")

    def load_models(self, path_prefix: str):
        """
        Loads the policy and value networks from .keras files.

        Parameters
        ----------
        path_prefix : str
            The prefix for the filenames. Models will be loaded from
            `{path_prefix}_policy.keras` and `{path_prefix}_value.keras`.

        Raises
        ------
        Exception
            Catches and prints exceptions during file loading (e.g., file not found).
        """
        policy_path = Path(f"{path_prefix}_policy.keras")
        value_path = Path(f"{path_prefix}_value.keras")
        try:
            self.policy_network = tf.keras.models.load_model(policy_path)
            self.value_network = tf.keras.models.load_model(value_path)
            logger.info(f"Models loaded from {policy_path} and {value_path}")
        except Exception as e:
            logger.warning(f"Error loading models from {policy_path} and {value_path}: {e}.")
