"""
PPO Agent implementation using TensorFlow.

Combines the policy/value networks, buffer, and PPO training logic.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf
from loguru import logger

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
        num_envs: int = 1,
        steps_per_update: int = 2048,
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
        num_envs : int, optional
            Number of parallel environments being used. Default is 1.
        steps_per_update : int, optional
            Number of steps collected per environment before an update. Default is 2048.
        """
        if observation_space.shape is None:
            raise ValueError("Observation space must have a defined shape.")
        self.obs_shape: tuple[int, ...] = observation_space.shape
        self.input_shape: tuple[int, ...] = observation_space.shape

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("PPOAgent currently only supports Discrete action spaces.")
        self.num_actions: int = int(action_space.n)

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

        # ##: Initialize buffer with size information.
        self.buffer = Buffer(
            obs_shape=self.obs_shape, num_envs=num_envs, steps_per_env=steps_per_update, gamma=self.gamma, lam=self.lam
        )

    def _preprocess_state(self, state: np.ndarray) -> tf.Tensor:
        """
        Preprocesses the environment state(s) for network input.

        Converts state(s) to a TensorFlow tensor. Assumes input might already have a batch dimension.

        Parameters
        ----------
        state : np.ndarray
            The environment observation.

        Returns
        -------
        tf.Tensor
            The processed state tensor ready for network input.
        """
        return tf.convert_to_tensor(state, dtype=tf.float32)

    def get_action(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects actions based on the current policy and a batch of states.

        Uses the policy network to sample actions from the distribution for the
        given states and the value network to estimate the states' values.

        Parameters
        ----------
        state : np.ndarray
            The current environment observation.

        Returns
        -------
        actions : np.ndarray
            The batch of actions selected by the policy.
        values : np.ndarray
            The batch of value estimates for the states from the critic.
        action_log_probs : np.ndarray
            The batch of log probabilities of the selected actions under the current policy.
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
        values = self.value_network(processed_state, training=False)

        # ##: Return numpy arrays for interaction with gym envs.
        return action.numpy(), tf.squeeze(values).numpy(), action_prob.numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: np.ndarray,
        action_log_prob: np.ndarray,
    ):
        """
        Stores a batch of transitions in the experience buffer.

        Parameters
        ----------
        state : np.ndarray
            Batch of states observed. Shape: (num_envs, *obs_shape)
        action : np.ndarray
            Batch of actions taken. Shape: (num_envs,)
        reward : np.ndarray
            Batch of rewards received. Shape: (num_envs,)
        value : np.ndarray
            Batch of value estimates of the states. Shape: (num_envs,)
        done : np.ndarray
            Batch of done flags. Shape: (num_envs,)
        action_log_prob : np.ndarray
            Batch of log probabilities of the actions taken. Shape: (num_envs,)
        """
        self.buffer.store(state, action, reward, value, done, action_log_prob)

    def learn(self, last_state: np.ndarray | None = None):
        """
        Performs the PPO learning update step using collected batch experiences.

        Computes advantages and returns for the collected trajectory, then updates the policy
        and value networks using mini-batch gradient descent over multiple epochs based on
        the PPO objective. Clears the buffer afterwards.

        Parameters
        ----------
        last_state : Optional[np.ndarray], optional
            The batch of final states observed after the last step of the trajectory
            from each parallel environment. Shape: (num_envs, *obs_shape) or None.
            Default is None.
        """
        # ##: Estimate the value of the last states for GAE calculation.
        last_values = np.zeros(self.buffer.num_envs)
        if last_state is not None:
            current_last_state: np.ndarray = last_state
            # ##: Ensure last_state has the correct shape (num_envs, *obs_shape).
            if len(current_last_state.shape) == len(self.obs_shape):
                current_last_state = np.expand_dims(current_last_state, 0)
            elif current_last_state.shape[0] != self.buffer.num_envs:
                logger.warning(
                    f"last_state batch size ({current_last_state.shape[0]}) doesn't match "
                    f"num_envs ({self.buffer.num_envs})."
                )
                processed_last_states = self._preprocess_state(current_last_state)
                last_values_tensor = self.value_network(processed_last_states, training=False)
                last_values.fill(tf.reduce_mean(last_values_tensor).numpy())
                current_last_state = None  # type: ignore[assignment]

            if current_last_state is not None:
                processed_last_states = self._preprocess_state(current_last_state)
                last_values_tensor = self.value_network(processed_last_states, training=False)
                last_values = tf.squeeze(last_values_tensor).numpy()

                if self.buffer.num_envs == 1 and last_values.ndim == 0:
                    last_values = np.expand_dims(last_values, 0)

        # ##: Compute advantages and returns using the batch of last values.
        self.buffer.compute_advantages_and_returns(last_values)

        # ##: Get batched data. Buffer needs modification.
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
        except (OSError, FileNotFoundError, tf.errors.OpError) as exc:
            logger.warning(f"Error loading models from {policy_path} and {value_path}: {exc}.")
