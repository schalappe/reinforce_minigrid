"""
PPO Agent implementation using TensorFlow.

Combines the policy/value networks, buffer, and PPO training logic with learning rate annealing
and proper gradient clipping.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import tensorflow as tf
from loguru import logger

from reinforce.core.base_agent import BaseAgent
from reinforce.core.schedules import LinearSchedule
from reinforce.ppo.buffer import Buffer
from reinforce.ppo.network import build_actor_critic_networks
from reinforce.ppo.ppo import get_action_distribution, train_step


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent.

    Implements PPO with:
    - Learning rate annealing
    - Global gradient clipping
    - Optional value function clipping
    - IMPALA-style CNN architecture
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
        max_grad_norm: float = 0.5,
        total_timesteps: int = 1_000_000,
        use_lr_annealing: bool = True,
        use_value_clipping: bool = False,
    ):
        """
        Initialize the PPO Agent.

        Parameters
        ----------
        observation_space : gym.Space
            The observation space of the environment.
        action_space : gym.Space
            The action space of the environment. Must be Discrete.
        learning_rate : float, optional
            Initial learning rate. Default is 3e-4.
        gamma : float, optional
            Discount factor. Default is 0.99.
        lam : float, optional
            GAE lambda parameter. Default is 0.95.
        clip_param : float, optional
            PPO clipping parameter. Default is 0.2.
        entropy_coef : float, optional
            Entropy bonus coefficient. Default is 0.01.
        vf_coef : float, optional
            Value function loss coefficient. Default is 0.5.
        epochs : int, optional
            PPO epochs per update. Default is 4.
        batch_size : int, optional
            Mini-batch size. Default is 64.
        num_envs : int, optional
            Number of parallel environments. Default is 1.
        steps_per_update : int, optional
            Steps per environment per update. Default is 2048.
        max_grad_norm : float, optional
            Maximum gradient norm. Default is 0.5.
        total_timesteps : int, optional
            Total training timesteps for LR scheduling. Default is 1_000_000.
        use_lr_annealing : bool, optional
            Whether to use learning rate annealing. Default is True.
        use_value_clipping : bool, optional
            Whether to use value function clipping. Default is False.
        """
        super().__init__(observation_space, action_space)

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("PPOAgent currently only supports Discrete action spaces.")
        self.num_actions: int = int(action_space.n)
        self.input_shape: tuple[int, ...] = self.obs_shape

        # ##>: Store hyperparameters.
        self.gamma = gamma
        self.lam = lam
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.use_value_clipping = use_value_clipping
        self.use_lr_annealing = use_lr_annealing
        self.initial_learning_rate = learning_rate

        # ##>: Build networks with IMPALA architecture and orthogonal init.
        self.policy_network, self.value_network = build_actor_critic_networks(self.input_shape, self.num_actions)

        # ##>: Setup optimizers with epsilon=1e-5 (per "37 Implementation Details").
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)

        # ##>: Learning rate scheduler.
        self.lr_scheduler = LinearSchedule(learning_rate, 0.0, total_timesteps) if use_lr_annealing else None
        self.current_timestep = 0

        # ##>: Initialize buffer.
        self.buffer = Buffer(
            obs_shape=self.obs_shape,
            num_envs=num_envs,
            steps_per_env=steps_per_update,
            gamma=self.gamma,
            lam=self.lam,
        )

    def _preprocess_state(self, state: np.ndarray) -> tf.Tensor:
        """Preprocess state for network input."""
        return tf.convert_to_tensor(state, dtype=tf.float32)

    def _update_learning_rate(self) -> None:
        """Update optimizer learning rates based on scheduler."""
        if self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.value(self.current_timestep)
            self.policy_optimizer.learning_rate.assign(new_lr)
            self.value_optimizer.learning_rate.assign(new_lr)

    def get_action(self, state: np.ndarray, training: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Select actions based on the current policy.

        Parameters
        ----------
        state : np.ndarray
            Current observation(s).
        training : bool, optional
            Whether in training mode. Default is True.

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            Tuple of (actions, info_dict) where info_dict contains 'values' and 'log_probs'.
        """
        processed_state = self._preprocess_state(state)

        action_logits = self.policy_network(processed_state, training=False)
        dist = get_action_distribution(action_logits)
        action = dist.sample()
        action_prob = dist.log_prob(action)
        values = self.value_network(processed_state, training=False)

        return action.numpy(), {"values": tf.squeeze(values).numpy(), "log_probs": action_prob.numpy()}

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: np.ndarray,
        action_log_prob: np.ndarray,
    ) -> None:
        """
        Store a batch of transitions in the experience buffer.

        Parameters
        ----------
        state : np.ndarray
            Batch of states. Shape: (num_envs, *obs_shape)
        action : np.ndarray
            Batch of actions. Shape: (num_envs,)
        reward : np.ndarray
            Batch of rewards. Shape: (num_envs,)
        value : np.ndarray
            Batch of value estimates. Shape: (num_envs,)
        done : np.ndarray
            Batch of done flags. Shape: (num_envs,)
        action_log_prob : np.ndarray
            Batch of action log probabilities. Shape: (num_envs,)
        """
        self.buffer.store(state, action, reward, value, done, action_log_prob)

    def learn(self, last_state: np.ndarray | None = None, steps_collected: int = 0, **kwargs: Any) -> dict[str, float]:
        """
        Perform the PPO learning update.

        Parameters
        ----------
        last_state : np.ndarray, optional
            Final states for bootstrapping. Shape: (num_envs, *obs_shape).
        steps_collected : int, optional
            Number of steps collected for LR scheduling.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        self.current_timestep += steps_collected
        self._update_learning_rate()

        # ##>: Estimate value of last states for GAE.
        last_values = np.zeros(self.buffer.num_envs)
        if last_state is not None:
            current_last_state: np.ndarray = last_state
            if len(current_last_state.shape) == len(self.obs_shape):
                current_last_state = np.expand_dims(current_last_state, 0)
            elif current_last_state.shape[0] != self.buffer.num_envs:
                logger.warning(
                    f"last_state batch size ({current_last_state.shape[0]}) does not match "
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

        self.buffer.compute_advantages_and_returns(last_values)
        dataset = self.buffer.get_batches(self.batch_size)

        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "approx_kl": [],
        }

        for _ in range(self.epochs):
            for batch in dataset:
                states, actions, old_action_probs, returns, advantages = batch
                pi_loss, v_loss, ent_bonus, clip_frac, approx_kl = train_step(
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
                    self.max_grad_norm,
                    old_values=None,
                    use_value_clipping=self.use_value_clipping,
                )

                metrics["policy_loss"].append(float(pi_loss.numpy()))
                metrics["value_loss"].append(float(v_loss.numpy()))
                metrics["entropy"].append(float(ent_bonus.numpy()))
                metrics["clip_fraction"].append(float(clip_frac.numpy()))
                metrics["approx_kl"].append(float(approx_kl.numpy()))

        self.buffer.clear()
        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def save_models(self, path_prefix: str) -> None:
        """Save policy and value networks."""
        policy_path = f"{path_prefix}_policy.keras"
        value_path = f"{path_prefix}_value.keras"
        self.policy_network.save(policy_path)
        self.value_network.save(value_path)
        logger.info(f"Models saved to {policy_path} and {value_path}")

    def load_models(self, path_prefix: str) -> None:
        """Load policy and value networks."""
        policy_path = Path(f"{path_prefix}_policy.keras")
        value_path = Path(f"{path_prefix}_value.keras")
        try:
            self.policy_network = tf.keras.models.load_model(policy_path)
            self.value_network = tf.keras.models.load_model(value_path)
            logger.info(f"Models loaded from {policy_path} and {value_path}")
        except (OSError, FileNotFoundError, tf.errors.OpError) as exc:
            logger.warning(f"Error loading models from {policy_path} and {value_path}: {exc}.")

    def get_current_lr(self) -> float:
        """Return current learning rate."""
        return float(self.policy_optimizer.learning_rate.numpy())
