"""
Rainbow DQN Agent Implementation.

Combines all Rainbow enhancements:
- Double DQN (target network)
- Dueling Architecture
- Prioritized Experience Replay
- Multi-step Learning
- Noisy Networks
- Categorical DQN (C51)
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import tensorflow as tf
from loguru import logger

from reinforce.core.base_agent import BaseAgent
from reinforce.dqn.buffer import PrioritizedReplayBuffer
from reinforce.dqn.losses import categorical_dqn_loss, compute_target_distribution
from reinforce.dqn.network import build_rainbow_network


class RainbowAgent(BaseAgent):
    """
    Rainbow DQN Agent combining six algorithmic improvements.

    Parameters
    ----------
    observation_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space (must be Discrete).
    learning_rate : float, optional
        Optimizer learning rate. Default is 6.25e-5.
    gamma : float, optional
        Discount factor. Default is 0.99.
    n_step : int, optional
        Steps for multi-step returns. Default is 3.
    num_atoms : int, optional
        Atoms for distributional RL. Default is 51.
    v_min : float, optional
        Minimum value support. Default is -10.0.
    v_max : float, optional
        Maximum value support. Default is 10.0.
    buffer_size : int, optional
        Replay buffer capacity. Default is 100_000.
    batch_size : int, optional
        Training batch size. Default is 32.
    target_update_freq : int, optional
        Steps between target network updates. Default is 8_000.
    learning_starts : int, optional
        Steps before training begins. Default is 20_000.
    train_freq : int, optional
        Steps between training updates. Default is 4.
    priority_alpha : float, optional
        PER alpha parameter. Default is 0.6.
    priority_beta_start : float, optional
        Initial PER beta. Default is 0.4.
    priority_beta_frames : int, optional
        Frames for beta annealing. Default is 100_000.
    use_noisy : bool, optional
        Whether to use noisy networks. Default is True.
    use_dueling : bool, optional
        Whether to use dueling architecture. Default is True.
    use_double : bool, optional
        Whether to use double Q-learning. Default is True.
    use_per : bool, optional
        Whether to use prioritized replay. Default is True.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float = 6.25e-5,
        gamma: float = 0.99,
        n_step: int = 3,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        buffer_size: int = 100_000,
        batch_size: int = 32,
        target_update_freq: int = 8_000,
        learning_starts: int = 20_000,
        train_freq: int = 4,
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_frames: int = 100_000,
        use_noisy: bool = True,
        use_dueling: bool = True,
        use_double: bool = True,
        use_per: bool = True,
    ):
        super().__init__(observation_space, action_space)

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("RainbowAgent only supports Discrete action spaces.")

        self.num_actions = int(action_space.n)
        self.gamma = gamma
        self.n_step = n_step
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.use_double = use_double
        self.use_per = use_per
        self.use_noisy = use_noisy

        # ##>: Value support for distributional RL.
        self.support = np.linspace(v_min, v_max, num_atoms).astype(np.float32)

        # ##>: Build online and target networks.
        self.online_network = build_rainbow_network(
            self.obs_shape, self.num_actions, num_atoms, v_min, v_max, use_noisy, use_dueling
        )
        self.target_network = build_rainbow_network(
            self.obs_shape, self.num_actions, num_atoms, v_min, v_max, use_noisy, use_dueling
        )
        self._update_target_network()

        # ##>: Optimizer.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1.5e-4)

        # ##>: Replay buffer.
        self.buffer = PrioritizedReplayBuffer(
            obs_shape=self.obs_shape,
            capacity=buffer_size,
            alpha=priority_alpha if use_per else 0.0,
            beta_start=priority_beta_start,
            beta_frames=priority_beta_frames,
            n_step=n_step,
            gamma=gamma,
        )

        # ##>: Training state.
        self.total_steps = 0
        self.updates = 0

        # ##>: Epsilon for non-noisy exploration fallback.
        self.epsilon = 1.0 if not use_noisy else 0.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01

    def _update_target_network(self) -> None:
        """Hard update: copy online weights to target."""
        self.target_network.set_weights(self.online_network.get_weights())

    def get_action(self, state: np.ndarray, training: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Select action using distributional Q-values.

        With Noisy Networks, no explicit epsilon-greedy is needed during training.
        For non-noisy networks, uses epsilon-greedy exploration.

        Parameters
        ----------
        state : np.ndarray
            Current observation(s).
        training : bool, optional
            Whether in training mode. Default is True.

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            Tuple of (actions, info_dict) where info_dict contains 'q_values'.
        """
        if state.ndim == len(self.obs_shape):
            state = np.expand_dims(state, 0)

        batch_size = state.shape[0]

        # ##>: Epsilon-greedy for non-noisy networks.
        if not self.use_noisy and training and np.random.random() < self.epsilon:
            actions = np.random.randint(0, self.num_actions, size=batch_size)
            return actions, {"q_values": np.zeros((batch_size, self.num_actions))}

        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        q_dist = self.online_network(state_tensor, training=training)

        # ##>: Expected Q-value = sum(probability * support).
        q_values = tf.reduce_sum(q_dist * self.support, axis=-1)
        actions = tf.argmax(q_values, axis=-1).numpy()

        return actions, {"q_values": q_values.numpy()}

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store transition in replay buffer.

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Next observation.
        done : bool
            Whether episode ended.
        """
        self.buffer.store(state, action, reward, next_state, done)
        self.total_steps += 1

        if done:
            self.buffer.flush_n_step_buffer()

        # ##>: Decay epsilon for non-noisy networks.
        if not self.use_noisy:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def store_transitions_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Store a batch of transitions from vectorized environments.

        More efficient than calling store_transition in a loop.

        Parameters
        ----------
        states : np.ndarray
            Batch of observations, shape (num_envs, *obs_shape).
        actions : np.ndarray
            Batch of actions, shape (num_envs,).
        rewards : np.ndarray
            Batch of rewards, shape (num_envs,).
        next_states : np.ndarray
            Batch of next observations, shape (num_envs, *obs_shape).
        dones : np.ndarray
            Batch of done flags, shape (num_envs,).
        """
        self.buffer.store_batch(states, actions, rewards, next_states, dones)
        self.total_steps += len(states)

        # ##>: Decay epsilon for non-noisy networks.
        if not self.use_noisy:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, **kwargs: Any) -> dict[str, float]:
        """
        Perform one training step if conditions are met.

        Returns
        -------
        dict[str, float]
            Training metrics (loss, mean_q) or empty dict if no update.
        """
        if self.total_steps < self.learning_starts:
            return {}

        if self.total_steps % self.train_freq != 0:
            return {}

        if len(self.buffer) < self.batch_size:
            return {}

        # ##>: Sample from buffer.
        states, actions, rewards, next_states, dones, tree_indices, weights = self.buffer.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        # ##>: Double DQN: select action with online, evaluate with target.
        if self.use_double:
            next_q_dist_online = self.online_network(next_states, training=False)
            next_q_values = tf.reduce_sum(next_q_dist_online * self.support, axis=-1)
            next_actions = tf.argmax(next_q_values, axis=-1)
        else:
            next_q_dist_target = self.target_network(next_states, training=False)
            next_q_values = tf.reduce_sum(next_q_dist_target * self.support, axis=-1)
            next_actions = tf.argmax(next_q_values, axis=-1)

        # ##>: Get distribution for selected actions from target.
        next_q_dist_target = self.target_network(next_states, training=False)
        batch_indices = tf.stack([tf.range(self.batch_size), tf.cast(next_actions, tf.int32)], axis=1)
        next_q_dist = tf.gather_nd(next_q_dist_target, batch_indices)

        # ##>: Compute target distribution.
        target_dist = compute_target_distribution(
            rewards, next_q_dist, dones, self.gamma, self.n_step, self.v_min, self.v_max, self.num_atoms
        )

        # ##>: Compute loss and update.
        with tf.GradientTape() as tape:
            q_dist = self.online_network(states, training=True)
            loss, td_errors = categorical_dqn_loss(q_dist, target_dist, actions, weights)

        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        # ##>: Clip gradients.
        gradients = [tf.clip_by_norm(g, 10.0) if g is not None else g for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.online_network.trainable_variables))

        # ##>: Update priorities if using PER.
        if self.use_per:
            self.buffer.update_priorities(tree_indices, td_errors.numpy())

        # ##>: Update target network periodically.
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self._update_target_network()
            logger.debug("Target network updated.")

        return {"loss": float(loss.numpy()), "mean_q": float(tf.reduce_mean(next_q_values).numpy())}

    def save_models(self, path_prefix: str) -> None:
        """Save online and target networks."""
        Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
        self.online_network.save(f"{path_prefix}_online.keras")
        self.target_network.save(f"{path_prefix}_target.keras")
        logger.info(f"Rainbow models saved to {path_prefix}_*.keras")

    def load_models(self, path_prefix: str) -> None:
        """Load online and target networks."""
        online_path = Path(f"{path_prefix}_online.keras")
        target_path = Path(f"{path_prefix}_target.keras")

        try:
            self.online_network = tf.keras.models.load_model(
                online_path,
                custom_objects={
                    "NoisyDense": __import__("reinforce.core.network_utils", fromlist=["NoisyDense"]).NoisyDense
                },
            )
            self.target_network = tf.keras.models.load_model(
                target_path,
                custom_objects={
                    "NoisyDense": __import__("reinforce.core.network_utils", fromlist=["NoisyDense"]).NoisyDense
                },
            )
            logger.info(f"Rainbow models loaded from {path_prefix}_*.keras")
        except (OSError, FileNotFoundError, tf.errors.OpError) as exc:
            logger.warning(f"Error loading models: {exc}")

    def on_episode_end(self) -> None:
        """Handle episode end for buffer flushing."""
        self.buffer.flush_n_step_buffer()
