"""
Hybrid exploration strategies for PPO.

Implements:
1. ε-greedy exploration: Random actions with probability ε
2. UCB (Upper Confidence Bound): Bonus for infrequently chosen actions
3. Adaptive entropy: Adjusts entropy coefficient based on policy entropy

These strategies complement the intrinsic motivation from RND by adding explicit action-level
exploration incentives.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from reinforce.config.training_config import ExplorationConfig


class ExplorationManager:
    """
    Manages hybrid exploration strategies for PPO training.

    Provides action modification through ε-greedy and UCB bonuses,
    plus adaptive entropy coefficient adjustment.
    """

    def __init__(
        self,
        num_actions: int,
        num_envs: int,
        config: ExplorationConfig | None = None,
        base_entropy_coef: float = 0.01,
    ):
        """
        Initialize the exploration manager.

        Parameters
        ----------
        num_actions : int
            Number of discrete actions.
        num_envs : int
            Number of parallel environments.
        config : ExplorationConfig, optional
            Exploration configuration. Uses defaults if not provided.
        base_entropy_coef : float, optional
            Base entropy coefficient from PPO config. Default is 0.01.
        """
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.config = config or ExplorationConfig()

        # ##>: ε-greedy state.
        self.current_epsilon = self.config.epsilon_start
        self.total_steps = 0

        # ##>: UCB action counts per environment.
        self.action_counts = np.ones((num_envs, num_actions), dtype=np.float32)
        self.total_action_counts = np.ones(num_envs, dtype=np.float32) * num_actions

        # ##>: Adaptive entropy state.
        self.entropy_coef = base_entropy_coef
        self.entropy_history: deque[float] = deque(maxlen=100)
        self.max_entropy = np.log(num_actions)

    def get_epsilon(self) -> float:
        """Get current epsilon value with linear decay."""
        if not self.config.use_epsilon_greedy:
            return 0.0

        progress = min(1.0, self.total_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def compute_ucb_bonus(self, env_indices: np.ndarray | None = None) -> np.ndarray:
        """
        Compute UCB exploration bonus for action logits.

        Parameters
        ----------
        env_indices : np.ndarray, optional
            Indices of environments to compute bonus for.
            If None, computes for all environments.

        Returns
        -------
        np.ndarray
            UCB bonus to add to action logits. Shape: (num_envs, num_actions).
        """
        if not self.config.use_ucb:
            return np.zeros((self.num_envs, self.num_actions), dtype=np.float32)

        if env_indices is None:
            action_counts = self.action_counts
            total_counts = self.total_action_counts
        else:
            action_counts = self.action_counts[env_indices]
            total_counts = self.total_action_counts[env_indices]

        # ##>: UCB formula: sqrt(2 * ln(N) / n_a) where N is total, n_a is action count.
        log_total = np.log(total_counts + 1)[:, np.newaxis]
        ucb_bonus = self.config.ucb_coefficient * np.sqrt(2 * log_total / (action_counts + 1e-8))

        return ucb_bonus.astype(np.float32)

    def apply_exploration(
        self,
        action_logits: np.ndarray,
        sampled_actions: np.ndarray,
    ) -> np.ndarray:
        """
        Apply hybrid exploration to modify actions.

        Parameters
        ----------
        action_logits : np.ndarray
            Raw action logits from policy network. Shape: (num_envs, num_actions).
        sampled_actions : np.ndarray
            Actions sampled from policy distribution. Shape: (num_envs,).

        Returns
        -------
        np.ndarray
            Modified actions after exploration. Shape: (num_envs,).
        """
        actions = sampled_actions.copy()
        num_envs = len(actions)

        # ##>: ε-greedy: Random action with probability ε.
        if self.config.use_epsilon_greedy:
            epsilon = self.get_epsilon()
            random_mask = np.random.random(num_envs) < epsilon
            random_actions = np.random.randint(0, self.num_actions, size=num_envs)
            actions = np.where(random_mask, random_actions, actions)

        return actions

    def modify_logits_with_ucb(self, action_logits: np.ndarray) -> np.ndarray:
        """
        Add UCB bonus to action logits before sampling.

        Parameters
        ----------
        action_logits : np.ndarray
            Raw action logits. Shape: (num_envs, num_actions).

        Returns
        -------
        np.ndarray
            Modified logits with UCB bonus. Shape: (num_envs, num_actions).
        """
        if not self.config.use_ucb:
            return action_logits

        ucb_bonus = self.compute_ucb_bonus()
        return action_logits + ucb_bonus

    def update_action_counts(self, actions: np.ndarray) -> None:
        """
        Update UCB action counts after taking actions.

        Parameters
        ----------
        actions : np.ndarray
            Actions taken. Shape: (num_envs,).
        """
        for i, action in enumerate(actions):
            self.action_counts[i, action] += 1
            self.total_action_counts[i] += 1

    def reset_counts_for_env(self, env_idx: int) -> None:
        """Reset action counts for a specific environment (e.g., on episode end)."""
        self.action_counts[env_idx] = 1.0
        self.total_action_counts[env_idx] = float(self.num_actions)

    def step(self, num_steps: int = 1) -> None:
        """Update internal step counter for epsilon decay."""
        self.total_steps += num_steps

    def update_entropy_coef(self, current_entropy: float) -> float:
        """
        Update entropy coefficient based on policy entropy.

        Adaptive entropy: If policy entropy drops too low (overconfident),
        increase entropy coefficient to encourage exploration.
        If entropy is too high (undecided), decrease it to exploit more.

        Parameters
        ----------
        current_entropy : float
            Current policy entropy from training.

        Returns
        -------
        float
            Updated entropy coefficient.
        """
        if not self.config.use_adaptive_entropy:
            return self.entropy_coef

        self.entropy_history.append(current_entropy)

        if len(self.entropy_history) < 10:
            return self.entropy_coef

        # ##>: Target entropy as fraction of maximum possible entropy.
        target_entropy = self.config.target_entropy_ratio * self.max_entropy
        mean_entropy = np.mean(list(self.entropy_history))

        # ##>: Adjust entropy coefficient based on deviation from target.
        entropy_error = target_entropy - mean_entropy

        # ##>: Positive error = entropy too low, increase coef.
        # Negative error = entropy too high, decrease coef.
        adjustment = self.config.entropy_lr * entropy_error
        self.entropy_coef = np.clip(
            self.entropy_coef + adjustment,
            self.config.min_entropy_coef,
            self.config.max_entropy_coef,
        )

        return self.entropy_coef

    def get_stage_entropy_coef(self, stage_index: int, num_stages: int = 4) -> float:
        """
        Get recommended entropy coefficient based on curriculum stage.

        Higher stages (harder mazes) benefit from more exploration.

        Parameters
        ----------
        stage_index : int
            Current curriculum stage index (0-based).
        num_stages : int, optional
            Total number of curriculum stages. Default is 4.

        Returns
        -------
        float
            Stage-specific entropy coefficient multiplier.
        """
        # ##>: Exponential increase in exploration for harder stages.
        # Base stage (0): 1.0x, Easy (1): 1.5x, Medium (2): 2.0x, Hard (3): 3.0x.
        multipliers = [1.0, 1.5, 2.0, 3.0]
        if stage_index < len(multipliers):
            return self.entropy_coef * multipliers[stage_index]
        return self.entropy_coef * multipliers[-1]

    def get_stats(self) -> dict:
        """Return current exploration statistics."""
        return {
            "epsilon": self.get_epsilon(),
            "entropy_coef": self.entropy_coef,
            "total_exploration_steps": self.total_steps,
            "mean_action_counts": float(np.mean(self.action_counts)),
        }
