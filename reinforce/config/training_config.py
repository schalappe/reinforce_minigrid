# -*- coding: utf-8 -*-
"""
Dataclass definitions for PPO training configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvironmentConfig:
    """Configuration related to the training environment."""

    seed: int = 42
    """Random seed for reproducibility across environment and libraries."""


@dataclass
class PPOHyperparameters:
    """Hyperparameters specific to the PPO algorithm."""

    learning_rate: float = 3e-4
    """Learning rate for the Adam optimizer."""
    gamma: float = 0.99
    """Discount factor for future rewards."""
    lambda_gae: float = field(default=0.95, metadata={"yaml_name": "lambda"})
    """Lambda parameter for Generalized Advantage Estimation (GAE)."""
    clip_param: float = 0.2
    """Clipping parameter epsilon for the PPO surrogate objective."""
    entropy_coef: float = 0.01
    """Coefficient for the entropy bonus in the loss function."""
    value_coef: float = field(default=0.5, metadata={"yaml_name": "vf_coef"})
    """Coefficient for the value function loss."""
    epochs: int = 10
    """Number of optimization epochs per learning update."""
    batch_size: int = 64
    """Mini-batch size for optimization."""


@dataclass
class TrainingConfig:
    """Configuration for the overall training process."""

    total_timesteps: int = 1_000_000
    """Total number of environment steps to train for."""
    steps_per_update: int = 2048
    """Number of steps collected from the environment before each agent learning update."""


@dataclass
class LoggingConfig:
    """Configuration related to logging and model saving."""

    log_interval: int = 1
    """Log progress every N update cycles."""
    save_interval: int = 10
    """Save models every N update cycles."""
    save_path: str = "models/ppo_maze"
    """Directory and prefix for saving models. Suffixes like '_final.keras' will be added."""
    load_path: Optional[str] = None
    """Path prefix to load pre-trained models from. If None, training starts from scratch."""


@dataclass
class MainConfig:
    """Root configuration class aggregating all sub-configurations."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ppo: PPOHyperparameters = field(default_factory=PPOHyperparameters)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Placeholder for potential future validation logic
    def __post_init__(self):
        """Perform validation after initialization."""
        # Example: Ensure learning rate is positive
        if self.ppo.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        # Add more complex validation logic here or in a dedicated validator module
        pass
