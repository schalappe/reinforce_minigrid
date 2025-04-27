# -*- coding: utf-8 -*-
"""
Dataclass definitions for PPO training configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvironmentConfig:
    """
    Configuration related to the training environment.

    Attributes
    ----------
    seed : int, optional
        Random seed for reproducibility across environment and libraries. Default is 42.
    """

    seed: int = 42


@dataclass
class PPOHyperparameters:
    """
    Hyperparameters specific to the PPO algorithm.

    Attributes
    ----------
    learning_rate : float, optional
        Learning rate for the Adam optimizer. Default is 3e-4.
    gamma : float, optional
        Discount factor for future rewards. Default is 0.99.
    lambda_gae : float, optional
        Lambda parameter for Generalized Advantage Estimation (GAE). Default is 0.95.
    clip_param : float, optional
        Clipping parameter epsilon for the PPO surrogate objective. Default is 0.2.
    entropy_coef : float, optional
        Coefficient for the entropy bonus in the loss function. Default is 0.01.
    value_coef : float, optional
        Coefficient for the value function loss. Default is 0.5.
    epochs : int, optional
        Number of optimization epochs per learning update. Default is 10.
    batch_size : int, optional
        Mini-batch size for optimization. Default is 64.
    """

    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = field(default=0.95, metadata={"yaml_name": "lambda"})
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = field(default=0.5, metadata={"yaml_name": "vf_coef"})
    epochs: int = 10
    batch_size: int = 64


@dataclass
class TrainingConfig:
    """
    Configuration for the overall training process.

    Attributes
    ----------
    total_timesteps : int, optional
        Total number of environment steps to train for. Default is 1,000,000.
    steps_per_update : int, optional
        Number of steps collected from the environment before each agent learning update. Default is 2048.
    num_envs : int, optional
        Number of parallel environments to use for experience collection. Default is 4.
    """

    total_timesteps: int = 1_000_000
    steps_per_update: int = 2048
    num_envs: int = 4


@dataclass
class LoggingConfig:
    """
    Configuration related to logging and model saving.

    Attributes
    ----------
    log_interval : int, optional
        Interval for logging training progress. Default is 1.
    save_interval : int, optional
        Interval for saving the model. Default is 10.
    save_path : str, optional
        Directory path for saving models. Default is 'models/ppo_maze'.
    load_path : str, optional
        Path prefix to load pre-trained models from. Default is None.
    """

    log_interval: int = 1
    save_interval: int = 10
    save_path: str = "models/ppo_maze"
    load_path: Optional[str] = None


@dataclass
class MainConfig:
    """
    Root configuration class aggregating all sub-configurations.

    Attributes
    ----------
    environment : EnvironmentConfig, optional
        Configuration related to the environment. Default is an instance of EnvironmentConfig.
    training : TrainingConfig, optional
        Configuration related to the training process. Default is an instance of TrainingConfig.
    ppo : PPOHyperparameters, optional
        Configuration specific to the PPO algorithm. Default is an instance of PPOHyperparameters.
    logging : LoggingConfig, optional
        Configuration related to logging and model saving. Default is an instance of LoggingConfig.
    """

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ppo: PPOHyperparameters = field(default_factory=PPOHyperparameters)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Perform validation after initialization."""
        if self.ppo.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
