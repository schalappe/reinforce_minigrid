# -*- coding: utf-8 -*-
"""
Pydantic model for PPO trainer configuration.
"""

from typing import Literal

from pydantic import Field

from .trainer_config import TrainerConfig


class PPOTrainerConfig(TrainerConfig):
    """
    Configuration settings for the PPOTrainer.

    Inherits from TrainerConfig and adds PPO-specific training loop parameters.

    Attributes
    ----------
    trainer_type : Literal["ppo"]
        Trainer type must be 'ppo'. Default is "ppo".
    n_steps : int
        Number of steps to collect per rollout (buffer size). Default is 2048.
    n_epochs : int
        Number of optimization epochs per rollout. Default is 10.
    batch_size : int
        Minibatch size for PPO updates. Default is 64.
    max_total_steps : int
        Total number of environment steps to train for. Default is 1,000,000.
    eval_frequency : int
        Number of episodes between evaluations. Default is 50.
    save_frequency : int
        Number of episodes between saving the model. Default is 100.
    log_frequency : int
        Number of episodes between logging progress. Default is 1.

    Notes
    -----
    - All integer parameters must be >= 1 (enforced via Pydantic Field constraints)
    - Configuration is immutable once created (frozen=True)
    """

    trainer_type: Literal["PPOTrainer"] = Field("PPOTrainer", description="Trainer type must be 'ppo'")

    # ##: PPO specific training loop parameters
    n_steps: int = Field(2048, ge=1, description="Number of steps to collect per rollout (buffer size)")
    n_epochs: int = Field(10, ge=1, description="Number of optimization epochs per rollout")
    batch_size: int = Field(64, ge=1, description="Minibatch size for PPO updates")
    max_total_steps: int = Field(1_000_000, ge=1, description="Total number of environment steps to train for")

    # ##: Override some base fields if defaults differ or interpretation changes
    eval_frequency: int = Field(50, ge=1, description="Number of episodes between evaluations")
    save_frequency: int = Field(100, ge=1, description="Number of episodes between saving the model")
    log_frequency: int = Field(1, ge=1, description="Number of episodes between logging progress")

    class Config:
        """
        Pydantic configuration settings.

        Attributes
        ----------
        arbitrary_types_allowed : bool
            Whether to allow arbitrary types. Default is True.
        validate_assignment : bool
            Whether to validate on assignment. Default is True.
        frozen : bool
            Whether the config is immutable after creation. Default is True.
        """

        arbitrary_types_allowed = True
        validate_assignment = True
        frozen = True
