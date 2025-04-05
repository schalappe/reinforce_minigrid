# -*- coding: utf-8 -*-
"""
Pydantic model for distributed trainer configuration.
"""

from typing import Literal

from pydantic import Field

from reinforce.configs.models.trainer_config import TrainerConfig


class DistributedTrainerConfig(TrainerConfig):
    """
    Pydantic model for distributed trainer configuration.

    Attributes
    ----------
    trainer_type : Literal["DistributedTrainer"]
        Type of the trainer, fixed as "DistributedTrainer".
    num_workers : int
        Number of worker processes to use. Must be greater than or equal to 1.
    max_episodes_per_worker : int
        Maximum number of episodes per worker. Must be greater than or equal to 1.
    update_frequency : int, optional
        Number of steps between agent updates. Default is 1. Must be greater than or equal to 1.

    Notes
    -----
    This configuration extends the base `TrainerConfig` to support distributed training.
    """

    trainer_type: Literal["DistributedTrainer"] = "DistributedTrainer"
    num_workers: int = Field(..., ge=1, description="Number of worker processes to use")
    max_episodes_per_worker: int = Field(..., ge=1, description="Maximum number of episodes per worker")
    update_frequency: int = Field(1, ge=1, description="Number of steps between agent updates")
