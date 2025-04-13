# -*- coding: utf-8 -*-
"""
Pydantic model for episode trainer configuration.

This module defines the EpisodeTrainerConfig class, which extends TrainerConfig to provide
configuration specific to episode-based training.
"""

from typing import Literal

from pydantic import Field

from .trainer_config import TrainerConfig


class A2CTrainerConfig(TrainerConfig):
    """
    Pydantic model for episode trainer configuration.

    This configuration class is used for trainers that operate on an episode basis,
    where training progresses through discrete episodes rather than continuous steps.

    Attributes
    ----------
    trainer_type : Literal["A2CTrainer"]
        The type of trainer, fixed as "EpisodeTrainer" for this configuration.
    max_episodes : int
        Maximum number of episodes to train for. Must be at least 1.
        Default: 10000.
    update_frequency : int
        Number of steps between agent updates. Must be at least 1.
        Default: 1.
    """

    trainer_type: Literal["A2CTrainer"] = "A2CTrainer"
    max_episodes: int = Field(10000, ge=1, description="Maximum number of episodes to train for")
    update_frequency: int = Field(1, ge=1, description="Number of steps between agent updates")
    buffer_capacity: int = Field(10000, ge=1, description="Capacity of the replay buffer")
    batch_size: int = Field(64, ge=1, description="Batch size for sampling from the replay buffer")

    def get_max_steps(self) -> int:
        """
        Get the maximum number of steps for the full training run.

        Returns
        -------
        int
            Maximum number of steps for the full training.
        """
        return self.max_episodes * self.max_steps_per_episode
