# -*- coding: utf-8 -*-
"""
Pydantic model for episode trainer configuration.

This module defines the EpisodeTrainerConfig class, which extends TrainerConfig to provide
configuration specific to episode-based training.
"""

from typing import Literal

from pydantic import Field

from reinforce.configs.models.trainer_config import TrainerConfig


class EpisodeTrainerConfig(TrainerConfig):
    """
    Pydantic model for episode trainer configuration.

    This configuration class is used for trainers that operate on an episode basis,
    where training progresses through discrete episodes rather than continuous steps.

    Attributes
    ----------
    trainer_type : Literal["EpisodeTrainer"]
        The type of trainer, fixed as "EpisodeTrainer" for this configuration.
    max_episodes : int
        Maximum number of episodes to train for. Must be at least 1.
        Default: 10000.
    update_frequency : int
        Number of steps between agent updates. Must be at least 1.
        Default: 1.

    Notes
    -----
    This class inherits all attributes from TrainerConfig and adds episode-specific configuration parameters.
    """

    trainer_type: Literal["EpisodeTrainer"] = "EpisodeTrainer"
    max_episodes: int = Field(10000, ge=1, description="Maximum number of episodes to train for")
    update_frequency: int = Field(1, ge=1, description="Number of steps between agent updates")
