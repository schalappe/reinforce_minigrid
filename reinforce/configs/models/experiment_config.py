# -*- coding: utf-8 -*-
"""
Pydantic model for the overall experiment configuration.

This module defines the main configuration structure for reinforcement learning experiments,
including agent, trainer, and environment configurations.
"""

from typing import Annotated, Optional, Union

from pydantic import BaseModel, Field

from reinforce.configs.models.a2c_config import A2CConfig
from reinforce.configs.models.distributed_trainer_config import DistributedTrainerConfig
from reinforce.configs.models.environment_config import EnvironmentConfig
from reinforce.configs.models.episode_trainer_config import EpisodeTrainerConfig

# ##: Define discriminated unions for agent and trainer configs.
AgentConfigUnion = Annotated[Union[A2CConfig], Field(discriminator="agent_type")]
TrainerConfigUnion = Annotated[
    Union[EpisodeTrainerConfig, DistributedTrainerConfig], Field(discriminator="trainer_type")
]


class ExperimentConfig(BaseModel):
    """
    Pydantic model for the overall experiment configuration.

    Attributes
    ----------
    agent : AgentConfigUnion
        Configuration for the reinforcement learning agent.
    trainer : TrainerConfigUnion
        Configuration for the training process (episode-based or distributed).
    environment : EnvironmentConfig, optional
        Configuration for the RL environment (default: EnvironmentConfig()).
    aim_experiment_name : str, optional
        Custom name for the AIM experiment tracking (default: None).
    trial_info : dict, optional
        Internal field for Optuna trial information (excluded from serialization).

    Notes
    -----
    - The Config class enables arbitrary types to support custom callbacks.
    - Agent and trainer configurations use discriminated unions for type safety.
    """

    agent: AgentConfigUnion = Field(description="Agent configuration")
    trainer: TrainerConfigUnion = Field(description="Trainer configuration")
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Environment configuration")
    aim_experiment_name: Optional[str] = Field(None, description="Custom name for the AIM experiment")

    # ##: Field for Optuna trial info (internal, added by hyperparameter search).
    trial_info: Optional[dict] = Field(None, exclude=True)

    class Config:
        """
        Pydantic model configuration settings.

        Attributes
        ----------
        arbitrary_types_allowed : bool
            Allows arbitrary Python types in model fields (default: True).
            Required for custom callback objects and other non-serializable types.
        """

        # ##: Allow arbitrary types for callbacks etc.
        arbitrary_types_allowed = True
