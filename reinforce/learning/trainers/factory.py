# -*- coding: utf-8 -*-
"""
Factory for creating trainer instances.
"""

from typing import Union

from reinforce.agents.actor_critic import A2CAgent, PPOAgent
from reinforce.configs.models import TrainerConfigUnion
from reinforce.environments import BaseEnvironment
from reinforce.learning.trainers import A2CTrainer, BaseTrainer, PPOTrainer
from reinforce.utils.management import AimTracker

Agent = Union[A2CAgent, PPOAgent]


class TrainerFactory:
    """Factory class for creating trainer instances based on configuration."""

    @staticmethod
    def create(
        trainer_config: TrainerConfigUnion, agent: Agent, environment: BaseEnvironment, tracker: AimTracker
    ) -> BaseTrainer:
        """
        Create a trainer based on the Pydantic configuration model, injecting dependencies.

        This method creates and returns a trainer instance based on the provided config model.
        It uses the `trainer_type` field to determine which trainer class to instantiate.

        Parameters
        ----------
        trainer_config : TrainerConfigUnion
            Pydantic trainer configuration model (e.g., A2CTrainerConfig).
        agent : A2CAgent | PPOAgent
            The agent instance to be trained.
        environment : BaseEnvironment
            The environment instance to train in.
        tracker : AimTracker
            Tracker instance for experiment tracking.

        Returns
        -------
        BaseTrainer
            Created trainer instance.

        Raises
        ------
        ValueError
            If the trainer type specified in the config is not supported.
        """
        if trainer_config.trainer_type == "A2CTrainer":
            return A2CTrainer(agent=agent, environment=environment, config=trainer_config, tracker=tracker)

        if trainer_config.trainer_type == "PPOTrainer":
            return PPOTrainer(agent=agent, environment=environment, config=trainer_config, tracker=tracker)

        raise ValueError(f"Unsupported trainer type: {trainer_config.trainer_type}")
