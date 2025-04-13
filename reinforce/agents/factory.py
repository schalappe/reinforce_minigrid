# -*- coding: utf-8 -*-
"""
Factory for creating agent instances.
"""

from typing import Union

from reinforce.agents.actor_critic import A2CAgent, PPOAgent
from reinforce.agents.models import SeparateNetACModel
from reinforce.configs.models.agent import A2CConfig, PPOConfig


class AgentFactory:
    """Factory class for creating agent instances based on configuration."""

    @staticmethod
    def create(agent_config: Union[A2CConfig, PPOConfig], env_action_space_size: int) -> Union[A2CAgent, PPOAgent]:
        """
        Create an agent based on the Pydantic configuration model.

        This method creates and returns an agent instance based on the provided config model.
        It uses the `agent_type` field to determine which agent class to instantiate.

        Parameters
        ----------
        agent_config : A2CConfig | PPOConfig
            Pydantic agent configuration model (e.g., A2CConfig).
        env_action_space_size : int
            The size of the environment's action space.

        Returns
        -------
        A2CAgent | PPOAgent
            Created agent instance.

        Raises
        ------
        ValueError
            If the agent type specified in the config is not supported.
        """
        model = SeparateNetACModel(action_space=env_action_space_size, embedding_size=agent_config.embedding_size)

        if agent_config.agent_type == "A2C":
            return A2CAgent(model=model, hyperparameters=agent_config)
        if agent_config.agent_type == "PPO":
            return PPOAgent(model=model, hyperparameters=agent_config)

        raise ValueError(f"Unsupported agent type: {agent_config.agent_type}")
