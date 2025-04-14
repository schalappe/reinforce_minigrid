# -*- coding: utf-8 -*-
"""
Pydantic model for Proximal Policy Optimization (PPO) agent configuration.
"""

from typing import Literal

from pydantic import Field

from .agent_config import AgentConfig


class PPOConfig(AgentConfig):
    """
    Configuration settings for the Proximal Policy Optimization (PPO) agent.

    Attributes
    ----------
    agent_type: Literal["PPO"]
        Always set to "PPO" to indicate this is a PPO agent.
    clip_range: float
        Clipping parameter epsilon for the PPO objective. Default: 0.2.
    use_adaptive_kl: bool
        Whether to use adaptive KL penalty. Default: False.
    target_kl: float
        Target KL divergence for adaptive penalty. Default: 0.01.
    initial_kl_coeff: float
        Initial coefficient for the KL penalty. Default: 1.0.
    """

    agent_type: Literal["PPO"] = "PPO"
    clip_range: float = Field(0.2, description="Clipping parameter epsilon for the PPO objective")

    # ##: Adaptive KL Penalty options.
    use_adaptive_kl: bool = Field(False, description="Whether to use adaptive KL penalty")
    target_kl: float = Field(0.01, description="Target KL divergence for adaptive penalty")
    initial_kl_coeff: float = Field(1.0, description="Initial coefficient for the KL penalty")

    class Config:
        """
        Pydantic configuration settings for the `PPOConfig` class.

        Attributes
        ----------
        allow_population_by_field_name: bool
            Allows populating model fields by both field name and alias.
        validate_assignment: bool
            Enables validation on attribute assignment.
        frozen: bool
            Makes the model immutable after creation.
        """

        allow_population_by_field_name = True
        validate_assignment = True
