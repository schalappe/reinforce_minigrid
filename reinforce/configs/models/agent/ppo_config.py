# -*- coding: utf-8 -*-
"""
Pydantic model for Proximal Policy Optimization (PPO) agent configuration.
"""

from typing import Literal, Optional

from pydantic import Field

from .agent_config import AgentConfig


class PPOConfig(AgentConfig):
    """
    Configuration settings for the Proximal Policy Optimization (PPO) agent.

    Attributes
    ----------
    agent_type: Literal["PPO"]
        Always set to "PPO" to indicate this is a PPO agent.
    embedding_size: int
        Size of the embedding layer in the model. Default: 128.
    learning_rate: float
        Learning rate for the Adam optimizer. Default: 3e-4.
    gae_lambda: float
        Factor for Generalized Advantage Estimation (GAE). Default: 0.95.
    clip_range: float
        Clipping parameter epsilon for the PPO objective. Default: 0.2.
    entropy_coef: float
        Coefficient for the entropy bonus in the loss. Default: 0.01.
    value_coef: float
        Coefficient for the value function loss. Default: 0.5.
    max_grad_norm: float, optional
        Maximum norm for gradient clipping. Set to `None` to disable. Default: 0.5.
    use_adaptive_kl: bool
        Whether to use adaptive KL penalty. Default: False.
    target_kl: float
        Target KL divergence for adaptive penalty. Default: 0.01.
    initial_kl_coeff: float
        Initial coefficient for the KL penalty. Default: 1.0.
    """

    agent_type: Literal["PPO"] = "PPO"
    embedding_size: int = Field(128, description="Size of the embedding layer in the model")
    learning_rate: float = Field(3e-4, description="Learning rate for the Adam optimizer")
    gae_lambda: float = Field(0.95, description="Factor for Generalized Advantage Estimation (GAE)")
    clip_range: float = Field(0.2, description="Clipping parameter epsilon for the PPO objective")
    entropy_coef: float = Field(0.01, description="Coefficient for the entropy bonus in the loss")
    value_coef: float = Field(0.5, description="Coefficient for the value function loss")
    max_grad_norm: Optional[float] = Field(0.5, description="Maximum norm for gradient clipping (None to disable)")

    # Adaptive KL Penalty options
    use_adaptive_kl: bool = Field(False, description="Whether to use adaptive KL penalty")
    target_kl: float = Field(0.01, description="Target KL divergence for adaptive penalty")
    initial_kl_coeff: float = Field(1.0, description="Initial coefficient for the KL penalty")

    class Config:
        """
        Pydantic configuration settings for the `PPOConfig` class.

        Attributes:
            allow_population_by_field_name (bool): Allows populating model fields by both field name and alias.
            validate_assignment (bool): Enables validation on attribute assignment.
            frozen (bool): Makes the model immutable after creation.
        """

        allow_population_by_field_name = True
        validate_assignment = True
        frozen = True
