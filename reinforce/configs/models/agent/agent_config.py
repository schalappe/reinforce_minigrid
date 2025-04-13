# -*- coding: utf-8 -*-
"""
Base Pydantic model for agent configurations.
"""

from typing import Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """
    Base model for agent configurations.

    Attributes
    ----------
    agent_type: str
        Type of agent to use. This field is required.
    discount_factor: float, default=0.99
        Discount factor for future rewards. Must be between 0 and 1 (inclusive).
    embedding_size: int, default=128
        Size of the embedding layer in the model. Must be at least 1.
    learning_rate: float, default=3e-4
        Learning rate for the optimizer. Must be non-negative.
    gae_lambda: float, default=0.95
        Factor for Generalized Advantage Estimation (GAE). Must be between 0 and 1.
    entropy_coef: float, default=0.01
        Coefficient for the entropy bonus in the loss. Must be non-negative.
    value_coef: float, default=0.5
        Coefficient for the value function loss. Must be non-negative.
    max_grad_norm: Optional[float], default=0.5
        Maximum norm for gradient clipping. Set to `None` to disable. Must be non-negative if set.
    lr_schedule_enabled: bool, default=False
        Whether to use learning rate scheduling.
    lr_decay_factor: float, default=0.1
        Factor by which to decay learning rate. Must be between 0 and 1.
    max_total_steps: int, default=1_000_000
        Maximum total steps for learning rate schedule. Must be at least 1.
    """

    agent_type: str = Field(..., description="Type of agent to use")
    discount_factor: float = Field(0.99, ge=0, le=1, description="Discount factor for future rewards")
    embedding_size: int = Field(128, ge=1, description="Size of the embedding layer in the model")

    # ##: Loss coefficients.
    entropy_coef: float = Field(0.01, ge=0, description="Coefficient for the entropy bonus in the loss")
    value_coef: float = Field(0.5, ge=0, description="Coefficient for the value function loss")
    gae_lambda: float = Field(0.95, ge=0, le=1, description="Factor for Generalized Advantage Estimation (GAE)")
    max_grad_norm: Optional[float] = Field(
        0.5, ge=0, description="Maximum norm for gradient clipping (None to disable)"
    )

    # ##: Learning rate scheduling.
    lr_schedule_enabled: bool = Field(False, description="Whether to use learning rate scheduling")
    lr_decay_factor: float = Field(0.1, ge=0, le=1, description="Factor by which to decay learning rate")
    max_total_steps: int = Field(1_000_000, ge=1, description="Maximum total steps for learning rate schedule")
    learning_rate: float = Field(3e-4, ge=0, description="Learning rate for the optimizer")
