# -*- coding: utf-8 -*-
"""
Pydantic model for A2C (Advantage Actor-Critic) agent configuration.
"""

from typing import Literal, Optional

from pydantic import Field

from .agent_config import AgentConfig


class A2CConfig(AgentConfig):
    """
    Pydantic model for A2C (Advantage Actor-Critic) agent configuration.

    Attributes
    ----------
    agent_type : Literal["A2C"]
        Type of the agent, fixed as "A2C".
    embedding_size : int, optional
        Size of the embedding layer (default=128). Must be at least 1.
    learning_rate : float, optional
        Learning rate for the optimizer (default=0.001). Must be non-negative.
    entropy_coef : float, optional
        Entropy regularization coefficient (default=0.01). Must be non-negative.
    value_coef : float, optional
        Value loss coefficient (default=0.5). Must be non-negative.
    gae_lambda : float, optional
        Factor for Generalized Advantage Estimation (GAE) (default=0.95). Must be between 0 and 1.
    lr_schedule_enabled : bool, optional
        Whether to use learning rate scheduling (default=False).
    lr_decay_factor : float, optional
        Factor by which to decay learning rate (default=0.1). Must be between 0 and 1.
    max_total_steps : int, optional
        Maximum total steps for learning rate schedule (default=1,000,000). Must be at least 1.
    max_grad_norm : Optional[float], optional
        Maximum gradient norm for clipping (default=None). Must be non-negative if set.
    """

    agent_type: Literal["A2C"] = "A2C"
    embedding_size: int = Field(128, ge=1, description="Size of the embedding layer")
    learning_rate: float = Field(0.001, ge=0, description="Learning rate for the optimizer")
    entropy_coef: float = Field(0.01, ge=0, description="Entropy regularization coefficient")
    value_coef: float = Field(0.5, ge=0, description="Value loss coefficient")
    gae_lambda: float = Field(0.95, ge=0, le=1, description="Factor for Generalized Advantage Estimation (GAE)")
    lr_schedule_enabled: bool = Field(False, description="Whether to use learning rate scheduling")
    lr_decay_factor: float = Field(0.1, ge=0, le=1, description="Factor by which to decay learning rate")
    max_total_steps: int = Field(1000000, ge=1, description="Maximum total steps for learning rate schedule")
    max_grad_norm: Optional[float] = Field(None, ge=0, description="Maximum gradient norm for clipping")
