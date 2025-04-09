# -*- coding: utf-8 -*-
"""
Pydantic model for A2C (Advantage Actor-Critic) agent configuration.
"""

from typing import Literal

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
    """

    agent_type: Literal["A2C"] = "A2C"
    embedding_size: int = Field(128, ge=1, description="Size of the embedding layer")
    learning_rate: float = Field(0.001, ge=0, description="Learning rate for the optimizer")
    entropy_coef: float = Field(0.01, ge=0, description="Entropy regularization coefficient")
    value_coef: float = Field(0.5, ge=0, description="Value loss coefficient")
