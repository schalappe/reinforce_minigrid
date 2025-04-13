# -*- coding: utf-8 -*-
"""
Base Pydantic model for agent configurations.
"""

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """
    Base model for agent configurations.

    Attributes
    ----------
    agent_type : str
        Type of agent to use. This field is required.
    discount_factor : float, default=0.99
        Discount factor for future rewards. Must be between 0 and 1 (inclusive).

    Notes
    -----
    This is a base configuration model that should be extended by specific agent configurations.
    All agent configurations should inherit from this class.
    """

    agent_type: str = Field(..., description="Type of agent to use")
    discount_factor: float = Field(0.99, ge=0, le=1, description="Discount factor for future rewards")
