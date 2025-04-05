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

    Notes
    -----
    This is a base configuration model that should be extended by specific agent configurations.
    All agent configurations should inherit from this class.
    """

    agent_type: str = Field(..., description="Type of agent to use")
