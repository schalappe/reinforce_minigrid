# -*- coding: utf-8 -*-
"""
Pydantic model for A2C (Advantage Actor-Critic) agent configuration.
"""

from typing import Literal

from .agent_config import AgentConfig


class A2CConfig(AgentConfig):
    """
    Pydantic model for A2C (Advantage Actor-Critic) agent configuration.

    Attributes
    ----------
    agent_type : Literal["A2C"]
        Type of the agent, fixed as "A2C".
    """

    agent_type: Literal["A2C"] = "A2C"
