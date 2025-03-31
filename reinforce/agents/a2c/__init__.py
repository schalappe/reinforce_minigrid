# -*- coding: utf-8 -*-
"""
A2C (Advantage Actor-Critic) algorithm implementation.
"""

from reinforce.agents.a2c.agent import A2CAgent
from reinforce.agents.a2c.model import A2CModel, CNNEncoder

__all__ = [
    "A2CModel",
    "CNNEncoder",
    "A2CAgent",
]
