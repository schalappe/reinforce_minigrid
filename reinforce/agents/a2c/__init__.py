# -*- coding: utf-8 -*-
"""
A2C (Advantage Actor-Critic) algorithm implementation.
"""

from reinforce.agents.a2c.a2c_model import A2CModel
from reinforce.agents.a2c.agent import A2CAgent
from reinforce.agents.a2c.cnn_encoder import CNNEncoder

__all__ = [
    "A2CModel",
    "CNNEncoder",
    "A2CAgent",
]
