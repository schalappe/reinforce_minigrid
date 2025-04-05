# -*- coding: utf-8 -*-
"""
Loss functions for reinforcement learning.
"""

from reinforce.utils.losses.a2c import compute_a2c_loss
from reinforce.utils.losses.base import (
    compute_entropy_loss,
    compute_policy_gradient_loss,
    compute_value_loss,
    huber_loss,
)

__all__ = [
    "huber_loss",
    "compute_policy_gradient_loss",
    "compute_value_loss",
    "compute_entropy_loss",
    "compute_a2c_loss",
]
