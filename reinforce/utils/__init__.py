# -*- coding: utf-8 -*-
"""
Utility functions and classes for reinforcement learning.
"""

from reinforce.utils.preprocessing import (
    frame_stack,
    normalize_image,
    preprocess_observation,
)
from reinforce.utils.buffers import PrioritizedReplayBuffer, ReplayBuffer
from reinforce.utils.losses import (
    huber_loss,
    compute_policy_gradient_loss,
    compute_value_loss,
    compute_entropy_loss,
    compute_a2c_loss
)

__all__ = [
    "normalize_image",
    "preprocess_observation",
    "frame_stack",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "huber_loss",
    "compute_policy_gradient_loss",
    "compute_value_loss",
    "compute_entropy_loss",
    "compute_a2c_loss"
]
