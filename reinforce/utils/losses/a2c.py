# -*- coding: utf-8 -*-
"""
A2C-specific loss functions.
"""

from typing import Optional, Tuple, cast

import tensorflow as tf

from reinforce.utils.losses.base import (
    compute_entropy_loss,
    compute_policy_gradient_loss,
    compute_value_loss,
)


def compute_a2c_loss(
    action_log_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor,
    action_probs: Optional[tf.Tensor] = None,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute the combined Actor-Critic loss.

    Parameters
    ----------
    action_log_probs : tf.Tensor
        Log probabilities of actions taken.
    values : tf.Tensor
        Predicted values.
    returns : tf.Tensor
        Target returns.
    action_probs : tf.Tensor, optional
        Action probabilities (for entropy regularization), by default ``None``.
    value_coef : float, optional
        Value loss coefficient, by default 0.5.
    entropy_coef : float, optional
        Entropy coefficient, by default 0.01.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
        Tuple of (total loss, policy loss, value loss, entropy loss).
    """
    # ##: Calculate advantages.
    advantages = tf.subtract(returns, values)

    # ##: Policy gradient loss and value function loss.
    policy_loss = compute_policy_gradient_loss(action_log_probs, advantages)
    value_loss = compute_value_loss(values, returns, value_coef)

    # ##: Entropy regularization loss (optional).
    entropy_loss = tf.constant(0.0, dtype=tf.float32)
    if action_probs is not None:
        entropy_loss = compute_entropy_loss(action_probs, entropy_coef)

    # ##: Combined loss.
    total_loss = tf.add_n([policy_loss, value_loss, entropy_loss])

    # ##: Ensure all outputs are Tensors and satisfy type checker.
    return cast(
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        (
            tf.identity(total_loss, name="total_loss"),
            tf.identity(policy_loss, name="policy_loss"),
            tf.identity(value_loss, name="value_loss"),
            tf.identity(entropy_loss, name="entropy_loss"),
        ),
    )
