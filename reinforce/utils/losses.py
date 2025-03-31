# -*- coding: utf-8 -*-
"""
Loss functions for reinforcement learning algorithms.
"""

from typing import Optional, Tuple

import tensorflow as tf
from keras import losses


def huber_loss(y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 1.0) -> tf.Tensor:
    """
    Huber loss function.

    The Huber loss is a combination of mean squared error and mean absolute error.
    It is less sensitive to outliers than MSE.

    Parameters
    ----------
    y_true : tf.Tensor
        Target values.
    y_pred : tf.Tensor
        Predicted values.
    delta : float, optional
        Threshold parameter, by default 1.0.

    Returns
    -------
    tf.Tensor
        Huber loss.
    """
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta

    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)

    return tf.where(is_small_error, squared_loss, linear_loss)


def compute_policy_gradient_loss(action_log_probs: tf.Tensor, advantages: tf.Tensor) -> tf.Tensor:
    """
    Compute policy gradient loss.

    Parameters
    ----------
    action_log_probs : tf.Tensor
        Log probabilities of actions taken.
    advantages : tf.Tensor
        Advantage estimates.

    Returns
    -------
    tf.Tensor
        Policy gradient loss.
    """
    return -tf.reduce_mean(action_log_probs * advantages)


def compute_value_loss(values: tf.Tensor, returns: tf.Tensor, value_coef: float = 0.5) -> tf.Tensor:
    """
    Compute value function loss.

    Parameters
    ----------
    values : tf.Tensor
        Predicted values.
    returns : tf.Tensor
        Target returns.
    value_coef : float, optional
        Value loss coefficient, by default 0.5.

    Returns
    -------
    tf.Tensor
        Value function loss.
    """
    return value_coef * losses.mean_squared_error(returns, values)


def compute_entropy_loss(action_probs: tf.Tensor, entropy_coef: float = 0.01) -> tf.Tensor:
    """
    Compute entropy regularization loss.

    Parameters
    ----------
    action_probs : tf.Tensor
        Action probabilities.
    entropy_coef : float, optional
        Entropy coefficient, by default 0.01.

    Returns
    -------
    tf.Tensor
        Entropy regularization loss.
    """
    entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=-1)
    return -entropy_coef * tf.reduce_mean(entropy)


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
    advantages = returns - values

    # ##: Policy gradient loss and value function loss.
    policy_loss = compute_policy_gradient_loss(action_log_probs, advantages)
    value_loss = compute_value_loss(values, returns, value_coef)

    # ##: Entropy regularization loss (optional)
    entropy_loss = tf.constant(0.0)
    if action_probs is not None:
        entropy_loss = compute_entropy_loss(action_probs, entropy_coef)

    # ##: Combined loss.
    total_loss = policy_loss + value_loss + entropy_loss

    return total_loss, policy_loss, value_loss, entropy_loss
