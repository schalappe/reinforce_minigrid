# -*- coding: utf-8 -*-
"""
Base loss functions for reinforcement learning algorithms.
"""

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
    error: tf.Tensor = tf.subtract(y_true, y_pred)
    is_small_error: tf.Tensor = tf.less_equal(tf.abs(error), delta)

    squared_loss: tf.Tensor = tf.multiply(0.5, tf.square(error))
    linear_loss: tf.Tensor = tf.multiply(delta, tf.subtract(tf.abs(error), tf.multiply(0.5, delta)))

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
    return tf.negative(tf.reduce_mean(tf.multiply(action_log_probs, advantages)))


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
    return tf.multiply(tf.cast(value_coef, dtype=tf.float32), losses.mean_squared_error(returns, values))


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
    entropy: tf.Tensor = tf.negative(
        tf.reduce_sum(tf.multiply(action_probs, tf.math.log(tf.add(action_probs, 1e-10))), axis=-1)
    )
    return tf.negative(tf.multiply(tf.cast(entropy_coef, dtype=tf.float32), tf.reduce_mean(entropy)))
