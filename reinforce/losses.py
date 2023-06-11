# -*- coding: utf-8 -*-
"""Loss function."""
import tensorflow as tf

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """
    Computes the combined Actor-Critic loss.

    Parameters
    ----------
    action_probs: tf.Tensor
        Action probabilities
    values: tf.Tensor
        Values of state
    returns: tf.Tensor
        Expected return

    Returns
    -------
    tf.Tensor
        Computed loss
    """

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss
