# -*- coding: utf-8 -*-
"""Set of function for replay experience."""
import numpy as np
import tensorflow as tf

# ##: Small epsilon value for stabilizing division operations
EPS = np.finfo(np.float32).eps.item()


def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
    """
    Compute expected returns per timestep.

    Parameters
    ----------
    rewards: tf.Tensor
        Rewards
    gamma: tf.Tensor
        Gamma value
    standardize: bool
        Standardize or not

    Returns
    -------
    tf.Tensor
        Expected return
    """
    size = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=size)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(size):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + EPS)

    return returns
