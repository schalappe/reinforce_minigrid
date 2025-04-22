# -*- coding: utf-8 -*-
"""Core PPO Algorithm Implementation using TensorFlow.

This module provides the core components for the Proximal Policy Optimization (PPO) algorithm, specificall
tailored for use with TensorFlow. It includes the definitions for the PPO clipped surrogate objective loss
the value function loss (Mean Squared Error), and the entropy bonus calculation. Additionally, it contains
the main `train_step` function which orchestrates a single update iteration for both the policy (actor)
and value (critic) networks.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from keras import Model, optimizers


def get_action_distribution(logits: tf.Tensor) -> tfp.distributions.Categorical:
    """
    Creates a categorical distribution from action logits.

    Parameters
    ----------
    logits : tf.Tensor
        The output logits from the policy network.

    Returns
    -------
    tfp.distributions.Categorical
        A TensorFlow Probability categorical distribution.
    """
    return tfp.distributions.Categorical(logits=logits)


@tf.function
def ppo_policy_loss(
    action_probs: tf.Tensor, old_action_probs: tf.Tensor, advantages: tf.Tensor, clip_param: float = 0.2
) -> tf.Tensor:
    """
    Calculates the PPO clipped surrogate objective loss.

    Parameters
    ----------
    action_probs : tf.Tensor
        Log probabilities of actions taken under the current policy.
    old_action_probs : tf.Tensor
        Log probabilities of actions taken under the policy used for data collection.
    advantages : tf.Tensor
        Estimated advantages for the state-action pairs.
    clip_param : float, optional
        The clipping parameter epsilon. Default is 0.2.

    Returns
    -------
    tf.Tensor
        The calculated policy loss.
    """
    # ##: Calculate the probability ratio: pi_theta(a|s) / pi_theta_old(a|s)
    # Using log probabilities: exp(log_prob - old_log_prob)
    ratio = tf.exp(action_probs - old_action_probs)

    # ##: Clip the ratio.
    clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param)

    # ##: Calculate the surrogate objectives.
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages

    # ##: The PPO loss is the negative minimum of the two surrogates, averaged over the batch.
    policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
    return policy_loss


@tf.function
def value_function_loss(values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """
    Calculates the value function loss (Mean Squared Error).

    Parameters
    ----------
    values : tf.Tensor
        Predicted state values from the critic network.
    returns : tf.Tensor
        Target returns calculated (e.g., using GAE).

    Returns
    -------
    tf.Tensor
        The calculated value function loss.
    """
    values = tf.squeeze(values, axis=-1) if len(values.shape) > 1 and values.shape[-1] == 1 else values
    return tf.reduce_mean(tf.square(returns - values))


@tf.function
def entropy_bonus(dist: tfp.distributions.Distribution) -> tf.Tensor:
    """
    Calculates the entropy bonus for exploration.

    Parameters
    ----------
    dist : tfp.distributions.Distribution
        The action distribution from the policy network.

    Returns
    -------
    tf.Tensor
        The mean entropy of the distribution (negative loss).
    """
    return tf.reduce_mean(dist.entropy())


@tf.function
def train_step(
    states: tf.Tensor,
    actions: tf.Tensor,
    old_action_probs: tf.Tensor,
    returns: tf.Tensor,
    advantages: tf.Tensor,
    policy_network: Model,
    value_network: Model,
    policy_optimizer: optimizers.Optimizer,
    value_optimizer: optimizers.Optimizer,
    clip_param: float,
    vf_coef: float,
    entropy_coef: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Performs a single training step for both policy and value networks.

    Parameters
    ----------
    states : tf.Tensor
        Batch of states.
    actions : tf.Tensor
        Batch of actions taken.
    old_action_probs : tf.Tensor
        Batch of log probabilities of actions under the old policy.
    returns : tf.Tensor
        Batch of calculated returns (targets for value function).
    advantages : tf.Tensor
        Batch of calculated advantages.
    policy_network : tf.keras.Model
        The policy network (actor).
    value_network : tf.keras.Model
        The value network (critic).
    policy_optimizer : tf.keras.optimizers.Optimizer
        Optimizer for the policy network.
    value_optimizer : tf.keras.optimizers.Optimizer
        Optimizer for the value network.
    clip_param : float
        PPO clipping parameter epsilon.
    vf_coef : float
        Coefficient for the value function loss in the total loss calculation.
    entropy_coef : float
        Coefficient for the entropy bonus in the total loss calculation.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        A tuple containing the policy loss, value loss, and entropy bonus for the step.
    """
    with tf.GradientTape() as policy_tape:
        # ##: Forward pass to get current action distribution and log probabilities.
        action_logits = policy_network(states, training=True)
        dist = get_action_distribution(action_logits)
        current_action_probs = dist.log_prob(actions)

        # ##: Calculate PPO policy loss.
        pi_loss = ppo_policy_loss(current_action_probs, old_action_probs, advantages, clip_param)

        # ##: Calculate entropy bonus (we want to maximize entropy, so minimize negative entropy).
        ent_bonus = entropy_bonus(dist)

        # ##: Total policy loss (including entropy bonus).
        total_policy_loss = pi_loss - entropy_coef * ent_bonus

    policy_grads = policy_tape.gradient(total_policy_loss, policy_network.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, policy_network.trainable_variables))

    with tf.GradientTape() as value_tape:
        # ##: Forward pass to get current value predictions.
        values = value_network(states, training=True)

        # ##: Calculate value function loss (MSE).
        v_loss = value_function_loss(values, returns)

        # ##: Total value loss (scaled by coefficient).
        total_value_loss = v_loss * vf_coef

    value_grads = value_tape.gradient(total_value_loss, value_network.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, value_network.trainable_variables))

    return pi_loss, v_loss, ent_bonus
