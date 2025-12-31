"""Core PPO Algorithm Implementation using TensorFlow.

This module provides the core components for the Proximal Policy Optimization (PPO) algorithm,
specifically tailored for use with TensorFlow. Implements best practices:
- Global gradient clipping (max_grad_norm=0.5)
- Value function clipping (optional, can hurt performance per ablation studies)
- Proper advantage normalization at mini-batch level
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
    # ##>: Calculate the probability ratio: pi_theta(a|s) / pi_theta_old(a|s)
    # Using log probabilities: exp(log_prob - old_log_prob)
    ratio = tf.exp(action_probs - old_action_probs)

    # ##>: Clip the ratio.
    clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param)

    # ##>: Calculate the surrogate objectives.
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages

    # ##>: The PPO loss is the negative minimum of the two surrogates, averaged over the batch.
    policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
    return policy_loss


def value_function_loss(
    values: tf.Tensor,
    returns: tf.Tensor,
    old_values: tf.Tensor | None = None,
    clip_param: float = 0.2,
    use_clipping: bool = False,
) -> tf.Tensor:
    """
    Calculates the value function loss with optional clipping.

    The clipped value loss prevents large value function updates, similar to
    how PPO clips the policy. However, ablation studies show this can sometimes
    hurt performance, so it's optional.

    Parameters
    ----------
    values : tf.Tensor
        Predicted state values from the critic network.
    returns : tf.Tensor
        Target returns calculated (e.g., using GAE).
    old_values : tf.Tensor, optional
        Value predictions from the old policy. Required if use_clipping=True.
    clip_param : float, optional
        Clipping parameter for value function. Default is 0.2.
    use_clipping : bool, optional
        Whether to use clipped value loss. Default is False.

    Returns
    -------
    tf.Tensor
        The calculated value function loss.
    """
    values = tf.squeeze(values, axis=-1) if len(values.shape) > 1 and values.shape[-1] == 1 else values

    if use_clipping and old_values is not None:
        # ##>: Clipped value loss (PPO v2 style).
        old_values = (
            tf.squeeze(old_values, axis=-1) if len(old_values.shape) > 1 and old_values.shape[-1] == 1 else old_values
        )
        values_clipped = old_values + tf.clip_by_value(values - old_values, -clip_param, clip_param)
        value_loss_unclipped = tf.square(returns - values)
        value_loss_clipped = tf.square(returns - values_clipped)
        return tf.reduce_mean(tf.maximum(value_loss_unclipped, value_loss_clipped))

    return tf.reduce_mean(tf.square(returns - values))


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


def _clip_gradients(gradients: list, max_grad_norm: float) -> list:
    """
    Clips gradients by global norm.

    Per the "37 Implementation Details of PPO", global gradient clipping
    with max_norm=0.5 improves training stability.

    Parameters
    ----------
    gradients : list
        List of gradient tensors.
    max_grad_norm : float
        Maximum gradient norm.

    Returns
    -------
    list
        Clipped gradient tensors.
    """
    clipped_grads, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
    return clipped_grads


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
    max_grad_norm: float = 0.5,
    old_values: tf.Tensor | None = None,
    use_value_clipping: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Performs a single training step for both policy and value networks.

    Implements modern PPO best practices:
    - Global gradient clipping (default max_norm=0.5)
    - Optional value function clipping
    - Proper mini-batch advantage normalization

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
    max_grad_norm : float, optional
        Maximum gradient norm for clipping. Default is 0.5.
    old_values : tf.Tensor, optional
        Old value predictions for value clipping. Default is None.
    use_value_clipping : bool, optional
        Whether to use value function clipping. Default is False.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
        A tuple containing the policy loss, value loss, entropy bonus,
        clip fraction, and approximate KL divergence for monitoring.
    """
    # ##>: Mini-batch advantage normalization (per "37 Implementation Details").
    advantages_normalized = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

    with tf.GradientTape() as policy_tape:
        # ##>: Forward pass to get current action distribution and log probabilities.
        action_logits = policy_network(states, training=True)
        dist = get_action_distribution(action_logits)
        current_action_probs = dist.log_prob(actions)

        # ##>: Calculate PPO policy loss.
        pi_loss = ppo_policy_loss(current_action_probs, old_action_probs, advantages_normalized, clip_param)

        # ##>: Calculate entropy bonus (we want to maximize entropy, so minimize negative entropy).
        ent_bonus = entropy_bonus(dist)

        # ##>: Total policy loss (including entropy bonus).
        total_policy_loss = pi_loss - entropy_coef * ent_bonus

    policy_grads = policy_tape.gradient(total_policy_loss, policy_network.trainable_variables)
    # ##>: Global gradient clipping for stability.
    policy_grads = _clip_gradients(policy_grads, max_grad_norm)
    policy_optimizer.apply_gradients(zip(policy_grads, policy_network.trainable_variables))

    with tf.GradientTape() as value_tape:
        # ##>: Forward pass to get current value predictions.
        values = value_network(states, training=True)

        # ##>: Calculate value function loss (with optional clipping).
        v_loss = value_function_loss(values, returns, old_values, clip_param, use_value_clipping)

        # ##>: Total value loss (scaled by coefficient).
        total_value_loss = v_loss * vf_coef

    value_grads = value_tape.gradient(total_value_loss, value_network.trainable_variables)
    # ##>: Global gradient clipping for stability.
    value_grads = _clip_gradients(value_grads, max_grad_norm)
    value_optimizer.apply_gradients(zip(value_grads, value_network.trainable_variables))

    # ##>: Calculate monitoring metrics.
    # Clip fraction: how often the ratio was clipped.
    ratio = tf.exp(current_action_probs - old_action_probs)
    clip_fraction = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > clip_param, tf.float32))

    # ##>: Approximate KL divergence for early stopping (if implemented).
    approx_kl = tf.reduce_mean(old_action_probs - current_action_probs)

    return pi_loss, v_loss, ent_bonus, clip_fraction, approx_kl
