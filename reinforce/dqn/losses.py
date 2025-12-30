"""
Rainbow DQN Loss Functions.

Implements:
- Categorical DQN (C51) cross-entropy loss
- Double DQN target computation
- Multi-step TD targets
"""

import tensorflow as tf


def compute_target_distribution(
    rewards: tf.Tensor,
    next_q_dist: tf.Tensor,
    dones: tf.Tensor,
    gamma: float,
    n_step: int,
    v_min: float,
    v_max: float,
    num_atoms: int,
) -> tf.Tensor:
    """
    Compute projected target distribution for C51.

    Uses the Bellman update: T_z = r + gamma^n * z (for non-terminal).
    Then projects back onto the fixed support.

    Parameters
    ----------
    rewards : tf.Tensor
        N-step discounted rewards. Shape: (batch,).
    next_q_dist : tf.Tensor
        Next state Q distribution for best action. Shape: (batch, num_atoms).
    dones : tf.Tensor
        Terminal flags. Shape: (batch,).
    gamma : float
        Discount factor.
    n_step : int
        Number of steps for multi-step returns.
    v_min : float
        Minimum value support.
    v_max : float
        Maximum value support.
    num_atoms : int
        Number of distribution atoms.

    Returns
    -------
    tf.Tensor
        Projected target distribution. Shape: (batch, num_atoms).
    """
    batch_size = tf.shape(rewards)[0]
    delta_z = (v_max - v_min) / (num_atoms - 1)
    support = tf.linspace(v_min, v_max, num_atoms)

    # ##>: Tz = r + gamma^n * z (clamped to [v_min, v_max]).
    gamma_n = gamma**n_step
    rewards = tf.cast(rewards, tf.float32)
    dones = tf.cast(dones, tf.float32)

    Tz = tf.expand_dims(rewards, 1) + gamma_n * tf.expand_dims(1.0 - dones, 1) * tf.expand_dims(support, 0)
    Tz = tf.clip_by_value(Tz, v_min, v_max)

    # ##>: Projection: compute which atoms Tz falls between.
    b = (Tz - v_min) / delta_z
    lower_idx = tf.cast(tf.math.floor(b), tf.int32)
    upper_idx = tf.cast(tf.math.ceil(b), tf.int32)

    # ##>: Handle edge case where lower_idx == upper_idx.
    lower_idx = tf.clip_by_value(lower_idx, 0, num_atoms - 1)
    upper_idx = tf.clip_by_value(upper_idx, 0, num_atoms - 1)

    # ##>: Distribute probability mass.
    target_dist = tf.zeros((batch_size, num_atoms), dtype=tf.float32)

    # ##>: Use scatter_nd to distribute probabilities.
    batch_indices = tf.repeat(tf.range(batch_size), num_atoms)
    batch_indices = tf.reshape(batch_indices, (batch_size, num_atoms))

    lower_flat = tf.reshape(lower_idx, [-1])
    upper_flat = tf.reshape(upper_idx, [-1])
    batch_flat = tf.reshape(batch_indices, [-1])

    # ##>: Weight by distance from b to lower and upper.
    b_float = tf.cast(b, tf.float32)
    lower_float = tf.cast(lower_idx, tf.float32)
    upper_float = tf.cast(upper_idx, tf.float32)

    # ##>: Lower weight: probability * (upper - b).
    lower_weight = next_q_dist * (upper_float - b_float)
    # ##>: Upper weight: probability * (b - lower).
    upper_weight = next_q_dist * (b_float - lower_float)

    lower_weight_flat = tf.reshape(lower_weight, [-1])
    upper_weight_flat = tf.reshape(upper_weight, [-1])

    # ##>: Create indices for scatter.
    lower_indices = tf.stack([batch_flat, lower_flat], axis=1)
    upper_indices = tf.stack([batch_flat, upper_flat], axis=1)

    # ##>: Scatter add the weights.
    target_dist = tf.tensor_scatter_nd_add(target_dist, lower_indices, lower_weight_flat)
    target_dist = tf.tensor_scatter_nd_add(target_dist, upper_indices, upper_weight_flat)

    return target_dist


def categorical_dqn_loss(
    q_dist: tf.Tensor,
    target_dist: tf.Tensor,
    actions: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute weighted cross-entropy loss for C51.

    Parameters
    ----------
    q_dist : tf.Tensor
        Current Q distribution. Shape: (batch, num_actions, num_atoms).
    target_dist : tf.Tensor
        Target distribution. Shape: (batch, num_atoms).
    actions : tf.Tensor
        Actions taken. Shape: (batch,).
    weights : tf.Tensor
        Importance sampling weights. Shape: (batch,).

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        (loss, td_errors) for priority updates.
    """
    batch_size = tf.shape(actions)[0]
    actions = tf.cast(actions, tf.int32)

    # ##>: Select distribution for taken action.
    indices = tf.stack([tf.range(batch_size), actions], axis=1)
    q_dist_action = tf.gather_nd(q_dist, indices)

    # ##>: Cross-entropy loss: -sum(target * log(predicted)).
    cross_entropy = -tf.reduce_sum(target_dist * tf.math.log(q_dist_action + 1e-8), axis=1)

    # ##>: TD error for priorities (use cross-entropy as proxy).
    td_errors = cross_entropy

    # ##>: Weighted loss.
    loss = tf.reduce_mean(weights * cross_entropy)

    return loss, td_errors
