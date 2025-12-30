"""
Neural network architectures for the PPO agent using Keras 3.

Implements RL best practices:
- Orthogonal weight initialization (per "37 Implementation Details of PPO")
- IMPALA-style residual CNN blocks for better feature extraction
- Separate policy/value heads with appropriate scaling
"""

import keras
from keras import Model, layers

from reinforce.core.network_utils import build_impala_cnn, get_orthogonal_initializer


def build_actor_critic_networks(input_shape: tuple, num_actions: int) -> tuple[keras.Model, keras.Model]:
    """
    Build the actor-critic networks using IMPALA-style CNN architecture.

    This architecture follows modern RL best practices:
    - IMPALA residual blocks for better visual feature extraction
    - Orthogonal initialization for stable training
    - Appropriate output scaling (0.01 for policy, 1.0 for value)
    - No dropout (shown to hurt performance in on-policy RL)

    Parameters
    ----------
    input_shape : tuple
        The shape of the input observations (e.g., (height, width, channels)).
    num_actions : int
        The number of possible discrete actions.

    Returns
    -------
    tuple[keras.Model, keras.Model]
        A tuple containing the policy network (actor) and value network (critic).
    """
    inputs = layers.Input(shape=input_shape, name="observation_input")

    # ##>: Use shared IMPALA CNN from core utilities.
    shared_features = build_impala_cnn(inputs, name_prefix="ppo")

    # ##>: Shared dense layer for feature extraction.
    shared_features = layers.Dense(
        256,
        activation="relu",
        kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
        bias_initializer="zeros",
        name="shared_dense",
    )(shared_features)

    # ##>: Policy Head (Actor) - 0.01 scale for small initial policy.
    action_logits = layers.Dense(
        num_actions,
        kernel_initializer=get_orthogonal_initializer(scale=0.01),
        bias_initializer="zeros",
        name="action_logits",
    )(shared_features)
    policy_network = Model(inputs=inputs, outputs=action_logits, name="PolicyNetwork")

    # ##>: Value Head (Critic) - 1.0 scale for value output.
    value_output = layers.Dense(
        1,
        kernel_initializer=get_orthogonal_initializer(scale=1.0),
        bias_initializer="zeros",
        name="value_output",
    )(shared_features)
    value_network = Model(inputs=inputs, outputs=value_output, name="ValueNetwork")

    return policy_network, value_network
