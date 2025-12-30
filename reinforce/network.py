"""
Neural network architectures for the PPO agent using Keras 3.

Implements RL best practices:
- Orthogonal weight initialization (per "37 Implementation Details of PPO")
- IMPALA-style residual CNN blocks for better feature extraction
- Separate policy/value heads with appropriate scaling
"""

import keras
from keras import Model, initializers, layers


def _get_orthogonal_initializer(scale: float = 1.0) -> initializers.Orthogonal:
    """
    Creates an orthogonal initializer with the specified gain.

    Per the "37 Implementation Details of PPO" paper:
    - Hidden layers use sqrt(2) gain
    - Policy output uses 0.01 gain
    - Value output uses 1.0 gain

    Parameters
    ----------
    scale : float
        The gain/scale for the orthogonal initialization.

    Returns
    -------
    initializers.Orthogonal
        Keras orthogonal initializer with specified gain.
    """
    return initializers.Orthogonal(gain=scale)


def _residual_block(x: layers.Layer, filters: int, name_prefix: str) -> layers.Layer:
    """
    Creates an IMPALA-style residual block.

    IMPALA (Importance Weighted Actor-Learner Architecture) introduced residual blocks that significantly
    improve feature extraction for visual RL tasks.

    Parameters
    ----------
    x : layers.Layer
        Input tensor.
    filters : int
        Number of convolutional filters.
    name_prefix : str
        Prefix for layer names.

    Returns
    -------
    layers.Layer
        Output tensor after residual connection.
    """
    # ##>: Residual connection preserves gradient flow through deep networks.
    residual = x

    out = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
    out = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer=_get_orthogonal_initializer(scale=2**0.5),
        bias_initializer="zeros",
        name=f"{name_prefix}_conv1",
    )(out)

    out = layers.Activation("relu", name=f"{name_prefix}_relu2")(out)
    out = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer=_get_orthogonal_initializer(scale=2**0.5),
        bias_initializer="zeros",
        name=f"{name_prefix}_conv2",
    )(out)

    return layers.Add(name=f"{name_prefix}_add")([residual, out])


def _impala_cnn_block(x: layers.Layer, filters: int, name_prefix: str) -> layers.Layer:
    """
    Creates an IMPALA CNN block with downsampling and residual connections.

    Parameters
    ----------
    x : layers.Layer
        Input tensor.
    filters : int
        Number of convolutional filters.
    name_prefix : str
        Prefix for layer names.

    Returns
    -------
    layers.Layer
        Output tensor.
    """
    # ##>: Initial convolution for channel adjustment.
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer=_get_orthogonal_initializer(scale=2**0.5),
        bias_initializer="zeros",
        name=f"{name_prefix}_conv_init",
    )(x)

    # ##>: Max pooling for spatial downsampling.
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name=f"{name_prefix}_pool")(x)

    # ##>: Two residual blocks per IMPALA block.
    x = _residual_block(x, filters, f"{name_prefix}_res1")
    x = _residual_block(x, filters, f"{name_prefix}_res2")

    return x


def build_actor_critic_networks(input_shape: tuple, num_actions: int) -> tuple[keras.Model, keras.Model]:
    """
    Builds the actor-critic networks using IMPALA-style CNN architecture.

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

    # ##>: Normalize pixel values to [0, 1] range for stable training.
    x = layers.Rescaling(scale=1.0 / 255.0, name="pixel_normalize")(inputs)

    # ##>: IMPALA-style CNN with 3 blocks of increasing filter count.
    # This architecture was shown to outperform Nature CNN on visual RL tasks.
    x = _impala_cnn_block(x, filters=16, name_prefix="impala_block1")
    x = _impala_cnn_block(x, filters=32, name_prefix="impala_block2")
    x = _impala_cnn_block(x, filters=32, name_prefix="impala_block3")

    # ##>: Final activation before flattening.
    x = layers.Activation("relu", name="final_relu")(x)
    x = layers.Flatten(name="flatten")(x)

    # ##>: Shared dense layer for feature extraction.
    shared_features = layers.Dense(
        256,
        activation="relu",
        kernel_initializer=_get_orthogonal_initializer(scale=2**0.5),
        bias_initializer="zeros",
        name="shared_dense",
    )(x)

    # ##>: Policy Head (Actor) - 0.01 scale for small initial policy.
    action_logits = layers.Dense(
        num_actions,
        kernel_initializer=_get_orthogonal_initializer(scale=0.01),
        bias_initializer="zeros",
        name="action_logits",
    )(shared_features)
    policy_network = Model(inputs=inputs, outputs=action_logits, name="PolicyNetwork")

    # ##>: Value Head (Critic) - 1.0 scale for value output.
    value_output = layers.Dense(
        1,
        kernel_initializer=_get_orthogonal_initializer(scale=1.0),
        bias_initializer="zeros",
        name="value_output",
    )(shared_features)
    value_network = Model(inputs=inputs, outputs=value_output, name="ValueNetwork")

    return policy_network, value_network
