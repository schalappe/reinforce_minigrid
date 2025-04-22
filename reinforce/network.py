# -*- coding: utf-8 -*-
"""
Neural network architectures for the PPO agent using TensorFlow/Keras.
"""

import tensorflow as tf

# Use tf.keras.layers directly for consistency
from tensorflow.keras import Model, layers


def build_policy_network(input_shape: tuple, num_actions: int) -> tf.keras.Model:
    """
    Builds the policy network (actor) using a CNN architecture.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input observations (e.g., (height, width, channels)).
    num_actions : int
        The number of possible discrete actions.

    Returns
    -------
    tf.keras.Model
        The compiled Keras model for the policy network.
    """
    inputs = layers.Input(shape=input_shape)

    # Use Rescaling layer for normalization (0-255 -> 0-1)
    # Input needs to be float32 for Rescaling
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # CNN layers for grid processing
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    # Flatten and FC layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    # Policy output (logits for discrete action space)
    action_logits = layers.Dense(num_actions, name="action_logits")(x)

    return Model(inputs=inputs, outputs=action_logits, name="PolicyNetwork")


def build_value_network(input_shape: tuple) -> tf.keras.Model:
    """
    Builds the value network (critic) using a CNN architecture.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input observations (e.g., (height, width, channels)).

    Returns
    -------
    tf.keras.Model
        The compiled Keras model for the value network.
    """
    inputs = layers.Input(shape=input_shape)

    # Use Rescaling layer for normalization (0-255 -> 0-1)
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Same CNN architecture as policy network
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    # Flatten and FC layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    # Value output (single node)
    value = layers.Dense(1, name="value_output")(x)

    return Model(inputs=inputs, outputs=value, name="ValueNetwork")
