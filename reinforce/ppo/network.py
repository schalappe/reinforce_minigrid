# -*- coding: utf-8 -*-
"""
Neural network architectures for the Proximal Policy Optimization (PPO) agent.

This module defines the function `build_actor_critic` which constructs the actor and critic neural networks
required by the PPO algorithm. These networks typically share a common feature extraction base.
"""

from typing import Tuple

import tensorflow as tf
from keras import Input, Model, layers


def build_actor_critic(
    observation_shape: Tuple[int, ...],
    num_actions: int,
    conv_filters: int = 16,
    conv_kernel_size: int = 3,
    dense_units: int = 64,
) -> Tuple[Model, Model]:
    """
    Builds the Actor and Critic networks using a shared CNN base.

    The architecture consists of a shared convolutional base for feature extraction from the observation,
    followed by separate dense heads for the actor (policy) and the critic (value function).

    Parameters
    ----------
    observation_shape : Tuple[int, ...]
        The shape of the environment observation space (e.g., height, width, channels).
    num_actions : int
        The number of discrete actions available in the environment.
    conv_filters : int, optional
        Number of filters in the first convolutional layer.
        The second layer will have twice this number. Defaults to 16.
    conv_kernel_size : int, optional
        Kernel size for the convolutional layers. Defaults to 3.
    dense_units : int, optional
        Number of units in the shared dense layer before the heads. Defaults to 64.

    Returns
    -------
    Tuple[keras.Model, keras.Model]
        A tuple containing the compiled Keras models for the Actor and Critic.
        - Actor: Outputs logits for each action.
        - Critic: Outputs a single value representing the state value estimate.
    """
    inputs = Input(shape=observation_shape, dtype=tf.float32, name="observation_input")

    # ##: Normalize pixel values to [0, 1].
    normalized_inputs = layers.Rescaling(1.0 / 255.0)(inputs)

    # ##: Shared Convolutional Base.
    conv = layers.Conv2D(
        filters=conv_filters, kernel_size=conv_kernel_size, activation="relu", padding="same", name="conv1"
    )(normalized_inputs)
    conv = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(conv)
    conv = layers.Conv2D(
        filters=conv_filters * 2, kernel_size=conv_kernel_size, activation="relu", padding="same", name="conv2"
    )(conv)
    conv = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(conv)
    flattened = layers.Flatten(name="flatten")(conv)

    # ##: Shared Dense Layer.
    shared_dense = layers.Dense(units=dense_units, activation="tanh", name="shared_dense")(flattened)

    # ##: Actor Head (outputs action logits).
    actor_logits = layers.Dense(units=num_actions, name="actor_logits")(shared_dense)
    actor = Model(inputs=inputs, outputs=actor_logits, name="actor")

    # ##: Critic Head (outputs state value).
    critic_value = layers.Dense(units=1, name="critic_value")(shared_dense)
    critic = Model(inputs=inputs, outputs=critic_value, name="critic")

    return actor, critic
