# -*- coding: utf-8 -*-
"""Neural network architectures for PPO."""

from typing import Tuple

from keras import Input, Model, layers


def build_actor_critic(
    observation_shape: Tuple[int, int, int], num_actions: int, conv_filters=16, conv_kernel_size=3, dense_units=64
) -> Tuple[Model, Model]:
    """
    Builds the Actor and Critic networks using a shared CNN base.

    Parameters
    ----------
    observation_shape : Tuple[int, int, int]
        Shape of the environment observation (height, width, channels).
    num_actions : int
       Number of possible actions in the environment.
    conv_filters : int, optional
        Number of filters in the convolutional layers, by default 16.
    conv_kernel_size : int, optional
        Kernel size for the convolutional layers, by default 3.
    dense_units : int, optional
        Number of units in the dense layers, by default 64.

    Returns
    -------
    Tuple[keras.Model, keras.Model]
        The Actor and Critic Keras models.
    """
    inputs = Input(shape=observation_shape, dtype="float32", name="observation_input")

    # ##: Shared CNN base.
    x = layers.Lambda(lambda layer: layer / 255.0)(inputs)

    # ##: Convolutional layers.
    conv = layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, activation="relu", padding="same")(x)
    conv = layers.MaxPooling2D(pool_size=(2, 2))(conv)
    conv = layers.Conv2D(filters=conv_filters * 2, kernel_size=conv_kernel_size, activation="relu", padding="same")(
        conv
    )
    conv = layers.MaxPooling2D(pool_size=(2, 2))(conv)
    conv = layers.Flatten()(conv)

    # ##: Shared Dense layer.
    shared_dense = layers.Dense(units=dense_units, activation="tanh", name="shared_dense")(conv)

    # ##: Actor head (outputs action probabilities/logits).
    actor_logits = layers.Dense(units=num_actions, name="actor_logits")(shared_dense)
    actor = Model(inputs=inputs, outputs=actor_logits, name="actor")

    # ##: Critic head (outputs state value).
    critic_value = layers.Dense(units=1, name="critic_value")(shared_dense)
    critic = Model(inputs=inputs, outputs=critic_value, name="critic")

    return actor, critic
