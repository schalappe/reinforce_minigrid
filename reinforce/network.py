# -*- coding: utf-8 -*-
"""
Neural network architectures for the PPO agent using Keras 3.
"""

import keras
from keras import Model, layers


def build_actor_critic_networks(input_shape: tuple, num_actions: int) -> tuple[keras.Model, keras.Model]:
    """
    Builds the shared-base actor-critic networks using a CNN-LSTM architecture.

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

    # ##: Shared CNN Base.
    x = layers.Conv2D(32, 3, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("swish", name="act1")(x)

    x = layers.Conv2D(64, 3, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("swish", name="act2")(x)

    conv_out_shape = x.shape[1:]
    features_dim = conv_out_shape[0] * conv_out_shape[1] * conv_out_shape[2]
    x = layers.Reshape((-1, features_dim), name="reshape_for_lstm")(x)

    # ##: LSTM Layer for temporal memory.
    lstm_out = layers.LSTM(128, name="lstm_memory")(x)

    # ##: Policy Head (Actor).
    policy_x = layers.Dense(256, activation="swish", name="policy_dense1")(lstm_out)
    policy_x = layers.Dropout(0.2, name="policy_dropout")(policy_x)
    action_logits = layers.Dense(num_actions, name="action_logits")(policy_x)
    policy_network = Model(inputs=inputs, outputs=action_logits, name="PolicyNetwork")

    # ##: Value Head (Critic).
    value_x = layers.Dense(128, activation="swish", name="value_dense1")(lstm_out)
    value_x = layers.Dropout(0.1, name="value_dropout")(value_x)
    value_output = layers.Dense(1, name="value_output")(value_x)
    value_network = Model(inputs=inputs, outputs=value_output, name="ValueNetwork")

    return policy_network, value_network
