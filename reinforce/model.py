# -*- coding: utf-8 -*-
"""Tensorflow models for MiniGrid."""
from typing import Tuple

import tensorflow as tf
from tensorflow import keras


class A2CModel(keras.Model):
    """Actor-Critic model."""

    def __init__(self, action_space: int, embedding_size: int = 128):
        super().__init__()

        # ##: Stack of convolutional layer.
        self.convolution_layer = keras.Sequential(
            [
                keras.layers.Conv2D(filters=16, kernel_size=3),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.MaxPool2D(pool_size=(2, 2)),
                keras.layers.Conv2D(filters=32, kernel_size=3),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Conv2D(filters=64, kernel_size=3),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Conv2D(filters=128, kernel_size=3),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Dense(embedding_size),
            ]
        )

        # ##: Actor model.
        self.actor = keras.Sequential(
            [
                keras.layers.Dense(64),
                keras.layers.Activation("tanh"),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(action_space),
            ]
        )

        # ##: Critic model.
        self.critic = keras.Sequential(
            [
                keras.layers.Dense(64),
                keras.layers.Activation("tanh"),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(1),
            ]
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Inference."""
        embedding = self.convolution_layer(inputs)
        return self.actor(embedding), self.critic(embedding)

    def get_config(self):
        return {}
