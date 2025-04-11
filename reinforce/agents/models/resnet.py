# -*- coding: utf-8 -*-
"""
Actor-Critic model with a ResNet-style convolutional base.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf
from keras import Model, Sequential, layers


def residual_block(x: tf.Tensor, filters: int, kernel_size: int = 3, strides: int = 1) -> tf.Tensor:
    """Creates a simple residual block."""
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, kernel_size=kernel_size, padding="same")(y)
    y = layers.BatchNormalization()(y)

    # ##: Add shortcut connection.
    if strides > 1 or x.shape[-1] != filters:
        x = layers.Conv2D(filters, kernel_size=1, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)

    out = layers.Add()([x, y])
    out = layers.Activation("relu")(out)
    return out


class ResNetA2CModel(Model):
    """
    Actor-Critic model using residual blocks in the convolutional base.
    """

    def __init__(self, action_space: int, embedding_size: int = 256, shape: Tuple[int, int, int] = (176, 176, 3)):
        """
        Initialize the ResNet Actor-Critic model.

        Parameters
        ----------
        action_space : int
            Number of possible actions.
        embedding_size : int, optional
            Size of the shared dense layer after convolutions, by default 256.
        shape : Tuple[int, int, int], optional
            Shape of the input image, by default (176, 176, 3).
        """
        super().__init__()
        self.action_space = action_space
        self.embedding_size = embedding_size

        input_layer = layers.Input(shape=shape)

        # ##: Initial Convolution.
        x = layers.Conv2D(32, kernel_size=3, strides=1, padding="same")(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)

        # ##: Residual Blocks.
        x = residual_block(x, filters=64, strides=1)
        x = residual_block(x, filters=64, strides=1)
        x = residual_block(x, filters=128, strides=2)

        # ##: Final Layers of Base.
        x = layers.Flatten()(x)
        output_layer = layers.Dense(self.embedding_size, activation="relu")(x)

        # ##: Create the convolutional base Model.
        self.convolutional_base = Model(inputs=input_layer, outputs=output_layer, name="resnet_base")

        # ##: Actor head for policy.
        self.actor = Sequential([layers.Dense(128, activation="relu"), layers.Dense(action_space)], name="actor_head")

        # ##: Critic head for value function.
        self.critic = Sequential([layers.Dense(128, activation="relu"), layers.Dense(1)], name="critic_head")

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        inputs : tf.Tensor
            Input observations.
        training : bool, optional
            Whether in training mode, by default ``None``.
        mask : tf.Tensor, optional
            Optional mask tensor, by default ``None``.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Action logits and value estimates.
        """
        shared_features = self.convolutional_base(inputs, training=training)
        action_logits = self.actor(shared_features, training=training)
        value_estimates = self.critic(shared_features, training=training)
        return action_logits, value_estimates

    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.

        Returns
        -------
        Dict[str, Any]
            Model configuration dictionary.
        """
        return {"action_space": self.action_space, "embedding_size": self.embedding_size}

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects=None) -> ResNetA2CModel:
        """
        Create a model from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Model configuration dictionary.
        custom_objects : dict, optional
            Optional dictionary mapping names to custom classes or functions, by default ``None``.

        Returns
        -------
        ResNetA2CModel
            Instantiated model.
        """
        return cls(**config)
