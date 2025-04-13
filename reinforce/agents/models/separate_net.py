# -*- coding: utf-8 -*-
"""
Actor-Critic model with separate ResNet-style convolutional bases for actor and critic.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf
from keras import Model, Sequential, layers

from .resnet import residual_block


class SeparateNetACModel(Model):
    """
    Actor-Critic model using separate residual block networks for actor and critic.
    """

    def __init__(self, action_space: int, embedding_size: int = 256, shape: Tuple[int, int, int] = (176, 176, 3)):
        """
        Initialize the SeparateNet Actor-Critic model.

        Parameters
        ----------
        action_space : int
            Number of possible actions.
        embedding_size : int, optional
            Size of the dense layer after convolutions for each head, by default 256.
        shape : Tuple[int, int, int], optional
            Shape of the input image, by default (176, 176, 3).
        """
        super().__init__()
        self.action_space = action_space
        self.embedding_size = embedding_size
        self.input_shape_spec = shape

        # ##: Define Input Layer.
        input_layer = layers.Input(shape=shape)

        # ##: Actor Convolutional Base.
        x_actor = layers.Conv2D(32, kernel_size=3, strides=1, padding="same", name="actor_conv1")(input_layer)
        x_actor = layers.BatchNormalization(name="actor_bn1")(x_actor)
        x_actor = layers.Activation("relu", name="actor_relu1")(x_actor)
        x_actor = layers.MaxPool2D(pool_size=(2, 2), name="actor_pool1")(x_actor)
        x_actor = residual_block(x_actor, filters=64, strides=1)
        x_actor = residual_block(x_actor, filters=128, strides=2)
        x_actor = layers.Flatten(name="actor_flatten")(x_actor)
        actor_features = layers.Dense(self.embedding_size, activation="relu", name="actor_dense_features")(x_actor)
        self.actor_base = Model(inputs=input_layer, outputs=actor_features, name="actor_base")

        # ##: Critic Convolutional Base.
        x_critic = layers.Conv2D(32, kernel_size=3, strides=1, padding="same", name="critic_conv1")(input_layer)
        x_critic = layers.BatchNormalization(name="critic_bn1")(x_critic)
        x_critic = layers.Activation("relu", name="critic_relu1")(x_critic)
        x_critic = layers.MaxPool2D(pool_size=(2, 2), name="critic_pool1")(x_critic)
        x_critic = residual_block(x_critic, filters=64, strides=1)
        x_critic = residual_block(x_critic, filters=128, strides=2)
        x_critic = layers.Flatten(name="critic_flatten")(x_critic)
        critic_features = layers.Dense(self.embedding_size, activation="relu", name="critic_dense_features")(x_critic)
        self.critic_base = Model(inputs=input_layer, outputs=critic_features, name="critic_base")

        # ##: Actor head for policy.
        self.actor_head = Sequential(
            [layers.Dense(128, activation="relu"), layers.Dense(128, activation="relu"), layers.Dense(action_space)],
            name="actor_head",
        )

        # ##: Critic head for value function.
        self.critic_head = Sequential(
            [layers.Dense(128, activation="relu"), layers.Dense(128, activation="relu"), layers.Dense(1)],
            name="critic_head",
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass of the model using separate networks.

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
        actor_features = self.actor_base(inputs, training=training)
        critic_features = self.critic_base(inputs, training=training)

        action_logits = self.actor_head(actor_features, training=training)
        value_estimates = self.critic_head(critic_features, training=training)

        return action_logits, value_estimates

    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.

        Returns
        -------
        Dict[str, Any]
            Model configuration dictionary.
        """
        return {
            "action_space": self.action_space,
            "embedding_size": self.embedding_size,
            "shape": self.input_shape_spec,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects=None) -> SeparateNetACModel:
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
        SeparateNetACModel
            Instantiated model.
        """
        # ##: Ensure residual_block is available if needed during deserialization
        custom_objects = custom_objects or {}
        custom_objects["residual_block"] = residual_block
        return cls(**config)
