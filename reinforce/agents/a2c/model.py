# -*- coding: utf-8 -*-
"""
Actor-Critic neural network models.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf
from keras import Model, Sequential, layers

# ##: TODO: Split into separate files.


class A2CModel(Model):
    """
    Actor-Critic neural network model.

    This model takes in observations and outputs action probabilities (actor) and value estimates (critic).
    The actor and critic share convolutional feature extraction layers.
    """

    def __init__(self, action_space: int, embedding_size: int = 128):
        """
        Initialize the Actor-Critic model.

        Parameters
        ----------
        action_space : int
            Number of possible actions.
        embedding_size : int, optional
            Size of the embedding layer, by default 128.
        """
        super().__init__()

        # ##: Store parameters.
        self.action_space = action_space
        self.embedding_size = embedding_size

        # ##: Stack of convolutional layers for feature extraction.
        self.convolution_layer = Sequential(
            [
                layers.Conv2D(filters=16, kernel_size=3),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Conv2D(filters=32, kernel_size=3),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(filters=64, kernel_size=3),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(filters=128, kernel_size=3),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(embedding_size),
            ]
        )

        # ##: Actor model for policy.
        self.actor = Sequential(
            [
                layers.Dense(64),
                layers.Activation("relu"),
                layers.GlobalAveragePooling2D(),
                layers.Dense(action_space),
            ]
        )

        # ##: Critic model for value function.
        self.critic = Sequential(
            [
                layers.Dense(64),
                layers.Activation("relu"),
                layers.GlobalAveragePooling2D(),
                layers.Dense(1),
            ]
        )

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
            Tuple of (action_logits, value_estimates).
        """
        embedding = self.convolution_layer(inputs)
        return self.actor(embedding), self.critic(embedding)

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
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects=None) -> A2CModel:
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
        A2CModel
            Instantiated model.
        """
        return cls(**config)


class CNNEncoder(Model):
    """
    CNN-based encoder for preprocessing observations.

    This model processes raw observations with convolutional layers to extract meaningful features.
    """

    def __init__(self, output_dim: int = 256):
        """
        Initialize the CNN encoder.

        Parameters
        ----------
        output_dim : int, optional
            Output feature dimension, by default 256.
        """
        super().__init__()

        self.output_dim = output_dim
        self.cnn_layers = Sequential(
            [
                layers.Conv2D(32, 8, strides=4, activation="relu"),
                layers.Conv2D(64, 4, strides=2, activation="relu"),
                layers.Conv2D(64, 3, strides=1, activation="relu"),
                layers.Flatten(),
                layers.Dense(output_dim, activation="relu"),
            ]
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """
        Forward pass of the encoder.

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
        tf.Tensor
            Encoded features.
        """
        return self.cnn_layers(inputs)
