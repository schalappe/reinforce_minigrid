# -*- coding: utf-8 -*-
"""
CNN-based encoder for preprocessing observations.
"""

from __future__ import annotations

import tensorflow as tf
from keras import Model, Sequential, layers


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
