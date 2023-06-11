# -*- coding: utf-8 -*-
"""Other function."""
import tensorflow as tf


def preprocess_input(inputs: tf.Tensor) -> tf.Tensor:
    """
    Preprocesses a tensor encoding a batch of images (Normalization).

    Parameters
    ----------
    inputs: tf.Tensor
        Input tensor, 3D or 4D.

    Returns
    -------
        Preprocessed tensor.
    """
    inputs = tf.cast(inputs, dtype=tf.float32)
    inputs /= 127.5
    inputs -= 1.0

    return inputs
