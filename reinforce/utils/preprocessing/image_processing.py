# -*- coding: utf-8 -*-
"""
Image preprocessing utilities for reinforcement learning.
"""

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from numpy import ndarray


def normalize_image(image: Union[np.ndarray, tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
    """
    Normalize an image to [-1, 1] range.

    Parameters
    ----------
    image : np.ndarray | tf.Tensor
        Input image with values in [0, 255].

    Returns
    -------
    np.ndarray | tf.Tensor
        Normalized image with values in [-1, 1].
    """
    if isinstance(image, tf.Tensor):
        image = tf.cast(image, dtype=tf.float32)
        image /= 127.5
        image -= 1.0
    else:
        image = image.astype(np.float32)
        image /= 127.5
        image -= 1.0

    return image


def preprocess_observation(
    observation: Union[ndarray, tf.Tensor], resize_shape: Optional[Tuple[int, int]] = None
) -> Union[ndarray, tf.Tensor]:
    """
    Preprocess an observation for input to a neural network.

    Parameters
    ----------
    observation : np.ndarray | tf.Tensors
        Input observation.
    resize_shape : Optional[Tuple[int, int]], optional
        Optional shape to resize the observation to.

    Returns
    -------
    np.ndarray | tf.Tensor
        Preprocessed observation.
    """
    observation = normalize_image(observation)

    if resize_shape is not None:
        if isinstance(observation, tf.Tensor):
            observation = tf.image.resize(observation, resize_shape)
        else:
            observation_tensor = tf.convert_to_tensor(observation)
            observation_tensor = tf.image.resize(observation_tensor, resize_shape)
            observation = observation_tensor.numpy()

    return observation
