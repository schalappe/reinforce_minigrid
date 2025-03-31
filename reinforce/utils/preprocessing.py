# -*- coding: utf-8 -*-
"""
Preprocessing utilities for observations.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from numpy import ndarray


def normalize_image(image: Union[ndarray, tf.Tensor]) -> Union[ndarray, tf.Tensor]:
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
    observation : np.ndarray | tf.Tensor
        Input observation.
    resize_shape : Optional[Tuple[int, int]], optional
        Optional shape to resize the observation to.

    Returns
    -------
    np.ndarray | tf.Tensor
        Preprocessed observation.
    """
    observation = normalize_image(observation)

    # ##: Resize if needed.
    if resize_shape is not None:
        if isinstance(observation, tf.Tensor):
            observation = tf.image.resize(observation, resize_shape)
        else:
            observation_tensor = tf.convert_to_tensor(observation)
            observation_tensor = tf.image.resize(observation_tensor, resize_shape)
            observation = observation_tensor.numpy()

    return observation


def frame_stack(frames: List[Union[ndarray, tf.Tensor]], num_frames: int = 4) -> Union[ndarray, tf.Tensor]:
    """
    Stack multiple frames along the channel dimension.

    Parameters
    ----------
    frames : List[np.ndarray | tf.Tensor]
        List of frames to stack.
    num_frames : int, optional
        Number of frames to stack, by default 4.

    Returns
    -------
    np.ndarray | tf.Tensor
        Stacked frames.
    """
    if not frames:
        raise ValueError("Empty frames list provided")

    # ##: Ensure we have enough frames.
    while len(frames) < num_frames:
        frames.insert(0, frames[0])

    # ##: Use only the most recent num_frames.
    if len(frames) > num_frames:
        frames = frames[-num_frames:]

    # ##: Stack frames.
    if isinstance(frames[0], tf.Tensor):
        return tf.concat(frames, axis=-1)
    return np.concatenate(frames, axis=-1)
