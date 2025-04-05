# -*- coding: utf-8 -*-
"""
Frame stacking utilities for reinforcement learning.
"""

from typing import List, Union

import numpy as np
import tensorflow as tf


def frame_stack(frames: List[Union[np.ndarray, tf.Tensor]], num_frames: int = 4) -> Union[np.ndarray, tf.Tensor]:
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

    Raises
    ------
    ValueError
        If the frames list is empty.
    """
    if not frames:
        raise ValueError("Empty frames list provided")

    # ##: Ensure we have enough frames.
    while len(frames) < num_frames:
        frames.insert(0, frames[0])

    # ##: Use only the most recent num_frames.
    if len(frames) > num_frames:
        frames = frames[-num_frames:]

    # ##: Stack frames with proper type casting.
    if isinstance(frames[0], tf.Tensor):
        result = tf.concat(frames, axis=-1)
        return tf.cast(result, tf.float32)
    result = np.concatenate(frames, axis=-1)
    return np.asarray(result, dtype=np.float32)
