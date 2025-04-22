# -*- coding: utf-8 -*-
"""
Utility functions for the PPO reinforcement learning implementation.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Union


def preprocess_observation(obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
    """
    Preprocesses the environment observation.

    Handles MiniGrid's dictionary observation format by extracting the 'image' component.
    If the observation is already a NumPy array, it's returned directly.

    Parameters
    ----------
    obs : Union[np.ndarray, Dict[str, np.ndarray]]
        The observation from the Gymnasium environment.

    Returns
    -------
    np.ndarray
        The preprocessed observation (typically the image array).

    Raises
    ------
    ValueError
        If the observation format is unexpected.
    """
    if isinstance(obs, dict) and 'image' in obs:
        # MiniGrid often returns observations as dicts containing the image
        return obs['image']
    elif isinstance(obs, np.ndarray):
        # Standard NumPy array observation
        return obs
    else:
        raise ValueError(f"Unexpected observation format: {type(obs)}")
