# -*- coding: utf-8 -*-
"""
Visualization tools for reinforcement learning agents.
"""

import os
from typing import List, Optional

import imageio
import numpy as np
from numpy import ndarray
from PIL import Image

# ##: TODO: Reduce code duplication.


class Visualizer:
    """
    Visualizer for reinforcement learning environments and agents.

    This class provides functionality for rendering and saving visualizations of agent behavior in environments.
    """

    def __init__(self, save_dir: str = "outputs/visualizations"):
        """
        Initialize the visualizer.

        Parameters
        ----------
        save_dir : str, optional
            Directory to save visualizations, by default "outputs/visualizations".
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def render_episode(
        self, observations: List[ndarray], path: Optional[str] = None, fps: int = 10, format: str = "gif"
    ) -> str:
        """
        Render an episode as a GIF or MP4.

        Parameters
        ----------
        observations : List[np.ndarray]
            List of observations.
        path : Optional[str], optional
            Path to save the visualization, by default ``None``.
        fps : int, optional
            Frames per second, by default 10.
        format : str, optional
            Format to save the visualization (gif or mp4), by default "gif".

        Returns
        -------
        str
            Path to the saved visualization.

        Raises
        ------
        ValueError
            If the format is not supported.
        ValueError
            If there are no observations.
        """
        if not observations:
            raise ValueError("No observations to render")

        if format not in ["gif", "mp4"]:
            raise ValueError(f"Unsupported format: {format}")

        # ##: Convert observations to image frames.
        frames = []
        for obs in observations:
            if obs.dtype != np.uint8:
                if obs.max() <= 1.0:
                    obs = (obs * 255).astype(np.uint8)
                else:
                    obs = obs.astype(np.uint8)

            img = Image.fromarray(obs)
            frames.append(img)

        # ##: Generate output path if not provided.
        if path is None:
            timestamp = np.random.randint(0, 10000)
            path = os.path.join(self.save_dir, f"episode_{timestamp}.{format}")

        # ##: Save visualization.
        if format == "gif":
            frames[0].save(
                path, save_all=True, append_images=frames[1:], optimize=False, duration=int(1000 / fps), loop=0
            )
        elif format == "mp4":
            np_frames = [np.array(frame) for frame in frames]
            imageio.mimsave(path, np_frames, fps=fps)

        return path

    def save_frame(self, observation: np.ndarray, path: Optional[str] = None, format: str = "png") -> str:
        """
        Save a single observation frame.

        Parameters
        ----------
        observation : np.ndarray
            Observation to save.
        path : Optional[str], optional
            Path to save the frame, by default ``None``.
        format : str, optional
            Format to save the frame (png, jpg, etc.), by default "png".

        Returns
        -------
        str
            Path to the saved frame.

        Raises
        ------
        ValueError
            If the format is not supported.
        """
        if format not in ["png", "jpg", "jpeg"]:
            raise ValueError(f"Unsupported format: {format}")

        # ##: Ensure observation is in uint8 format.
        if observation.dtype != np.uint8:
            if observation.max() <= 1.0:
                observation = (observation * 255).astype(np.uint8)
            else:
                observation = observation.astype(np.uint8)

        img = Image.fromarray(observation)

        # ##: Generate output path if not provided.
        if path is None:
            timestamp = np.random.randint(0, 10000)
            path = os.path.join(self.save_dir, f"frame_{timestamp}.{format}")

        img.save(path, format=format.upper())

        return path

    def make_grid(
        self, observations: List[ndarray], rows: int, cols: int, path: Optional[str] = None, format: str = "png"
    ) -> str:
        """
        Create a grid of observation frames.

        Parameters
        ----------
        observations : List[np.ndarray]
            List of observations.
        rows : int
            Number of rows in the grid.
        cols : int
            Number of columns in the grid.
        path : Optional[str], optional
            Path to save the grid, by default ``None``.
        format : str, optional
            Format to save the grid (png, jpg, etc.), by default "png".

        Returns
        -------
        str
            Path to the saved grid.

        Raises
        ------
        ValueError
            If there are not enough observations.
        ValueError
            If the format is not supported.
        """
        if len(observations) < rows * cols:
            raise ValueError(f"Not enough observations for {rows}x{cols} grid")

        if format not in ["png", "jpg", "jpeg"]:
            raise ValueError(f"Unsupported format: {format}")

        h, w = observations[0].shape[:2]
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # ##: Fill grid with observations.
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(observations):
                    break

                obs = observations[idx]

                if obs.dtype != np.uint8:
                    if obs.max() <= 1.0:
                        obs = (obs * 255).astype(np.uint8)
                    else:
                        obs = obs.astype(np.uint8)

                if len(obs.shape) == 2:
                    obs = np.stack([obs] * 3, axis=-1)

                grid[i * h : (i + 1) * h, j * w : (j + 1) * w] = obs

        img = Image.fromarray(grid)

        # ##: Generate output path if not provided.
        if path is None:
            timestamp = np.random.randint(0, 10000)
            path = os.path.join(self.save_dir, f"grid_{timestamp}.{format}")

        img.save(path, format=format.upper())

        return path
