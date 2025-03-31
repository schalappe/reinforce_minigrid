# -*- coding: utf-8 -*-
"""
Visualization tools for reinforcement learning agents.
"""

import os
from typing import List, Optional, Set

import imageio.v2 as imageio
import numpy as np
from numpy import ndarray
from PIL import Image


class Visualizer:
    """
    Visualizer for reinforcement learning environments and agents.

    This class provides functionality for rendering and saving visualizations of agent behavior in environments.
    """

    # ##: Format validation constants.
    _IMAGE_FORMATS: Set[str] = {"png", "jpg", "jpeg"}
    _VIDEO_FORMATS: Set[str] = {"gif", "mp4"}

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

    @staticmethod
    def _preprocess_observation(observation: ndarray) -> ndarray:
        """
        Preprocess an observation to ensure it is in uint8 format.

        Parameters
        ----------
        observation : ndarray
            The observation to preprocess.

        Returns
        -------
        ndarray
            Preprocessed observation in uint8 format.
        """
        if observation.dtype != np.uint8:
            if observation.max() <= 1.0:
                observation = (observation * 255).astype(np.uint8)
            else:
                observation = observation.astype(np.uint8)
        return observation

    @staticmethod
    def _validate_format(format: str, allowed_formats: Set[str]) -> None:
        """
        Validate that the given format is supported.

        Parameters
        ----------
        format : str
            Format to validate.
        allowed_formats : Set[str]
            Set of allowed formats.

        Raises
        ------
        ValueError
            If the format is not supported.
        """
        if format not in allowed_formats:
            formats_str = ", ".join(allowed_formats)
            raise ValueError(f"Unsupported format: {format}. Supported formats: {formats_str}")

    def _generate_path(self, prefix: str, format: str, path: Optional[str] = None) -> str:
        """
        Generate a path for saving visualizations.

        Parameters
        ----------
        prefix : str
            Prefix for the filename.
        format : str
            File format.
        path : str, optional
            User-specified path, by default ``None``.

        Returns
        -------
        str
            Generated or user-specified path.
        """
        if path is None:
            timestamp = np.random.randint(0, 10000)
            path = os.path.join(self.save_dir, f"{prefix}_{timestamp}.{format}")
        return path

    def _convert_to_pil_images(self, observations: List[ndarray]) -> List[Image.Image]:
        """
        Convert a list of observations to PIL images.

        Parameters
        ----------
        observations : List[ndarray]
            List of observations.

        Returns
        -------
        List[Image.Image]
            List of PIL images.
        """
        return [Image.fromarray(self._preprocess_observation(obs)) for obs in observations]

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

        self._validate_format(format, self._VIDEO_FORMATS)

        # ##: Convert observations to image frames.
        frames = self._convert_to_pil_images(observations)

        # ##: Generate output path if not provided.
        path = self._generate_path("episode", format, path)

        # ##: Save visualization.
        if format == "gif":
            frames[0].save(
                path, save_all=True, append_images=frames[1:], optimize=False, duration=int(1000 / fps), loop=0
            )
        elif format == "mp4":
            np_frames = [np.array(frame) for frame in frames]
            writer = imageio.get_writer(path, fps=fps)
            for frame in np_frames:
                writer.append_data(frame)
            writer.close()

        return path

    def save_frame(self, observation: ndarray, path: Optional[str] = None, format: str = "png") -> str:
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
        self._validate_format(format, self._IMAGE_FORMATS)

        # ##: Process the observation and convert to PIL Image.
        img = Image.fromarray(self._preprocess_observation(observation))

        # ##: Generate output path if not provided.
        path = self._generate_path("frame", format, path)

        # ##: Save the image.
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

        self._validate_format(format, self._IMAGE_FORMATS)

        h, w = observations[0].shape[:2]
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # ##: Fill grid with observations.
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(observations):
                    break

                # ##: Preprocess the observation.
                obs = self._preprocess_observation(observations[idx])

                # ##: Convert grayscale to RGB if needed.
                if len(obs.shape) == 2:
                    obs = np.stack([obs] * 3, axis=-1)

                grid[i * h : (i + 1) * h, j * w : (j + 1) * w] = obs

        img = Image.fromarray(grid)

        # ##: Generate output path if not provided.
        path = self._generate_path("grid", format, path)

        # ##: Save the image.
        img.save(path, format=format.upper())

        return path
