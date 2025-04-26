# -*- coding: utf-8 -*-
"""
Base Maze Environment for MiniGrid Curriculum Learning.

A very simple maze, intended as the starting point for training.
"""

from .maze import Maze


class BaseMaze(Maze):
    """
    A simple 2x2 room maze with open doors.

    Inherits from the main Maze class but simplifies the layout.
    """

    def __init__(self, room_size: int = 8, **kwargs):
        """
        Initialize the BaseMaze environment.

        Parameters
        ----------
        room_size : int, optional
            Size (width and height) of each room (default is 8).
        **kwargs
            Additional keyword arguments passed to the parent Maze constructor.
        """
        super().__init__(
            num_rows=2,
            num_cols=2,
            room_size=room_size,
            doors_open=True,
            num_dists=0,
            **kwargs,
        )
