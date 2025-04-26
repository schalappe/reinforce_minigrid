# -*- coding: utf-8 -*-
"""
Hard Maze Environment for MiniGrid Curriculum Learning.

A larger and potentially more complex maze.
"""

from .maze import Maze


class HardMaze(Maze):
    """
    A larger 3x3 room maze with closed doors.

    Increases the grid size and complexity compared to MediumMaze.
    """

    def __init__(self, room_size: int = 8, **kwargs):
        """
        Initialize the HardMaze environment.

        Parameters
        ----------
        room_size : int, optional
            Size (width and height) of each room (default is 8).
        **kwargs
            Additional keyword arguments passed to the parent Maze constructor.
        """
        super().__init__(
            num_rows=3,
            num_cols=3,
            room_size=room_size,
            doors_open=False,
            num_dists=4,
            **kwargs,
        )
