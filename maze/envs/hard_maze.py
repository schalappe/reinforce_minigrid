# -*- coding: utf-8 -*-
"""
Hard Maze Environment for MiniGrid Curriculum Learning.

A larger and potentially more complex maze.
"""

from .maze import Maze


class HardMaze(Maze):
    """
    A larger 4x4 room maze with closed doors.

    Increases the grid size and complexity compared to MediumMaze.
    """

    def __init__(self, room_size: int = 8, doors_open: bool = False, **kwargs):
        """
        Initialize the HardMaze environment.

        Parameters
        ----------
        room_size : int, optional
            Size (width and height) of each room (default is 8).
        doors_open : bool, optional
            If True, all doors in the maze start open (default is False).
        **kwargs
            Additional keyword arguments passed to the parent Maze constructor.
        """
        super().__init__(
            num_rows=4,
            num_cols=4,
            room_size=room_size,
            doors_open=doors_open,
            num_dists=0,
            **kwargs,
        )
