"""
Easy Maze Environment for MiniGrid Curriculum Learning.

A slightly larger maze than BaseMaze, still relatively simple.
"""

from maze.envs.maze import Maze


class EasyMaze(Maze):
    """
    A 2x2 room maze with potentially closed doors.

    Inherits from the main Maze class with a slightly larger grid.
    """

    def __init__(self, room_size: int = 8, **kwargs):
        """
        Initialize the EasyMaze environment.

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
            doors_open=False,
            num_dists=0,
            **kwargs,
        )
