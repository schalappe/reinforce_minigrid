"""
Medium Maze Environment for MiniGrid Curriculum Learning.

A more complex maze than EasyMaze, using the default 3x3 layout.
"""

from .maze import Maze


class MediumMaze(Maze):
    """
    A standard 3x3 room maze with closed doors.

    This uses the default configuration of the parent Maze class.
    """

    def __init__(self, room_size: int = 8, **kwargs):
        """
        Initialize the MediumMaze environment.

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
            num_dists=0,
            **kwargs,
        )
