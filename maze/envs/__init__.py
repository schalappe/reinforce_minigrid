# -*- coding: utf-8 -*-
"""
Initialize the maze environments module.

Exposes the different maze environment classes.
"""

from .base_maze import BaseMaze
from .easy_maze import EasyMaze
from .hard_maze import HardMaze
from .maze import Maze
from .medium_maze import MediumMaze

__all__ = ["Maze", "BaseMaze", "EasyMaze", "MediumMaze", "HardMaze"]
