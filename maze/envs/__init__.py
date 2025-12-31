"""
Initialize the maze environments module.

Exposes the different maze environment classes.
"""

from maze.envs.base_maze import BaseMaze
from maze.envs.easy_maze import EasyMaze
from maze.envs.hard_maze import HardMaze
from maze.envs.maze import Maze
from maze.envs.medium_maze import MediumMaze

__all__ = ["Maze", "BaseMaze", "EasyMaze", "MediumMaze", "HardMaze"]
