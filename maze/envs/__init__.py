# -*- coding: utf-8 -*-
"""
Maze Environments for MiniGrid.

This module makes the Maze environment classes available for import. It includes the base Maze class
and several pre-defined maze configurations of varying difficulty levels.
"""

from maze.envs.maze import (
    BaseMaze,
    EasyMazeOne,
    EasyMazeTwo,
    HardMaze,
    Maze,
    MediumMazeOne,
    MediumMazeTwo,
)

__all__ = [
    "Maze",
    "BaseMaze",
    "EasyMazeOne",
    "EasyMazeTwo",
    "MediumMazeOne",
    "MediumMazeTwo",
    "HardMaze",
]
