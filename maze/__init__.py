# -*- coding: utf-8 -*-
"""Register environment."""

from gymnasium.envs.registration import register

register(
    id="Maze-v0",
    entry_point="reinforce_minigrid.maze.envs:Maze",
)
