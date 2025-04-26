# -*- coding: utf-8 -*-
"""
Maze Environment Registration.

This module registers the custom Maze environment with Gymnasium, making it available
for use through `gymnasium.make()`.
"""

from gymnasium.envs.registration import register

register(
    id="Maze-v0",
    entry_point="reinforce_minigrid.maze.envs:Maze",
)
