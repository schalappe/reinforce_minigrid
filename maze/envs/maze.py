# -*- coding: utf-8 -*-
"""
A maze environment using the ``MiniGrid`` framework, where an agent must navigate to a green goal.

This module implements a simple maze navigation task where the agent needs to find and reach a
green ball goal. The environment uses the ``MiniGrid`` framework and inherits from ``RoomGridLevel``.
"""

from typing import Dict, Tuple

from minigrid.core.world_object import WorldObj
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc


class Maze(RoomGridLevel):
    """
    A maze environment where the agent must navigate to a green goal.

    The environment consists of a 3x3 grid of rooms connected by doors. The agent is randomly placed
    in the grid and must find its way to a green ball placed in a random room.

    Attributes
    ----------
    num_dists : int
        Number of distractors in the environment.
    doors_open : bool
        Flag indicating if doors should be open.
    instrs : str
        Instructions for the agent, typically a GoToInstr object.
    visited : Dict
        Dictionary tracking visited positions and visit counts.
    distance : int or None
        Manhattan distance to the goal from the current position.
    goal_position : Tuple
        The (x, y) coordinates of the goal.
    """

    def __init__(self, **kwargs):
        """
        Initialize the maze environment.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the RoomGridLevel constructor.
        """
        self.num_dists = 1
        self.doors_open = False
        self.instrs = ""
        self.visited: Dict = {}
        self.distance = None
        self.goal_position: Tuple = ()
        super().__init__(num_rows=3, num_cols=3, room_size=8, **kwargs)

    def reward(self) -> float:
        """
        Compute the reward for the current state.

        The reward calculation is based on the agent's distance to the goal and the number of times the current
        position has been visited. Moving toward the goal provides positive rewards, while revisiting the same
        positions decreases rewards exponentially.

        Returns
        -------
        float
            The calculated reward value.

        Notes
        -----
        The reward mechanism encourages:
        1. Moving toward the goal (higher reward for shorter distances)
        2. Exploring new areas (revisiting penalizes through higher power)
        3. Following optimal paths to the goal
        """
        self.visited[self.agent_pos] = self.visited.get(self.agent_pos, 0) + 1

        # ##: Add position and compute distance.
        power = self.visited.get(self.agent_pos)
        distance = abs(self.goal_position[1] - self.agent_pos[1]) + abs(self.goal_position[0] - self.agent_pos[0])

        # ##: Punish agent or not.
        good_direction = self.distance > distance
        self.distance = distance
        reward = (1 / distance) ** power if good_direction else 0

        return reward

    def add_goal(self) -> WorldObj:
        """
        Add a goal object (green ball) to the maze for the agent to reach.

        This method randomly selects a room in the maze and places a green ball as the goal object.
        The goal position is stored in the `goal_position` attribute for later reference.

        Returns
        -------
        WorldObj:
            The WorldObj instance representing the goal that was added. This represents the object
            and distance information.
        """
        # ##: Add the object to a random room if no room specified.
        room_i = self._rand_int(0, self.num_cols)
        room_j = self._rand_int(0, self.num_rows)

        dist, self.goal_position = self.add_object(room_i, room_j, *("ball", "green"))

        return dist

    def gen_mission(self):
        """
        Generate a new mission (level) for the agent.

        This method:
        1. Places the agent in a random position
        2. Connects all rooms with doors
        3. Adds a goal object
        4. Creates instruction for the agent
        5. Initializes the distance to goal

        If doors_open is ``True``, all doors in the maze will be opened.
        """
        self.place_agent()
        self.connect_all()

        obj = self.add_goal()
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        if self.doors_open:
            self.open_all_doors()

        self.distance = abs(self.goal_position[1] - self.agent_pos[1]) + abs(self.goal_position[0] - self.agent_pos[0])
