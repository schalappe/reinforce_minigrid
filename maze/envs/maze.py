# -*- coding: utf-8 -*-
"""Create maze with minigrid."""
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc
from minigrid.core.world_object import WorldObj
from typing import Tuple, Dict


class Maze(RoomGridLevel):
    """Go to green goal."""

    def __init__(self, **kwargs):
        self.num_dists = 1
        self.doors_open = False
        self.instrs = ""
        self.visited: Dict = {}
        self.distance = None
        self.goal_position: Tuple = ()
        super().__init__(num_rows=3, num_cols=3, room_size=8, **kwargs)

    def reward(self) -> float:
        """Compute reward"""
        # ##: If not moved.
        self.visited[self.agent_pos] = self.visited.get(self.agent_pos, 0) + 1

        # ##: Add position and compute distance.
        power = self.visited.get(self.agent_pos)
        distance = abs(self.goal_position[1] - self.agent_pos[1]) + abs(self.goal_position[0] - self.agent_pos[0])

        # ##: Punish agent or not.
        good_direction = self.distance > distance
        self.distance = distance
        reward = (1/distance) ** power if good_direction else 0

        return reward

    def add_goal(self) -> WorldObj:
        """
        Add goal for agent.
        """
        # Add the object to a random room if no room specified
        room_i = self._rand_int(0, self.num_cols)
        room_j = self._rand_int(0, self.num_rows)

        dist, self.goal_position = self.add_object(room_i, room_j, *("ball", "green"))

        return dist

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        obj = self.add_goal()
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()

        self.distance = abs(self.goal_position[1] - self.agent_pos[1]) + abs(self.goal_position[0] - self.agent_pos[0])
