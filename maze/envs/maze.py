"""
Maze Environment for MiniGrid.

A maze environment using the ``MiniGrid`` framework, where an agent must navigate from a starting position
to a green goal ball placed in a random room. The maze consists of a grid of rooms connected by doors.
"""

from typing import Any, SupportsFloat

from minigrid.core.world_object import WorldObj
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc


class Maze(RoomGridLevel):
    """
    A maze environment with a goal.

    The agent starts at a random position and must navigate a grid of rooms
    (defined by `num_rows` and `num_cols`, each `room_size` x `room_size` cells)
    to reach a green ball. Doors connect the rooms.

    Parameters
    ----------
    num_rows : int, optional
        Number of rows of rooms (default is 3).
    num_cols : int, optional
        Number of columns of rooms (default is 3).
    room_size : int, optional
        Size (width and height) of each room (default is 8).
    num_dists : int, optional
        Number of distractor objects to place (default is 1).
    doors_open : bool, optional
        If True, all doors in the maze start open (default is False).
    **kwargs
        Additional keyword arguments passed to the `RoomGridLevel` constructor.
    """

    def __init__(
        self,
        num_rows: int = 3,
        num_cols: int = 3,
        room_size: int = 8,
        num_dists: int = 1,
        doors_open: bool = False,
        **kwargs,
    ):
        """
        Initialize the Maze environment.

        Parameters
        ----------
        num_rows : int, optional
            Number of rows of rooms (default is 3).
        num_cols : int, optional
            Number of columns of rooms (default is 3).
        room_size : int, optional
            Size (width and height) of each room (default is 8).
        num_dists : int, optional
            Number of distractor objects to place (default is 1).
        doors_open : bool, optional
            If True, all doors in the maze start open (default is False).
        **kwargs
            Additional keyword arguments passed to the `RoomGridLevel` constructor.
        """
        self.num_dists = num_dists
        self.doors_open = doors_open
        self.instrs: GoToInstr | None = None
        self.visited: dict[tuple[int, int], int] = {}
        self.distance: int | None = None
        self.goal_position: tuple[int, int] = (0, 0)
        self.current_room: tuple[int, int] | None = None
        self.goal_room: tuple[int, int] | None = None
        self.room_transitions: int = 0
        super().__init__(num_rows=num_rows, num_cols=num_cols, room_size=room_size, **kwargs)

    def get_room_coords(self, x: int, y: int) -> tuple[int, int]:
        """
        Get the room coordinates (i, j) containing the grid position (x, y).

        Parameters
        ----------
        x : int
            The x-coordinate of the grid position.
        y : int
            The y-coordinate of the grid position.

        Returns
        -------
        Tuple[int, int]
            The room coordinates (column_index, row_index).
        """
        i = x // (self.room_size - 1)
        j = y // (self.room_size - 1)

        return (min(i, self.num_cols - 1), min(j, self.num_rows - 1))

    def reward(self) -> float:
        """
        Calculate the reward based on the agent's current state.

        The reward encourages moving towards the goal, exploring new cells,
        and penalizes steps taken and wall collisions.

        Returns
        -------
        float
            The calculated reward for the current step.

        Notes
        -----
        Reward components:
        - Goal Reached: +10.0 if agent is at the goal position.
        - Progress Reward: +0.1 * (previous_distance - current_distance).
          Positive if closer.
        - Exploration Bonus: +0.05 if the agent enters a cell for the first time.
        - Wall Collision Penalty: -0.5 if the agent's last action resulted in hitting a wall.
        - Step Penalty: -0.01 for each step taken.
        """
        # ##: Check for wall collision.
        wall_collision_penalty = 0
        if hasattr(self, "last_action_hit_wall") and self.last_action_hit_wall:
            wall_collision_penalty = -0.5

        # ##: Check if goal is reached.
        distance = abs(self.goal_position[1] - self.agent_pos[1]) + abs(self.goal_position[0] - self.agent_pos[0])
        if distance == 0:
            return 10.0

        # ##: Calculate progress towards goal.
        previous_distance = self.distance if self.distance is not None else distance
        self.distance = distance
        progress_reward = (previous_distance - distance) * 0.1

        # ##: Calculate exploration bonus.
        exploration_bonus = 0
        agent_pos_key = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        visit_count = self.visited.get(agent_pos_key, 0)
        if visit_count == 0:
            exploration_bonus = 0.05
        self.visited[agent_pos_key] = visit_count + 1

        # ##: Standard step penalty.
        step_penalty = -0.01

        # ##: Combine reward components.
        total_reward = progress_reward + exploration_bonus + step_penalty + wall_collision_penalty

        return total_reward

    def step(self, action: Any) -> tuple[object, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one time step within the environment.

        Overrides the parent `step` method to use the custom `reward` function.

        Parameters
        ----------
        action : Any
            The action taken by the agent.

        Returns
        -------
        obs : object
            The agent's observation of the current environment.
        reward : SupportsFloat
            Amount of reward returned after previous action.
        terminated : bool
            Whether the episode has ended (e.g., goal reached).
        truncated : bool
            Whether the episode was ended prematurely (e.g., time limit).
        info : Dict[str, Any]
            Contains auxiliary diagnostic information.
        """
        # ##: Store position before taking the step.
        prev_pos = self.agent_pos

        obs, _, terminated, truncated, info = super().step(action)

        # ##: Check if position changed after the step.
        self.last_action_hit_wall = self.agent_pos == prev_pos

        # ##: Calculate reward using the updated logic.
        reward = self.reward()

        return obs, reward, terminated, truncated, info

    def add_goal(self) -> WorldObj:
        """
        Add the goal object (green ball) to a random room.

        Places a green ball in a randomly selected room and stores its position.

        Returns
        -------
        WorldObj
            The goal object that was placed in the environment.
        """
        room_i = self._rand_int(0, self.num_cols)
        room_j = self._rand_int(0, self.num_rows)

        goal_obj, self.goal_position = self.add_object(room_i, room_j, "ball", "green")
        return goal_obj

    def gen_mission(self) -> None:  # type: ignore[override]
        """
        Generate a new mission (episode).

        Sets up the environment for a new episode by placing the agent, connecting rooms, adding the goal,
        generating the instruction, and initializing tracking variables.
        """
        self.place_agent()
        self.connect_all()

        goal_obj = self.add_goal()
        self.instrs = GoToInstr(ObjDesc(goal_obj.type, goal_obj.color))

        if self.doors_open:
            self.open_all_doors()

        # ##: Initialize state variables for the new mission.
        self.distance = abs(self.goal_position[1] - self.agent_pos[1]) + abs(self.goal_position[0] - self.agent_pos[0])
        self.current_room = self.get_room_coords(self.agent_pos[0], self.agent_pos[1])
        self.goal_room = self.get_room_coords(self.goal_position[0], self.goal_position[1])
        self.room_transitions = 0
        self.visited = {}
