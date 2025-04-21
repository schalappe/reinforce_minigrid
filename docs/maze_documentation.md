# Technical Documentation: Maze Environment

This document provides a detailed technical overview of the `Maze` environment implemented within the `maze/` directory, utilizing the `MiniGrid` framework.

## Table of Contents

1.  [Architecture Overview](#architecture-overview)
2.  [Detailed Component Documentation](#detailed-component-documentation)
    *   [Class: `Maze`](#class-maze)
    *   [Attributes](#attributes)
    *   [Methods](#methods)
        *   [`__init__(self, **kwargs)`](#init)
        *   [`get_room_coords(self, x, y)`](#get_room_coords)
        *   [`reward(self) -> float`](#reward)
        *   [`step(self, action: object) -> Tuple[object, SupportsFloat, bool, bool, dict[str, Any]]`](#step)
        *   [`add_goal(self) -> WorldObj`](#add_goal)
        *   [`gen_mission(self)`](#gen_mission)
3.  [Relationships and Interdependencies](#relationships-and-interdependencies)
4.  [Algorithmic Principles](#algorithmic-principles)
    *   [Reward Calculation](#reward-calculation)
    *   [Mission Generation](#mission-generation)
5.  [Diagrams](#diagrams)
    *   [Conceptual Grid Layout](#conceptual-grid-layout)
6.  [Usage Examples](#usage-examples)
7.  [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## 1. Architecture Overview

The `Maze` environment is built upon the `MiniGrid` framework, specifically inheriting from the `RoomGridLevel` class provided within `minigrid.envs.babyai.core.roomgrid_level`. This inheritance provides a foundational structure for creating grid-based environments composed of multiple rooms.

The core architecture consists of:

*   **Grid Structure:** A 3x3 grid of rooms, each with a size of 8x8 cells (including walls).
*   **Agent:** A controllable agent placed within the grid.
*   **Goal:** A target object (a green ball) placed randomly in one of the rooms.
*   **Connectivity:** Rooms are connected by doors, which can be initially open or closed.
*   **Task:** The agent's objective is to navigate the maze and reach the goal location.
*   **Interaction:** The environment processes agent actions, updates the state, and calculates rewards based on the agent's progress and exploration.

---

## 2. Detailed Component Documentation

### Class: `Maze`

This class defines the maze navigation environment.

```python
class Maze(RoomGridLevel):
    # ... implementation ...
```

It encapsulates the environment's state, dynamics, and reward structure.

### Attributes

*   **`num_dists: int`**: (Default: 1) Specifies the number of distractor objects. *Note: Although initialized, distractors are not explicitly added or used in the current `add_goal` or `gen_mission` logic.*
*   **`doors_open: bool`**: (Default: `False`) A flag determining if doors between rooms should start in an open state. If `True`, `open_all_doors()` is called during mission generation.
*   **`instrs: str`**: Stores the instructions for the agent, typically generated as a `GoToInstr` object pointing to the goal object (`ObjDesc`). Initialized as an empty string.
*   **`visited: Dict`**: A dictionary used to track the agent's visited positions (`Tuple[int, int]`) and the count of visits to each position. Used in the reward calculation to penalize excessive revisiting.
*   **`distance: int | None`**: Stores the Manhattan distance from the agent's current position to the goal. Updated in the `reward` method and initialized in `gen_mission`. Starts as `None`.
*   **`goal_position: Tuple`**: Stores the (x, y) coordinates of the goal object (green ball). Set within the `add_goal` method.
*   **`current_room: Tuple | None`**: Stores the (i, j) coordinates of the room the agent is currently in. Updated in the `reward` method and initialized in `gen_mission`. Starts as `None`.
*   **`goal_room: Tuple | None`**: Stores the (i, j) coordinates of the room containing the goal. Initialized in `gen_mission`. Starts as `None`.
*   **`room_transitions: int`**: Counts the number of times the agent has transitioned between rooms. Updated in the `reward` method and initialized in `gen_mission`. Starts as 0.

### Methods

#### `__init__(self, **kwargs)`

```python
def __init__(self, **kwargs):
    # ... initialization of attributes ...
    super().__init__(num_rows=3, num_cols=3, room_size=8, **kwargs)
```

*   **Purpose:** Initializes a new `Maze` environment instance.
*   **Functionality:**
    *   Sets default values for `num_dists`, `doors_open`, `instrs`.
    *   Initializes `visited` dictionary, `distance` to `None`, and `goal_position` to an empty tuple.
    *   Calls the parent class (`RoomGridLevel`) constructor, specifying a 3x3 grid of 8x8 rooms. Accepts additional keyword arguments (`**kwargs`) for the parent constructor.

#### `get_room_coords(self, x, y)`

```python
def get_room_coords(self, x, y):
    # ... implementation ...
```

*   **Purpose:** Calculates the room coordinates (i, j) for a given grid position (x, y).
*   **Functionality:** Divides the x and y coordinates by the room size (minus 1 for wall overlap) and clamps the result to the valid room indices. Returns a tuple `(room_i, room_j)`.

#### `reward(self) -> float`

```python
def reward(self) -> float:
    # ... reward calculation logic ...
```

*   **Purpose:** Calculates the reward for the agent based on its current state, incorporating both distance-based progress and room-based progress.
*   **Functionality:**
    1.  Increments the visit count for the agent's current position in the `visited` dictionary.
    2.  Calculates the current Manhattan distance to the `goal_position`.
    3.  **Goal Reached:** If the distance is 0, returns a high positive reward (10.0).
    4.  **Distance Progress Reward:** Calculates the difference between the previous Manhattan distance and the current distance. A positive reward (0.1 * change) is given for getting closer, and a negative reward for moving farther away.
    5.  **Room Transition Reward:**
        *   Determines the agent's current room using `get_room_coords`.
        *   If the agent has moved to a *new* room (`new_room != self.current_room`):
            *   Increments `self.room_transitions`.
            *   Calculates the Manhattan distance from the previous room (`self.current_room`) to the `goal_room`.
            *   Calculates the Manhattan distance from the new room (`new_room`) to the `goal_room`.
            *   If the new room is closer to the `goal_room` than the previous room (`new_room_dist < old_room_dist`), a positive reward (0.5) is given.
        *   Updates `self.current_room` to the `new_room`.
    6.  **Exploration Penalty:** Applies a small penalty based on the number of times the current state has been visited (`-0.01 * min(visit_count - 1, 10)`). This encourages exploring new states and is capped.
    7.  **Step Penalty:** Applies a small constant penalty (-0.01) for each step taken to encourage finding shorter paths.
    8.  Updates `self.distance` with the newly calculated Manhattan distance.
    9.  Returns the sum of distance progress reward, room transition reward, exploration penalty, and step penalty.

#### `step(self, action: object) -> Tuple[object, SupportsFloat, bool, bool, dict[str, Any]]`

```python
def step(self, action: object) -> Tuple[object, SupportsFloat, bool, bool, dict[str, Any]]:
    obs, _, terminated, truncated, info = super().step(action)
    reward = self.reward()
    return obs, reward, terminated, truncated, info
```

*   **Purpose:** Executes one timestep in the environment given an agent action.
*   **Functionality:**
    1.  Calls the parent class's `step` method (`super().step(action)`) to handle the core state transition (moving the agent, interacting with objects, etc.). This returns the observation, original reward (ignored), termination status, truncation status, and info dictionary.
    2.  Calls the custom `self.reward()` method to calculate the environment-specific reward based on the new state.
    3.  Returns the observation, the *custom* reward, termination status, truncation status, and info dictionary.

#### `add_goal(self) -> WorldObj`

```python
def add_goal(self) -> WorldObj:
    # ... logic to place goal ...
```

*   **Purpose:** Adds the goal object (a green ball) to the environment.
*   **Functionality:**
    1.  Randomly selects a room (`room_i`, `room_j`) within the 3x3 grid.
    2.  Uses the `self.add_object` method (inherited from `RoomGridLevel`) to place a "ball" object with "green" color in the selected room. This method returns the object description and its position.
    3.  Stores the goal's position in `self.goal_position`.
    4.  Returns the `WorldObj` instance representing the goal.

#### `gen_mission(self)`

```python
def gen_mission(self):
    # ... logic to set up a new episode ...
```

*   **Purpose:** Generates a new mission or episode configuration. Called at the start of each episode.
*   **Functionality:**
    1.  Places the agent randomly within the grid using `self.place_agent()`.
    2.  Connects all rooms with doors using `self.connect_all()`.
    3.  Calls `self.add_goal()` to place the green ball goal and gets the goal object description.
    4.  Creates the agent's instruction using `GoToInstr(ObjDesc(obj.type, obj.color))`, storing it in `self.instrs`.
    5.  If `self.doors_open` is `True`, calls `self.open_all_doors()` to open all doors in the grid.
    6.  Initializes `self.distance` by calculating the Manhattan distance between the agent's starting position and the `goal_position`.
    7.  Initializes room tracking variables: `self.current_room` (based on agent start), `self.goal_room` (based on goal position), and `self.room_transitions` to 0.

---

## 3. Relationships and Interdependencies

*   **Inheritance:** `Maze` inherits directly from `minigrid.envs.babyai.core.roomgrid_level.RoomGridLevel`. This provides the core functionality for grid creation, room management, object placement (`add_object`), agent placement (`place_agent`), door handling (`connect_all`, `open_all_doors`), and the basic `step` mechanics.
*   **MiniGrid Core:** Relies heavily on `minigrid.core` for fundamental elements like `WorldObj` (representing objects like the ball and walls), agent representation, and grid structure.
*   **BabyAI Core:** Uses components from `minigrid.envs.babyai.core`, specifically `RoomGridLevel` for the room structure and `verifier.GoToInstr` and `verifier.ObjDesc` for creating task instructions.
*   **Internal Methods:** `step` calls `reward`; `gen_mission` calls `place_agent`, `connect_all`, `add_goal`, and potentially `open_all_doors`. `reward` uses `self.goal_position` (set by `add_goal`) and `self.visited`.

---

## 4. Algorithmic Principles

### Reward Calculation

The `reward` function implements a reward structure combining distance-based progress, room-based progress, and exploration incentives/penalties:

*   **Distance Progress:** A reward component (`progress_reward`) based on the change in Manhattan distance to the goal. This encourages general movement towards the target location.
*   **Room Transition Progress:** A significant reward (`room_transition_reward`) is given when the agent moves into a *new* room that is closer (in terms of room coordinates' Manhattan distance) to the `goal_room`. This specifically incentivizes finding the correct sequence of rooms.
*   **Exploration Penalty:** Discourages revisiting the same grid cell multiple times (`exploration_penalty`). The penalty increases slightly with visits but is capped, encouraging the agent to find new paths without overly punishing necessary backtracking.
*   **Efficiency Penalty:** A small constant `step_penalty` incentivizes finding shorter paths, as longer paths accumulate more penalties.
*   **Terminal Reward:** A large positive reward is given only upon reaching the exact goal cell.

This combination aims to guide the agent effectively through the maze by rewarding both fine-grained movement towards the goal and coarse-grained progress at the room level.

### Mission Generation

The `gen_mission` method defines the procedure for setting up each new episode:

*   **Randomization:** Both the agent's starting position and the goal's room are chosen randomly. This ensures variability across episodes.
*   **Environment Structure:** Consistently creates a 3x3 grid of rooms connected by doors.
*   **Task Definition:** Clearly defines the task by placing a specific goal (green ball) and generating a corresponding instruction (`GoToInstr`).

---

## 5. Diagrams

### Conceptual Grid Layout

A simplified text representation of the 3x3 room structure:

```
+-------+-------+-------+
|       |       |       |
| Room  | Room  | Room  |
| (0,0) | (1,0) | (2,0) |
|       |       |       |
+---D---+---D---+---D---+
|       |       |       |
| Room  | Room  | Room  |
| (0,1) | (1,1) | (2,1) |
|       |       |       |
+---D---+---D---+---D---+
|       |       |       |
| Room  | Room  | Room  |
| (0,2) | (1,2) | (2,2) |
|       |       |       |
+-------+-------+-------+

D = Door (potentially open or closed)
Agent (A) starts in a random cell.
Goal (G) starts in a random cell within a random room.
```

*(Note: Each "Room" is internally an 8x8 grid including walls)*

---

## 6. Usage Examples

```python
import gymnasium as gym
from maze.envs.maze import Maze  # Assuming registration or direct import
# from minigrid.wrappers import ImgObsWrapper # Example wrapper

# --- Option 1: Direct Instantiation ---
env = Maze(room_size=8, doors_open=True) # Example: Start with open doors

# --- Option 2: Using Gymnasium Registry (Requires prior registration) ---
# gym.register(
#     id='MiniGrid-Maze-Custom-v0',
#     entry_point='maze.envs:Maze',
#     kwargs={'room_size': 8, 'doors_open': False} # Pass custom args during registration
# )
# env = gym.make('MiniGrid-Maze-Custom-v0')

# --- Environment Interaction Loop ---
observation, info = env.reset()
print(f"Initial Observation Shape: {observation['image'].shape}") # MiniGrid typically provides dict observations
print(f"Mission: {info.get('mission', env.instrs)}") # Access mission/instructions

terminated = False
truncated = False
total_reward = 0
step_count = 0
max_steps = 100 # Example step limit

while not terminated and not truncated:
    # Render the environment (optional)
    # env.render() # May require additional setup depending on environment

    # Agent selects an action (replace with your agent's policy)
    # For MiniGrid, actions are typically integers (e.g., 0: left, 1: right, 2: forward, ...)
    action = env.action_space.sample() # Example: Random action

    # Take a step
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_count += 1

    # Optional: Check for truncation based on step count
    if step_count >= max_steps:
        truncated = True

    # Optional: Print step info
    # print(f"Step: {step_count}, Action: {action}, Reward: {reward:.4f}, Done: {terminated or truncated}")

# Episode finished
print(f"Episode finished after {step_count} steps.")
print(f"Total Reward: {total_reward}")
# env.close() # Close the environment renderer if used
```

---

## 7. Limitations and Future Improvements

### Current Limitations

*   **Fixed Grid Size:** The environment is hardcoded to a 3x3 grid of 8x8 rooms.
*   **Simple Goal:** Only one type of goal (green ball) is used.
*   **Basic Distractors:** The `num_dists` attribute exists but isn't actively used to place distractor objects that might make the task more complex.
*   **Manhattan Distance Reward:** While effective, Manhattan distance doesn't account for obstacles (walls, closed doors) directly in the reward calculation, only implicitly through the agent's movement constraints.
*   **Limited Instructions:** Uses only `GoToInstr`. More complex instructions (e.g., sequences, conditional tasks) are not supported.

### Potential Improvements

*   **Configurable Grid/Room Size:** Allow specifying `num_rows`, `num_cols`, and `room_size` during initialization.
*   **Varied Goals and Objects:** Introduce different types/colors of goals or other interactive objects (keys, boxes).
*   **Implement Distractors:** Add logic to place `num_dists` distractor objects randomly, potentially penalizing interaction with them.
*   **More Sophisticated Reward Functions:**
    *   Consider pathfinding algorithms (like A*) to calculate a more accurate distance-to-goal considering obstacles for reward shaping.
    *   Implement rewards/penalties for interacting with specific objects (e.g., picking up a key).
*   **Complex Task Structures:** Integrate with more advanced instruction sets or procedural task generation frameworks from BabyAI or similar libraries.
*   **Obstacle Variety:** Add different types of obstacles within rooms beyond just walls and doors.
*   **Partial Observability:** Explore wrappers or modifications to limit the agent's field of view for a more challenging POMDP setting.
