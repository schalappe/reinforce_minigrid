"""
Manual Control Script for MiniGrid Maze Environment.

This script provides an interface for users to manually control an agent within a MiniGrid maze
environment using keyboard inputs.

Functions
---------
show_helper()
    Displays available keyboard actions in the console.
"""

import random

from rich.console import Console
from rich.table import Table


def show_helper():
    """
    Display Available Keyboard Actions.

    Prints a formatted table to the console listing the keyboard controls and their corresponding
    actions within the MiniGrid environment.

    Notes
    -----
    Utilizes the `rich` library to generate a visually appealing table for displaying key mappings.
    """
    print(64 * "-")
    table = Table(title="Available actions:")

    # ##: Header
    table.add_column("Key")
    table.add_column("Action")

    # ##: Content.
    table.add_row(*("LEFT", "LEFT"))
    table.add_row(*("RIGHT", "RIGHT"))
    table.add_row(*("UP", "FORWARD"))
    table.add_row(*("SPACE", "TOGGLE"))
    table.add_row(*("PAGEUP", "PICK UP"))
    table.add_row(*("PAGEDOWN", "DROP"))
    table.add_row(*("TAB", "PICK UP"))
    table.add_row(*("LEFT SHIFT", "DROP"))
    table.add_row(*("ENTER", "DONE"))

    # Table.
    console = Console()
    console.print(table)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from minigrid.manual_control import ManualControl

    from maze.envs import HardMaze

    # ##: Parser.
    parser = ArgumentParser(description="Manual control interface for MiniGrid maze environments")
    parser.add_argument(
        "--action",
        action="store_true",
        help="Display of the action that you have access to in the game",
        default=False,
    )
    args = parser.parse_args()

    if args.action:
        show_helper()
    else:
        # ##: Load environment.
        mini_grid = HardMaze(render_mode="human")

        # ##: Play MiniGrid.
        manual_control = ManualControl(env=mini_grid, seed=random.seed())
        manual_control.start()
