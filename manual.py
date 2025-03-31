# -*- coding: utf-8 -*-
"""
Script to try ``MiniGrid``.

This script allows users to manually control an agent in a MiniGrid maze environment. It provides a graphical
interface where users can navigate through a maze using keyboard controls.
"""

import random

from rich.console import Console
from rich.table import Table


def show_helper():
    """
    Print in console available actions.

    This function creates and displays a formatted table showing all keyboard controls and their corresponding
    actions in the MiniGrid environment.

    Notes
    -----
    Uses the rich library to create a visually appealing console output with a table displaying key mappings for
    the MiniGrid environment.
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
    """
    Main execution block.

    This block is executed when the script is run directly (not imported as a module). It handles command-line
    arguments and launches either the help display or the interactive MiniGrid environment.

    Parameters
    ----------
    --action : bool, optional
        Flag to display available actions instead of launching the game. Default is ``False``.

    Examples
    --------
    Display the action helper:

    >>> python manual.py --action

    Launch the interactive MiniGrid environment:

    >>> python manual.py
    """
    import argparse

    from minigrid.manual_control import ManualControl

    from maze.envs import Maze

    # ##: Parser.
    parser = argparse.ArgumentParser(description="Manual control interface for MiniGrid maze environments")
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
        mini_grid = Maze(render_mode="human")

        # ##: Play MiniGrid.
        manual_control = ManualControl(env=mini_grid, seed=random.seed())
        manual_control.start()
