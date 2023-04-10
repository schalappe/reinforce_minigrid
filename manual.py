# -*- coding: utf-8 -*-
"""Script to try MiniGrid."""
from rich.console import Console
from rich.table import Table


def show_helper():
    """Print in console available actions."""
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
    import argparse

    import gymnasium as gym
    from minigrid.manual_control import ManualControl

    # ##: Parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", help="Name of the environment to run", default="BabyAI-GoToRedBallNoDists-v0")
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
        mini_grid = gym.make(args.environment, render_mode="human")

        # ##: Play MiniGrid.
        manual_control = ManualControl(env=mini_grid, seed=1335)
        manual_control.start()
