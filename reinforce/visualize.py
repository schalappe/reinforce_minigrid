"""
Script to evaluate trained RL agents in the MiniGrid Maze environment and save the rendering as a GIF.

Supports both PPO and Rainbow DQN agents.
"""

import os
import random
from collections.abc import Callable

import click
import imageio
import numpy as np
import tensorflow as tf
from loguru import logger
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from maze.envs import BaseMaze, EasyMaze, HardMaze, MediumMaze
from reinforce import setup_logger

ENVS: dict[str, Callable] = {
    "base": BaseMaze,
    "easy": EasyMaze,
    "medium": MediumMaze,
    "hard": HardMaze,
}


@click.command()
@click.option(
    "--model-prefix",
    "model_path_prefix",
    type=click.Path(exists=False),
    default="./models/ppo_maze_final",
    help="Path prefix for loading the models.",
    show_default=True,
)
@click.option(
    "--output-gif",
    "output_gif_path",
    type=click.Path(exists=False),
    default="evaluation_render.gif",
    help="Path to save the output GIF file.",
    show_default=True,
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for environment and agent.",
    show_default="random",
)
@click.option(
    "--max-steps",
    type=int,
    default=1000,
    help="Maximum number of steps per evaluation episode.",
    show_default=True,
)
@click.option(
    "--level",
    type=click.Choice(["base", "easy", "medium", "hard"]),
    default="easy",
    help="Difficulty level of the maze.",
    show_default=True,
)
@click.option(
    "--algorithm",
    type=click.Choice(["ppo", "dqn"]),
    default="ppo",
    help="Algorithm type of the saved model.",
    show_default=True,
)
def evaluate_and_render(
    model_path_prefix: str, output_gif_path: str, seed: int | None, max_steps: int, level: str, algorithm: str
):
    """
    Load an RL agent, run it in the Maze environment, and save a GIF.

    Parameters
    ----------
    model_path_prefix : str
        Path prefix for loading the models.
    output_gif_path : str
        Path where the output GIF file will be saved.
    seed : int, optional
        Random seed for reproducibility.
    max_steps : int
        Maximum number of steps per episode.
    level : str
        Maze difficulty level.
    algorithm : str
        Algorithm type ('ppo' or 'dqn').
    """
    setup_logger()

    if seed is None:
        seed = random.randint(0, 1_000_000)

    logger.info(f"Starting evaluation with model prefix: {model_path_prefix}")
    logger.info(f"Algorithm: {algorithm.upper()}")
    logger.info(f"Output GIF will be saved to: {output_gif_path}")
    logger.info(f"Using seed: {seed}, Max steps: {max_steps}")

    # ##>: Seeding for reproducibility.
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # ##>: Create environment with RGB pixel observations.
    env = ENVS[level](render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    logger.info("Environment created successfully.")

    # ##>: Initialize agent based on algorithm type.
    if algorithm == "ppo":
        from reinforce.ppo.agent import PPOAgent

        agent = PPOAgent(env.observation_space, env.action_space)
    elif algorithm == "dqn":
        from reinforce.dqn.agent import RainbowAgent

        agent = RainbowAgent(env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # ##>: Load pre-trained models.
    logger.info(f"Loading models from prefix: {model_path_prefix}")
    try:
        agent.load_models(model_path_prefix)
        logger.info("Models loaded successfully.")
    except (FileNotFoundError, tf.errors.OpError) as exc:
        logger.error(f"Failed to load models: {exc}")
        env.close()
        return

    # ##>: Run evaluation loop and collect frames.
    frames = []
    current_obs, _ = env.reset(seed=seed)
    logger.info("Starting episode evaluation...")

    total_reward = 0.0
    for step in range(max_steps):
        frame = env.render()
        frames.append(frame)

        # ##>: Get action from the agent (add batch dimension).
        current_obs_batched = np.expand_dims(current_obs, axis=0)
        actions, _ = agent.get_action(current_obs_batched, training=False)
        action = actions[0]

        # ##>: Step the environment.
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        current_obs = next_obs

        if terminated or truncated:
            logger.info(
                f"Episode finished after {step + 1} steps. "
                f"Total reward: {total_reward:.2f} "
                f"(Terminated: {terminated}, Truncated: {truncated})"
            )
            frame = env.render()
            frames.append(frame)
            break
    else:
        logger.info(f"Reached max steps ({max_steps}). Total reward: {total_reward:.2f}")

    env.close()
    logger.info("Environment closed.")

    # ##>: Save frames as GIF.
    logger.info(f"Saving {len(frames)} frames to {output_gif_path}...")
    try:
        output_dir = os.path.dirname(output_gif_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        imageio.mimsave(output_gif_path, frames, fps=10)
        logger.info("GIF saved successfully.")
    except (OSError, ValueError) as exc:
        logger.error(f"Failed to save GIF: {exc}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    evaluate_and_render()
