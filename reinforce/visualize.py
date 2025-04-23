# -*- coding: utf-8 -*-
"""
Script to evaluate a trained PPO agent in the MiniGrid Maze environment and save the rendering as a GIF.
"""

import os
import random

import click
import imageio
import numpy as np
import tensorflow as tf
from loguru import logger
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

from maze.envs import Maze
from reinforce import setup_logger
from reinforce.agent import PPOAgent


@click.command()
@click.option(
    "--model-prefix",
    "model_path_prefix",
    type=click.Path(exists=False),
    default="../models/ppo_maze_final",
    help="Path prefix for loading the policy and value models.",
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
    default=random.randint(0, 1_000_000),
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
def evaluate_and_render(model_path_prefix: str, output_gif_path: str, seed: int, max_steps: int):
    """
    Loads a PPO agent, runs it in the Maze environment, and saves a GIF.

    Parameters
    ----------
    model_path_prefix : str
        Path prefix for loading the policy and value models (e.g., 'models/ppo_maze_final').
    output_gif_path : str
        Path where the output GIF file will be saved.
    seed : int, optional
        Random seed for reproducibility.
    max_steps : int, optional
        Maximum number of steps per episode evaluation.
    """
    setup_logger()
    logger.info(f"Starting evaluation with model prefix: {model_path_prefix}")
    logger.info(f"Output GIF will be saved to: {output_gif_path}")
    logger.info(f"Using seed: {seed}, Max steps: {max_steps}")

    # ##: Seeding for reproducibility.
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # ##: Create environment with RGB array rendering.
    env = ImgObsWrapper(FullyObsWrapper(Maze(render_mode="rgb_array")))
    logger.info("Successfully created environment.")

    # ##: Initialize agent.
    agent = PPOAgent(env.observation_space, env.action_space)

    # ##: Load pre-trained models.
    logger.info(f"Loading models from prefix: {model_path_prefix}")
    try:
        agent.load_models(model_path_prefix)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        env.close()
        return

    # ##: Run evaluation loop and collect frames.
    frames = []
    current_obs_dict, _ = env.reset(seed=seed)
    current_obs = current_obs_dict
    logger.info("Starting episode evaluation...")

    for step in range(max_steps):
        # ##: Render the environment.
        frame = env.render()
        frames.append(frame)

        # ##: Get action from the agent (only need action, ignore value/prob).
        action, _, _ = agent.get_action(current_obs)

        # ##: Step the environment.
        next_obs_dict, _, terminated, truncated, _ = env.step(action)
        next_obs = next_obs_dict

        # ##: Update current state.
        current_obs = next_obs

        # ##: Check if episode ended.
        if terminated or truncated:
            logger.info(f"Episode finished after {step + 1} steps (Terminated: {terminated}, Truncated: {truncated}).")
            frame = env.render()
            frames.append(frame)
            break
        else:
            frame = env.render()
            frames.append(frame)


    env.close()
    logger.info("Environment closed.")

    # ##: Save frames as GIF.
    logger.info(f"Saving {len(frames)} frames to {output_gif_path}...")
    try:
        output_dir = os.path.dirname(output_gif_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        imageio.mimsave(output_gif_path, frames, fps=10)
        logger.info("GIF saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save GIF: {e}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    evaluate_and_render()
