# -*- coding: utf-8 -*-
"""
Script to evaluate a trained PPO agent on the Maze environment.

Loads a pre-trained agent and runs it for a specified number of episodes, reporting performance metrics
and optionally saving a GIF.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np
from loguru import logger
from tqdm import tqdm

from reinforce.learning.utils.config import DEFAULT_EVAL_CONFIG
from reinforce.learning.utils.environment import setup_environment
from reinforce.ppo.agent import PPOAgent


def evaluate(config: Dict[str, Any]):
    """
    Evaluates the trained PPO agent based on the provided configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary containing evaluation configuration. Expected keys include
        'seed', 'load_dir', 'weights_prefix', 'num_episodes', 'save_gif',
        'gif_path', 'render', and network parameters ('conv_filters', etc.)
        matching the saved agent.
    """
    logger.info("--- PPO Evaluation Initialization ---")
    logger.info(f"Configuration: {config}")

    # --- Setup ---
    # ##: Environment setup now uses the utility function which also handles seeding.
    env = setup_environment(config["seed"])

    # --- Agent Initialization and Weight Loading ---
    # ##: Agent needs observation and action spaces. Network params must match saved model.
    network_params = {k: v for k, v in config.items() if k in ["conv_filters", "conv_kernel_size", "dense_units"]}
    agent = PPOAgent(env.observation_space, env.action_space, network_params=network_params)

    load_path_prefix = Path(config["load_dir"]) / config["weights_prefix"]
    logger.info(f"Loading agent weights from prefix: {load_path_prefix}")
    agent.load_weights(load_path_prefix)
    logger.info("Weights loaded successfully.")

    # --- Evaluation Loop ---
    logger.info(f"\n--- Starting Evaluation ({config['num_episodes']} episodes) ---")
    total_rewards: List[float] = []
    episode_lengths: List[int] = []
    frames: List[np.ndarray] = []

    for episode in tqdm(range(config["num_episodes"]), desc="Evaluating Episodes"):
        eval_seed = config["seed"] + episode
        observation, _ = env.reset(seed=eval_seed)
        episode_reward = 0.0
        episode_length = 0
        done = False
        step_count = 0

        while not done:
            if config["save_gif"]:
                frames.append(observation)

            # --- Agent Action Selection ---
            action_tensor, _, _ = agent.sample_action(observation)
            action = action_tensor[0].numpy()

            # --- Environment Step ---
            observation_new, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            step_count += 1

            # ##: Update observation.
            observation = observation_new

            # ##: Check for step limit truncation.
            if config.get("max_episode_steps") and step_count >= config["max_episode_steps"]:
                if not done:
                    done = True

        # --- End of Episode ---
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if config["save_gif"]:
            # ##: Add a blank frame as a separator between episodes.
            frames.append(np.zeros_like(frames[-1]))

    # --- Aggregate Results ---
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_length = np.mean(episode_lengths)

    logger.info("\n--- Evaluation Results ---")
    logger.info(f"Episodes Evaluated: {config['num_episodes']}")
    logger.info(f"Mean Reward:        {mean_reward:.2f} +/- {std_reward:.2f}")
    logger.info(f"Mean Episode Length:{mean_length:.2f}")

    # --- Save GIF ---
    if config["save_gif"] and frames:
        gif_path = Path(config["gif_path"])
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving evaluation GIF ({len(frames)} frames) to {gif_path}...")

        gif_frames = [frame.astype(np.uint8) for frame in frames]
        try:
            imageio.mimsave(gif_path, gif_frames, fps=10)
            logger.info("GIF saved successfully.")
        except Exception as e:
            logger.error(f"Error saving GIF: {e}")

    # --- Cleanup ---
    env.close()
    logger.info("Evaluation complete.")


def parse_arguments() -> Dict[str, Any]:
    """
    Parses command-line arguments for the evaluation script.

    Uses defaults from DEFAULT_EVAL_CONFIG and allows overrides.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Evaluate PPO agent on MiniGrid Maze")

    # Core Evaluation Parameters
    parser.add_argument(
        "--num-episodes", type=int, default=DEFAULT_EVAL_CONFIG["num_episodes"], help="Number of episodes to evaluate"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_CONFIG["seed"], help="Random seed for evaluation")
    parser.add_argument(
        "--load-dir", type=str, default=DEFAULT_EVAL_CONFIG["load_dir"], help="Directory containing saved weights"
    )
    parser.add_argument(
        "--weights-prefix",
        type=str,
        default=DEFAULT_EVAL_CONFIG["weights_prefix"],
        help="Prefix of the saved weights files (e.g., ppo_maze_epoch_50)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_EVAL_CONFIG["max_episode_steps"],
        help="Maximum steps per evaluation episode",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=DEFAULT_EVAL_CONFIG["render"],
        help="Render environment during evaluation",
    )

    # GIF Saving Parameters
    parser.add_argument(
        "--save-gif",
        action="store_true",
        default=DEFAULT_EVAL_CONFIG["save_gif"],
        help="Save a GIF of the evaluation episodes",
    )
    parser.add_argument(
        "--gif-path", type=str, default=DEFAULT_EVAL_CONFIG["gif_path"], help="Path to save the evaluation GIF"
    )

    # Network Parameters (must match the saved model)
    parser.add_argument(
        "--conv-filters",
        type=int,
        default=DEFAULT_EVAL_CONFIG["conv_filters"],
        help="Number of filters in convolutional layers (must match trained model)",
    )
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=DEFAULT_EVAL_CONFIG["conv_kernel_size"],
        help="Kernel size for convolutional layers (must match trained model)",
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=DEFAULT_EVAL_CONFIG["dense_units"],
        help="Number of units in dense layers (must match trained model)",
    )

    args = parser.parse_args()
    # Ensure kernel size is present if filters are specified (common dependency)
    if args.conv_filters is not None and args.conv_kernel_size is None:
        parser.error("--conv-kernel-size is required if --conv-filters is specified.")

    return vars(args)


if __name__ == "__main__":
    evaluation_config = parse_arguments()
    evaluate(evaluation_config)
