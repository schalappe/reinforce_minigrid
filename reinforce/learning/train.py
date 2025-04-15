# -*- coding: utf-8 -*-
"""
Main training script for PPO agent on the Maze environment.

This script initializes the environment, agent, and buffer, then runs the PPO training loop for a specified number
of epochs. Metrics are logged, and agent weights are saved periodically.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure Keras uses TensorFlow backend

import tensorflow as tf
from loguru import logger
from tqdm import tqdm

from reinforce.learning.utils.config import DEFAULT_TRAIN_CONFIG
from reinforce.learning.utils.environment import setup_environment
from reinforce.learning.utils.logging import MetricsLogger
from reinforce.ppo.agent import PPOAgent
from reinforce.ppo.buffer import Buffer


def train(config: Dict[str, Any]):
    """
    Trains the PPO agent on the Maze environment using the provided configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary containing training configuration and hyperparameters.
        Expected keys include 'seed', 'save_dir', 'epochs', 'steps_per_epoch',
        'gamma', 'lam', 'save_freq', 'render', and PPO/network hyperparameters.
    """
    logger.info("--- PPO Training Initialization ---")
    logger.info(f"Configuration: {config}")

    # --- Setup ---
    metrics = MetricsLogger(config["save_dir"])
    env = setup_environment(config["seed"])
    observation_shape = env.observation_space.shape

    # --- Agent and Buffer Initialization ---
    agent_hyperparams = {
        k: v
        for k, v in config.items()
        if k
        in [
            "gamma",
            "lam",
            "clip_ratio",
            "policy_learning_rate",
            "value_function_learning_rate",
            "train_policy_iterations",
            "train_value_iterations",
            "target_kl",
            "seed",
        ]
    }
    network_params = {k: v for k, v in config.items() if k in ["conv_filters", "conv_kernel_size", "dense_units"]}
    agent = PPOAgent(
        env.observation_space, env.action_space, hyperparams=agent_hyperparams, network_params=network_params
    )
    buffer = Buffer(observation_shape, config["steps_per_epoch"], gamma=config["gamma"], lam=config["lam"])

    # --- Training Loop ---
    logger.info("\n--- Starting Training Loop ---")
    start_time = time.time()
    observation, _ = env.reset(seed=config["seed"])
    episode_return, episode_length = 0.0, 0

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()
        sum_return = 0.0
        sum_length = 0
        num_episodes = 0

        pbar = tqdm(range(config["steps_per_epoch"]), desc=f"Epoch {epoch + 1}/{config['epochs']}", leave=False)
        for t in pbar:
            if config["render"]:
                env.render()

            # --- Agent Interaction ---
            action_tensor, value, logprobability = agent.sample_action(observation)
            action = action_tensor[0].numpy()

            # --- Environment Step ---
            observation_new, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1

            # --- Store Experience ---
            buffer.store(observation, action, reward, value[0].numpy(), logprobability[0].numpy())

            # ##: Update observation.
            observation = observation_new

            # --- Episode/Epoch End Handling ---
            terminal = done
            epoch_ended = t == config["steps_per_epoch"] - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    logger.info(f"\nEpoch {epoch+1} ended mid-trajectory. Bootstrapping value.")
                    obs_tensor = tf.expand_dims(tf.convert_to_tensor(observation, dtype=tf.float32), 0)
                    last_value = agent.critic(obs_tensor)[0].numpy()
                else:
                    last_value = 0.0
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1

                    # ##: Reset environment (use a different seed for subsequent episodes).
                    reset_seed = config["seed"] + epoch * config["steps_per_epoch"] + t + 1
                    observation, _ = env.reset(seed=reset_seed)
                    episode_return, episode_length = 0.0, 0

                buffer.finish_trajectory(last_value)

        # --- PPO Update ---
        logger.info(f"\nEpoch {epoch + 1} finished collecting data. Performing PPO update...")
        agent.train(buffer)

        # --- Logging ---
        epoch_duration = time.time() - epoch_start_time
        mean_return = sum_return / num_episodes if num_episodes > 0 else 0.0
        mean_length = sum_length / num_episodes if num_episodes > 0 else 0.0
        logger.info(f"Epoch: {epoch + 1}/{config['epochs']} | Duration: {epoch_duration:.2f}s")
        logger.info(f"Mean Return: {mean_return:.2f} | Mean Length: {mean_length:.2f} | Episodes: {num_episodes}")

        metrics.log_epoch(epoch + 1, mean_return, mean_length, num_episodes, epoch_duration)

        # --- Save Weights ---
        if (epoch + 1) % config["save_freq"] == 0 or (epoch + 1) == config["epochs"]:
            save_path_prefix = Path(config["save_dir"]) / f"ppo_maze_epoch_{epoch + 1}"
            agent.save_weights(save_path_prefix)
            logger.info(f"Saved agent weights to {save_path_prefix}_actor/critic.keras")

    # --- End of Training ---
    total_duration = time.time() - start_time
    logger.info("\n--- Training Finished ---")
    logger.info(f"Total duration: {total_duration:.2f}s")
    metrics.close()
    env.close()


def parse_arguments() -> Dict[str, Any]:
    """
    Parses command-line arguments for the training script.

    Uses defaults from DEFAULT_TRAIN_CONFIG and allows overrides.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Train PPO agent on MiniGrid Maze")

    # Core Training Parameters
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=DEFAULT_TRAIN_CONFIG["steps_per_epoch"],
        help="Number of steps per epoch",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=DEFAULT_TRAIN_CONFIG["save_freq"],
        help="Frequency (in epochs) to save weights",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=DEFAULT_TRAIN_CONFIG["render"],
        help="Render environment during training",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_CONFIG["seed"], help="Random seed")
    parser.add_argument(
        "--save-dir", type=str, default=DEFAULT_TRAIN_CONFIG["save_dir"], help="Directory to save checkpoints and logs"
    )

    # PPO Hyperparameters
    parser.add_argument("--gamma", type=float, default=DEFAULT_TRAIN_CONFIG["gamma"], help="Discount factor")
    parser.add_argument("--lam", type=float, default=DEFAULT_TRAIN_CONFIG["lam"], help="GAE lambda parameter")
    parser.add_argument(
        "--clip-ratio", type=float, default=DEFAULT_TRAIN_CONFIG["clip_ratio"], help="PPO clipping ratio"
    )
    parser.add_argument(
        "--plr",
        type=float,
        default=DEFAULT_TRAIN_CONFIG["policy_learning_rate"],
        dest="policy_learning_rate",
        help="Policy learning rate",
    )
    parser.add_argument(
        "--vlr",
        type=float,
        default=DEFAULT_TRAIN_CONFIG["value_function_learning_rate"],
        dest="value_function_learning_rate",
        help="Value function learning rate",
    )
    parser.add_argument(
        "--pi-iters",
        type=int,
        default=DEFAULT_TRAIN_CONFIG["train_policy_iterations"],
        dest="train_policy_iterations",
        help="Policy training iterations per epoch",
    )
    parser.add_argument(
        "--v-iters",
        type=int,
        default=DEFAULT_TRAIN_CONFIG["train_value_iterations"],
        dest="train_value_iterations",
        help="Value function training iterations per epoch",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=DEFAULT_TRAIN_CONFIG["target_kl"],
        help="Target KL divergence for early stopping policy training",
    )

    # Network Parameters
    parser.add_argument(
        "--conv-filters",
        type=int,
        default=DEFAULT_TRAIN_CONFIG["conv_filters"],
        help="Number of filters in convolutional layers",
    )
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=DEFAULT_TRAIN_CONFIG["conv_kernel_size"],
        help="Kernel size for convolutional layers",
    )  # Added missing default access
    parser.add_argument(
        "--dense-units", type=int, default=DEFAULT_TRAIN_CONFIG["dense_units"], help="Number of units in dense layers"
    )

    args = parser.parse_args()
    return vars(args)  # Convert argparse Namespace to dict


if __name__ == "__main__":
    training_config = parse_arguments()
    train(training_config)
