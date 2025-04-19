# -*- coding: utf-8 -*-
"""
Main training script for PPO agent on the Maze environment.

This script initializes the environment, agent, and buffer, then runs the PPO training loop for a specified number
of epochs. Metrics are logged, and agent weights are saved periodically.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna

import tensorflow as tf
from loguru import logger
from tqdm import tqdm

from reinforce import setup_logger

from reinforce.learning.utils.config import get_train_config
from reinforce.learning.utils.environment import setup_environment
from reinforce.learning.utils.logging import MetricsLogger
from reinforce.ppo.agent import PPOAgent
from reinforce.ppo.buffer import Buffer

setup_logger()

def train(config: Dict[str, Any], trial: Optional[optuna.Trial] = None) -> float:
    """
    Trains the PPO agent on the Maze environment using the provided configuration,
    optionally reporting intermediate results to Optuna for pruning.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary containing training configuration and hyperparameters.
    trial : Optional[optuna.Trial], optional
        An Optuna trial object. If provided, intermediate results are reported
        for pruning. Defaults to None.

    Returns
    -------
    float
        The average mean return over the last `avg_reward_window` epochs.
        Returns -infinity if pruned or an error occurs.
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
    all_epoch_mean_returns = []
    avg_reward_window = 10

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()
        sum_return = 0.0
        sum_length = 0
        num_episodes = 0

        desc = f"Epoch {epoch + 1}/{config['epochs']}"
        if trial:
            desc = f"Trial {trial.number} - {desc}"
        pbar = tqdm(range(config["steps_per_epoch"]), desc=desc, leave=False)
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
        mean_return = sum_return / num_episodes if num_episodes > 0 else -np.inf
        mean_length = sum_length / num_episodes if num_episodes > 0 else 0.0
        all_epoch_mean_returns.append(mean_return)
        logger.info(f"Epoch: {epoch + 1}/{config['epochs']} | Duration: {epoch_duration:.2f}s")
        logger.info(f"Mean Return: {mean_return:.2f} | Mean Length: {mean_length:.2f} | Episodes: {num_episodes}")

        metrics.log_epoch(epoch + 1, mean_return, mean_length, num_episodes, epoch_duration)

        # --- Optuna Pruning ---
        if trial:
            trial.report(-mean_return, epoch)
            if trial.should_prune():
                logger.warning(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
                metrics.close()
                env.close()
                return -float("inf")

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

    # ##: Calculate final metric: average return over the last N epochs.
    if len(all_epoch_mean_returns) >= avg_reward_window:
        final_avg_return = np.mean(all_epoch_mean_returns[-avg_reward_window:])
    elif all_epoch_mean_returns:
        final_avg_return = np.mean(all_epoch_mean_returns)
    else:
        final_avg_return = -float("inf")

    logger.info(f"Final average return (last {avg_reward_window} epochs): {final_avg_return:.2f}")
    return final_avg_return


if __name__ == "__main__":
    training_config = get_train_config()
    final_metric = train(training_config, trial=None)
    logger.info(f"Training complete. Final metric: {final_metric:.2f}")
