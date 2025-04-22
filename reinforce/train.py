# -*- coding: utf-8 -*-
"""
Main training script for the PPO agent on the MiniGrid Maze environment.
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import tensorflow as tf
from loguru import logger
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

from maze.envs.maze import Maze

from . import setup_logger
from .agent import PPOAgent
from .config import MainConfig, load_config


def train(config: MainConfig):
    """
    Trains the PPO agent based on the provided configuration.

    Parameters
    ----------
    config : MainConfig
        A dataclass object containing all necessary configuration parameters
        (environment, training, PPO hyperparameters, logging).
    """
    setup_logger()
    logger.info("Starting PPO training with loaded configuration...")
    logger.info(f"Environment Seed: {config.environment.seed}")
    logger.info(f"Total Timesteps: {config.training.total_timesteps}")
    logger.info(f"Steps per Update: {config.training.steps_per_update}")
    logger.info(
        f"PPO Hyperparameters: lr={config.ppo.learning_rate}, gamma={config.ppo.gamma}, "
        f"lambda={config.ppo.lambda_gae}, clip={config.ppo.clip_param}, "
        f"entropy={config.ppo.entropy_coef}, vf_coef={config.ppo.value_coef}, "
        f"epochs={config.ppo.epochs}, batch_size={config.ppo.batch_size}"
    )
    logger.info(
        f"Logging: Log Interval={config.logging.log_interval}, Save Interval={config.logging.save_interval}, "
        f"Save Path={config.logging.save_path}, Load Path={config.logging.load_path}"
    )

    # ##: Seeding for reproducibility.
    np.random.seed(config.environment.seed)
    tf.random.set_seed(config.environment.seed)

    # ##: Create environment.
    env = ImgObsWrapper(FullyObsWrapper(Maze()))
    logger.info("Successfully created environment using direct Maze instantiation.")

    # ##: Initialize agent.
    agent = PPOAgent(
        env.observation_space,
        env.action_space,
        learning_rate=config.ppo.learning_rate,
        gamma=config.ppo.gamma,
        lam=config.ppo.lambda_gae,
        clip_param=config.ppo.clip_param,
        entropy_coef=config.ppo.entropy_coef,
        vf_coef=config.ppo.value_coef,
        epochs=config.ppo.epochs,
        batch_size=config.ppo.batch_size,
    )

    # ##: Load pre-trained models if specified.
    if config.logging.load_path:
        logger.info(f"Loading models from {config.logging.load_path}...")
        agent.load_models(config.logging.load_path)

    # ##: Prepare for saving models.
    if config.logging.save_path:
        save_dir = os.path.dirname(config.logging.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Models will be saved to {config.logging.save_path}_*.keras")

    # ##: Training loop variables.
    current_obs, _ = env.reset(seed=config.environment.seed)
    episode_reward = 0
    episode_length = 0
    total_episodes = 0
    update_cycle = 0

    # ##: Logging setup.
    reward_deque = deque(maxlen=100)
    length_deque = deque(maxlen=100)
    start_time = time.time()

    logger.info("Starting training loop...")
    for timestep in range(0, config.training.total_timesteps, config.training.steps_per_update):
        update_cycle += 1
        logger.debug(f"Update Cycle {update_cycle} | Timestep {timestep}/{config.training.total_timesteps}")

        last_obs_for_bootstrap = None

        for step in range(config.training.steps_per_update):
            # ##: Get action, value estimate, and log probability from agent.
            action, value, action_prob = agent.get_action(current_obs)

            # ##: Step the environment.
            next_obs_dict, reward, terminated, truncated, _ = env.step(action)
            next_obs = next_obs_dict

            # ##: Store transition in buffer.
            done = terminated or truncated
            agent.store_transition(current_obs, action, reward, value, done, action_prob)

            # ##: Update current state and episode trackers.
            current_obs = next_obs
            episode_reward += reward
            episode_length += 1

            # ##: Handle episode end.
            if done:
                total_episodes += 1
                reward_deque.append(episode_reward)
                length_deque.append(episode_length)
                logger.debug(
                    f"Episode {total_episodes} finished. Reward: {episode_reward:.2f}, Length: {episode_length}"
                )

                # ##: Store last observation only if episode ended due to truncation.
                last_obs_for_bootstrap = next_obs if truncated else None

                # ##: Reset environment.
                obs_dict, _ = env.reset()
                current_obs = obs_dict
                episode_reward = 0
                episode_length = 0
            else:
                if step == config.training.steps_per_update - 1:
                    # ##: If loop finishes without done, store the last observation for bootstrapping.
                    last_obs_for_bootstrap = next_obs

        # ##: Perform learning update.
        logger.debug(f"Performing learning update for cycle {update_cycle}...")
        agent.learn(last_state=last_obs_for_bootstrap)
        logger.debug("Learning update complete.")

        # ##: Logging.
        if update_cycle % config.logging.log_interval == 0 and len(reward_deque) > 0:
            mean_reward = np.mean(reward_deque)
            mean_length = np.mean(length_deque)
            elapsed_time = time.time() - start_time
            steps_done = timestep + config.training.steps_per_update
            fps = int(steps_done / elapsed_time) if elapsed_time > 0 else 0
            logger.info(f"Timesteps: {steps_done}/{config.training.total_timesteps} | Episodes: {total_episodes}")
            logger.info(f"Mean Reward (last 100): {mean_reward:.2f} | Mean Length (last 100): {mean_length:.1f}")
            logger.info(f"FPS: {fps} | Elapsed Time: {elapsed_time:.2f}s")

        # ##: Save models periodically.
        if config.logging.save_path and update_cycle % config.logging.save_interval == 0:
            logger.info(f"Saving models at timestep {steps_done}...")
            agent.save_models(config.logging.save_path)

    logger.info("Training finished.")
    env.close()

    # ##: Final model save.
    if config.logging.save_path:
        logger.info("Saving final models...")
        agent.save_models(config.logging.save_path + "_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent on MiniGrid Maze using YAML configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Argument for specifying the config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_training.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total training timesteps")
    parser.add_argument("--steps-per-update", type=int, default=None, help="Override steps collected per agent update")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--gamma", type=float, default=None, help="Override discount factor")
    parser.add_argument("--lam", type=float, default=None, help="Override GAE lambda parameter")
    parser.add_argument("--clip", type=float, default=None, help="Override PPO clip parameter")
    parser.add_argument("--entropy", type=float, default=None, help="Override entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=None, help="Override value function loss coefficient")
    parser.add_argument("--epochs", type=int, default=None, help="Override PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=None, help="Override PPO batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log progress interval (in updates)")
    parser.add_argument("--save-interval", type=int, default=None, help="Override model save interval (in updates)")
    parser.add_argument("--save-path", type=str, default=None, help="Override path prefix to save models")
    parser.add_argument("--load-path", type=str, default=None, help="Override path prefix to load pre-trained models")

    args = parser.parse_args()

    # Load configuration from YAML and apply CLI overrides
    configuration = load_config(config_path=args.config, args=args)

    # Start training with the loaded configuration
    train(configuration)
