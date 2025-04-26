# -*- coding: utf-8 -*-
"""
Main training script for the PPO agent on the MiniGrid Maze environment.
"""

import argparse
import os
import time
from collections import deque
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from gymnasium import Env
from loguru import logger
from minigrid.wrappers import ImgObsWrapper

from maze.envs.base_maze import BaseMaze
from maze.envs.easy_maze import EasyMaze
from maze.envs.hard_maze import HardMaze
from maze.envs.medium_maze import MediumMaze

from . import setup_logger
from .agent import PPOAgent
from .config import MainConfig, load_config

# ##: Define the curriculum
CURRICULUM = [
    {"name": "BaseMaze", "env_class": BaseMaze, "threshold": 0.5, "max_steps": 500_000},
    {"name": "EasyMaze", "env_class": EasyMaze, "threshold": 1.0, "max_steps": 500_000},
    {"name": "MediumMaze", "env_class": MediumMaze, "threshold": 2.0, "max_steps": 1_000_000},
    {"name": "HardMaze", "env_class": HardMaze, "threshold": None, "max_steps": None},
]


def create_env(env_class: Callable[..., Any]) -> Env:
    """
    Helper function to create and wrap the environment.

    Parameters
    ----------
    env_class : Callable[..., Any]
        The environment class to instantiate and wrap.

    Returns
    -------
    env : gym.Env
        The wrapped environment.
    """
    env = ImgObsWrapper(env_class(render_mode="rgb_array"))
    return env


def train(config: MainConfig):
    """
    Trains the PPO agent based on the provided configuration using curriculum learning.

    Parameters
    ----------
    config : MainConfig
        A dataclass object containing all necessary configuration parameters
        (environment, training, PPO hyperparameters, logging).
    """
    setup_logger()
    logger.info("Starting PPO training with loaded configuration and curriculum learning...")
    logger.info(f"Environment Seed: {config.environment.seed}")
    logger.info(f"Total Timesteps (Target): {config.training.total_timesteps}")
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

    # ##: Curriculum setup.
    current_curriculum_index = 0
    current_stage = CURRICULUM[current_curriculum_index]
    stage_name = current_stage["name"]
    logger.info(f"Starting curriculum stage 1: {stage_name}")
    env = create_env(current_stage["env_class"])

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
    total_steps_overall = 0
    steps_in_current_stage = 0

    # ##: Logging setup.
    reward_deque = deque(maxlen=100)
    length_deque = deque(maxlen=100)
    start_time = time.time()

    logger.info("Starting training loop...")
    while total_steps_overall < config.training.total_timesteps:
        update_cycle += 1
        stage_name = current_stage["name"]
        logger.debug(
            f"Update Cycle {update_cycle} | "
            f"Overall Timestep {total_steps_overall}/{config.training.total_timesteps} | "
            f"Stage: {stage_name}"
        )

        last_obs_for_bootstrap = None
        steps_this_update = 0

        for step in range(config.training.steps_per_update):
            if total_steps_overall >= config.training.total_timesteps:
                break

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
            total_steps_overall += 1
            steps_in_current_stage += 1
            steps_this_update += 1

            # ##: Handle episode end.
            if done:
                total_episodes += 1
                reward_deque.append(episode_reward)
                length_deque.append(episode_length)
                stage_name = current_stage["name"]
                logger.debug(
                    f"Episode {total_episodes} finished. Reward: {episode_reward:.2f}, "
                    f"Length: {episode_length}, Stage: {stage_name}"
                )

                # ##: Store last observation only if episode ended due to truncation.
                last_obs_for_bootstrap = next_obs if truncated else None

                # ##: Reset environment (using a potentially different seed per episode if desired).
                obs_dict, _ = env.reset()
                current_obs = obs_dict
                episode_reward = 0
                episode_length = 0

                # ##: Check curriculum advancement criteria
                if len(reward_deque) == reward_deque.maxlen:
                    mean_reward = np.mean(reward_deque)
                    stage_threshold = current_stage["threshold"]
                    stage_max_steps = current_stage["max_steps"]
                    advance_curriculum = False
                    stage_name = current_stage["name"]

                    if stage_threshold is not None and mean_reward >= stage_threshold:
                        logger.info(
                            f"Stage {stage_name} threshold ({stage_threshold}) met "
                            f"with avg reward {mean_reward:.2f}."
                        )
                        advance_curriculum = True
                    elif stage_max_steps is not None and steps_in_current_stage >= stage_max_steps:
                        logger.info(f"Stage {stage_name} max steps ({stage_max_steps}) reached.")
                        advance_curriculum = True

                    if advance_curriculum and current_curriculum_index < len(CURRICULUM) - 1:
                        current_curriculum_index += 1
                        current_stage = CURRICULUM[current_curriculum_index]
                        stage_name = current_stage["name"]
                        logger.warning(f"Advancing to curriculum stage {current_curriculum_index + 1}: {stage_name}")
                        env.close()
                        env = create_env(current_stage["env_class"])

                        current_obs, _ = env.reset()
                        episode_reward = 0
                        episode_length = 0
                        reward_deque.clear()
                        length_deque.clear()
                        steps_in_current_stage = 0
                        last_obs_for_bootstrap = None

            else:
                if step == config.training.steps_per_update - 1:
                    # ##: If loop finishes without done, store the last observation for bootstrapping.
                    last_obs_for_bootstrap = next_obs

        # ##: Perform learning update only if steps were taken.
        if steps_this_update > 0:
            logger.debug(f"Performing learning update for cycle {update_cycle} with {steps_this_update} steps...")
            agent.learn(last_state=last_obs_for_bootstrap)
            logger.debug("Learning update complete.")
        else:
            logger.debug(f"Skipping learning update for cycle {update_cycle} as no steps were collected.")

        # ##: Logging.
        if update_cycle % config.logging.log_interval == 0 and len(reward_deque) > 0:
            mean_reward = np.mean(reward_deque)
            mean_length = np.mean(length_deque)
            elapsed_time = time.time() - start_time
            fps = int(total_steps_overall / elapsed_time) if elapsed_time > 0 else 0
            stage_name = current_stage["name"]
            logger.info(
                f"Timesteps: {total_steps_overall}/{config.training.total_timesteps} | "
                f"Episodes: {total_episodes} | Stage: {stage_name}"
            )
            logger.info(
                f"Mean Reward (last {reward_deque.maxlen}): {mean_reward:.2f} | "
                f"Mean Length (last {length_deque.maxlen}): {mean_length:.1f}"
            )
            logger.info(f"FPS: {fps} | Elapsed Time: {elapsed_time:.2f}s")

        # ##: Save models periodically.
        if config.logging.save_path and update_cycle % config.logging.save_interval == 0:
            save_file_path = f"{config.logging.save_path}_ts{total_steps_overall}_stage{current_curriculum_index}"
            logger.info(f"Saving models to {save_file_path}...")
            agent.save_models(save_file_path)

    logger.info("Training finished (total timesteps reached).")
    env.close()

    # ##: Final model save.
    if config.logging.save_path:
        final_save_path = f"{config.logging.save_path}_final_ts{total_steps_overall}_stage{current_curriculum_index}"
        logger.info(f"Saving final models to {final_save_path}...")
        agent.save_models(final_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent on MiniGrid Maze using YAML configuration and curriculum learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_training.yaml",
        help="Path to the YAML configuration file.",
    )

    # ##: Keep override arguments if needed, but curriculum might make some less relevant.
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
    configuration = load_config(config_path=args.config, args=args)

    train(configuration)
