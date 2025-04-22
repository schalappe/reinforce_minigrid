# -*- coding: utf-8 -*-
"""
Main training script for the PPO agent on the MiniGrid Maze environment.
"""

import argparse
import os
import time
from collections import deque
from typing import Optional

import numpy as np
import tensorflow as tf
from loguru import logger
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

from maze.envs.maze import Maze

from . import setup_logger
from .agent import PPOAgent


def train(
    total_timesteps: int = 1_000_000,
    steps_per_update: int = 2048,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_param: float = 0.2,
    entropy_coef: float = 0.01,
    vf_coef: float = 0.5,
    epochs: int = 10,
    batch_size: int = 64,
    seed: int = 42,
    log_interval: int = 1,
    save_interval: int = 10,
    save_path: str = "models/ppo_maze",
    load_path: Optional[str] = None,
):
    """
    Trains the PPO agent on the specified environment.

    Parameters
    ----------
    env_id : str, optional
        Gymnasium environment ID or class name. Default is "MiniGrid-Maze-Custom-v0".
    total_timesteps : int, optional
        Total number of environment steps to train for. Default is 1,000,000.
    steps_per_update : int, optional
        Number of steps to collect data for before each learning update. Default is 2048.
    learning_rate : float, optional
        Learning rate for optimizers. Default is 3e-4.
    gamma : float, optional
        Discount factor. Default is 0.99.
    lam : float, optional
        GAE lambda parameter. Default is 0.95.
    clip_param : float, optional
        PPO clipping parameter epsilon. Default is 0.2.
    entropy_coef : float, optional
        Coefficient for the entropy bonus. Default is 0.01.
    vf_coef : float, optional
        Coefficient for the value function loss. Default is 0.5.
    epochs : int, optional
        Number of optimization epochs per learning update. Default is 10.
    batch_size : int, optional
        Mini-batch size for optimization. Default is 64.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    log_interval : int, optional
        Log progress every N update cycles. Default is 1.
    save_interval : int, optional
        Save models every N update cycles. Default is 10.
    save_path : str, optional
        Directory and prefix for saving models. Default is "models/ppo_maze".
    load_path : Optional[str], optional
        Path prefix to load pre-trained models from. Default is None.
    """
    setup_logger()
    logger.info("Starting PPO training...")
    logger.info(f"Total Timesteps: {total_timesteps}")
    logger.info(
        f"Hyperparameters: lr={learning_rate}, gamma={gamma}, lambda={lam}, clip={clip_param}, "
        f"entropy={entropy_coef}, vf_coef={vf_coef}, epochs={epochs}, batch_size={batch_size}"
    )
    logger.info(f"Steps per update: {steps_per_update}")

    # ##: Seeding for reproducibility.
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ##: Create environment.
    env = ImgObsWrapper(FullyObsWrapper(Maze()))
    logger.info("Successfully created environment using direct Maze instantiation.")

    # ##: Initialize agent.
    agent = PPOAgent(
        env.observation_space,
        env.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        lam=lam,
        clip_param=clip_param,
        entropy_coef=entropy_coef,
        vf_coef=vf_coef,
        epochs=epochs,
        batch_size=batch_size,
    )

    # ##: Load pre-trained models if specified.
    if load_path:
        logger.info(f"Loading models from {load_path}...")
        agent.load_models(load_path)

    # ##: Prepare for saving models.
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        logger.info(f"Models will be saved to {save_path}_*.keras")

    # ##: Training loop variables.
    current_obs, _ = env.reset(seed=seed)
    logger.info(current_obs)
    episode_reward = 0
    episode_length = 0
    total_episodes = 0
    update_cycle = 0

    # ##: Logging setup.
    reward_deque = deque(maxlen=100)
    length_deque = deque(maxlen=100)
    start_time = time.time()

    logger.info("Starting training loop...")
    for timestep in range(0, total_timesteps, steps_per_update):
        update_cycle += 1
        logger.debug(f"Update Cycle {update_cycle} | Timestep {timestep}/{total_timesteps}")

        last_obs_for_bootstrap = None

        for step in range(steps_per_update):
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
                # ##: If loop finishes without done, store the last observation for bootstrapping.
                if step == steps_per_update - 1:
                    last_obs_for_bootstrap = next_obs

        # ##: Perform learning update.
        logger.debug(f"Performing learning update for cycle {update_cycle}...")
        agent.learn(last_state=last_obs_for_bootstrap)
        logger.debug("Learning update complete.")

        # ##: Logging.
        if update_cycle % log_interval == 0 and len(reward_deque) > 0:
            mean_reward = np.mean(reward_deque)
            mean_length = np.mean(length_deque)
            elapsed_time = time.time() - start_time
            fps = int((timestep + steps_per_update) / elapsed_time) if elapsed_time > 0 else 0
            logger.info(f"Timesteps: {timestep + steps_per_update}/{total_timesteps} | Episodes: {total_episodes}")
            logger.info(f"Mean Reward (last 100): {mean_reward:.2f} | Mean Length (last 100): {mean_length:.1f}")
            logger.info(f"FPS: {fps} | Elapsed Time: {elapsed_time:.2f}s")

        # ##: Save models periodically.
        if save_path and update_cycle % save_interval == 0:
            logger.info(f"Saving models at timestep {timestep + steps_per_update}...")
            agent.save_models(save_path)

    logger.info("Training finished.")
    env.close()

    # ##: Final model save.
    if save_path:
        logger.info("Saving final models...")
        agent.save_models(save_path + "_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on MiniGrid Maze.")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Maze-Custom-v0", help="Gymnasium Environment ID")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--steps-per-update", type=int, default=2048, help="Steps collected per agent update")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--entropy", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=1, help="Log progress every N updates")
    parser.add_argument("--save-interval", type=int, default=10, help="Save models every N updates")
    parser.add_argument("--save-path", type=str, default="models/ppo_maze", help="Path prefix to save models")
    parser.add_argument("--load-path", type=str, default=None, help="Path prefix to load pre-trained models")

    args = parser.parse_args()

    train(
        total_timesteps=args.total_timesteps,
        steps_per_update=args.steps_per_update,
        learning_rate=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_param=args.clip,
        entropy_coef=args.entropy,
        vf_coef=args.vf_coef,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_path=args.save_path,
        load_path=args.load_path,
    )
