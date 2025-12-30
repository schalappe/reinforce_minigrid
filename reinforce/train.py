"""
Main training script for the PPO agent on the MiniGrid Maze environment.

Implements curriculum learning with modern PPO enhancements including
learning rate annealing, gradient clipping, and IMPALA-style networks.
"""

import argparse
import multiprocessing as mp
import os
import time
from collections import deque
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import tensorflow as tf
from gymnasium import Env
from gymnasium.vector import AsyncVectorEnv
from loguru import logger
from minigrid.wrappers import ImgObsWrapper

from maze.envs.base_maze import BaseMaze
from maze.envs.easy_maze import EasyMaze
from maze.envs.hard_maze import HardMaze
from maze.envs.maze import Maze
from maze.envs.medium_maze import MediumMaze
from reinforce import setup_logger
from reinforce.agent import PPOAgent
from reinforce.config import MainConfig, load_config


class CurriculumStage(TypedDict):
    """Type definition for a curriculum stage configuration."""

    name: str
    env_class: type[Maze]
    threshold: float | None
    max_steps: int | None


# ##>: Define the curriculum with adjusted thresholds for IMPALA network.
CURRICULUM: list[CurriculumStage] = [
    {"name": "BaseMaze", "env_class": BaseMaze, "threshold": 0.5, "max_steps": 500_000},
    {"name": "EasyMaze", "env_class": EasyMaze, "threshold": 1.0, "max_steps": 500_000},
    {"name": "MediumMaze", "env_class": MediumMaze, "threshold": 2.0, "max_steps": 1_000_000},
    {"name": "HardMaze", "env_class": HardMaze, "threshold": None, "max_steps": None},
]


def create_env(env_class: type[Maze]) -> Callable[[], Env[Any, Any]]:
    """
    Helper function to create and wrap the environment.

    Parameters
    ----------
    env_class : type[BaseMaze]
        The environment class to instantiate and wrap.

    Returns
    -------
    Callable[[], Env]
        A factory function that creates and returns a wrapped environment.
    """

    def make_env() -> Env[Any, Any]:
        return ImgObsWrapper(env_class(render_mode="rgb_array"))

    return make_env


def train(config: MainConfig) -> None:
    """
    Trains the PPO agent based on the provided configuration using curriculum learning.

    Parameters
    ----------
    config : MainConfig
        A dataclass object containing all necessary configuration parameters
        (environment, training, PPO hyperparameters, logging).
    """
    setup_logger()
    logger.info("Starting PPO training with modern enhancements and curriculum learning...")
    num_envs = config.training.num_envs
    logger.info(f"Number of parallel environments: {num_envs}")
    logger.info(f"Environment Seed: {config.environment.seed}")
    logger.info(f"Total Timesteps (Target): {config.training.total_timesteps}")
    logger.info(f"Steps per Update (per env): {config.training.steps_per_update}")
    logger.info(f"Total Steps per Update (all envs): {config.training.steps_per_update * num_envs}")
    logger.info(
        f"PPO Hyperparameters: lr={config.ppo.learning_rate}, gamma={config.ppo.gamma}, "
        f"lambda={config.ppo.lambda_gae}, clip={config.ppo.clip_param}, "
        f"entropy={config.ppo.entropy_coef}, vf_coef={config.ppo.value_coef}, "
        f"epochs={config.ppo.epochs}, batch_size={config.ppo.batch_size}"
    )
    logger.info(
        f"Modern Enhancements: max_grad_norm={config.ppo.max_grad_norm}, "
        f"lr_annealing={config.ppo.use_lr_annealing}, value_clipping={config.ppo.use_value_clipping}"
    )
    logger.info(
        f"Logging: Log Interval={config.logging.log_interval}, Save Interval={config.logging.save_interval}, "
        f"Save Path={config.logging.save_path}, Load Path={config.logging.load_path}"
    )

    # ##>: Seeding for reproducibility.
    np.random.seed(config.environment.seed)
    tf.random.set_seed(config.environment.seed)

    # ##>: Curriculum setup.
    current_curriculum_index = 0
    current_stage = CURRICULUM[current_curriculum_index]
    stage_name = current_stage["name"]
    logger.info(f"Starting curriculum stage 1: {stage_name}")

    # ##>: Create vectorized environment.
    env_fns = [create_env(current_stage["env_class"]) for _ in range(num_envs)]

    # ##>: Determine the appropriate multiprocessing context.
    context_method = "fork"
    try:
        mp.get_context("fork")
    except ValueError:
        context_method = "spawn"
        logger.warning("Fork context not available, falling back to spawn context. This might be slower.")

    env = AsyncVectorEnv(env_fns, context=context_method)
    logger.info(f"Using {type(env).__name__} with context '{context_method}'")

    # ##>: Initialize agent with new parameters.
    agent = PPOAgent(
        env.single_observation_space,
        env.single_action_space,
        learning_rate=config.ppo.learning_rate,
        gamma=config.ppo.gamma,
        lam=config.ppo.lambda_gae,
        clip_param=config.ppo.clip_param,
        entropy_coef=config.ppo.entropy_coef,
        vf_coef=config.ppo.value_coef,
        epochs=config.ppo.epochs,
        batch_size=config.ppo.batch_size,
        num_envs=num_envs,
        steps_per_update=config.training.steps_per_update,
        max_grad_norm=config.ppo.max_grad_norm,
        total_timesteps=config.training.total_timesteps,
        use_lr_annealing=config.ppo.use_lr_annealing,
        use_value_clipping=config.ppo.use_value_clipping,
    )

    # ##>: Load pre-trained models if specified.
    if config.logging.load_path:
        logger.info(f"Loading models from {config.logging.load_path}...")
        agent.load_models(config.logging.load_path)

    # ##>: Prepare for saving models.
    if config.logging.save_path:
        save_dir = os.path.dirname(config.logging.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Models will be saved to {config.logging.save_path}_*.keras")

    # ##>: Training loop variables for vectorized envs.
    current_obs, _ = env.reset(seed=config.environment.seed)

    # ##>: Track rewards and lengths per environment.
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    total_episodes = 0
    update_cycle = 0
    total_steps_overall = 0
    steps_in_current_stage = 0

    # ##>: Logging setup (track stats across all envs).
    reward_deque: deque[float] = deque(maxlen=100 * num_envs)
    length_deque: deque[int] = deque(maxlen=100 * num_envs)
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

        # ##>: Collect experience from parallel environments.
        steps_per_env = config.training.steps_per_update
        collected_steps_this_update = 0
        last_obs_for_bootstrap = None

        for step in range(steps_per_env):
            if total_steps_overall >= config.training.total_timesteps:
                break

            # ##>: Get actions, values, log probs for the batch of observations.
            actions, values, action_probs = agent.get_action(current_obs)

            # ##>: Step the vectorized environment.
            next_obs_batch, rewards_batch, terminated_batch, truncated_batch, _ = env.step(actions)

            # ##>: Store transitions for the batch.
            dones_batch = np.logical_or(terminated_batch, truncated_batch)
            agent.store_transition(current_obs, actions, rewards_batch, values, dones_batch, action_probs)

            # ##>: Update current observations.
            current_obs = next_obs_batch

            # ##>: Update episode trackers for each environment.
            episode_rewards += rewards_batch
            episode_lengths += 1
            total_steps_overall += num_envs
            steps_in_current_stage += num_envs
            collected_steps_this_update += num_envs

            # ##>: Handle episode ends for each environment.
            for i, done in enumerate(dones_batch):
                if done:
                    finished_reward = episode_rewards[i]
                    finished_length = episode_lengths[i]
                    reward_deque.append(float(finished_reward))
                    length_deque.append(int(finished_length))
                    total_episodes += 1
                    stage_name = current_stage["name"]
                    logger.debug(
                        f"Env {i} | Episode {total_episodes} finished. Reward: {finished_reward:.2f}, "
                        f"Length: {finished_length}, Stage: {stage_name}"
                    )

                    # ##>: Reset individual trackers.
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0

                    # ##>: Check curriculum advancement criteria.
                    if len(reward_deque) == reward_deque.maxlen:
                        mean_reward = np.mean(list(reward_deque))
                        stage_threshold = current_stage["threshold"]
                        stage_max_steps = current_stage["max_steps"]
                        advance_curriculum = False
                        stage_name = current_stage["name"]

                        if stage_threshold is not None and mean_reward >= stage_threshold:
                            logger.info(
                                f"Stage {stage_name} threshold ({stage_threshold}) met "
                                f"with avg reward {mean_reward:.2f} across {num_envs} envs."
                            )
                            advance_curriculum = True
                        elif stage_max_steps is not None and steps_in_current_stage >= stage_max_steps:
                            logger.info(f"Stage {stage_name} max steps ({stage_max_steps}) reached.")
                            advance_curriculum = True

                        if advance_curriculum and current_curriculum_index < len(CURRICULUM) - 1:
                            current_curriculum_index += 1
                            current_stage = CURRICULUM[current_curriculum_index]
                            stage_name = current_stage["name"]
                            logger.warning(
                                f"Advancing to curriculum stage {current_curriculum_index + 1}: {stage_name}"
                            )

                            # ##>: Close old envs and create new ones for the next stage.
                            env.close()
                            env_fns = [create_env(current_stage["env_class"]) for _ in range(num_envs)]
                            env = AsyncVectorEnv(env_fns, context=context_method)

                            current_obs, _ = env.reset()
                            episode_rewards.fill(0)
                            episode_lengths.fill(0)
                            reward_deque.clear()
                            length_deque.clear()
                            steps_in_current_stage = 0
                            last_obs_for_bootstrap = None
                            break

            # ##>: Store last observation for bootstrapping if update cycle ends.
            if step == steps_per_env - 1:
                last_obs_for_bootstrap = next_obs_batch

        # ##>: Perform learning update only if steps were collected in this cycle.
        if collected_steps_this_update > 0:
            logger.info(
                f"Performing learning update for cycle {update_cycle} with {collected_steps_this_update} steps..."
            )
            metrics = agent.learn(last_state=last_obs_for_bootstrap, steps_collected=collected_steps_this_update)
            logger.info(
                f"Update complete | Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f} | Entropy: {metrics['entropy']:.4f} | "
                f"Clip Fraction: {metrics['clip_fraction']:.3f} | KL: {metrics['approx_kl']:.4f}"
            )
        else:
            logger.info(f"Skipping learning update for cycle {update_cycle} as no steps were collected.")

        # ##>: Logging with enhanced metrics.
        if update_cycle % config.logging.log_interval == 0 and len(reward_deque) > 0:
            mean_reward = np.mean(list(reward_deque))
            mean_length = np.mean(list(length_deque))
            elapsed_time = time.time() - start_time
            fps = int(total_steps_overall / elapsed_time) if elapsed_time > 0 else 0
            current_lr = agent.get_current_lr()
            stage_name = current_stage["name"]
            logger.info(
                f"Timesteps: {total_steps_overall}/{config.training.total_timesteps} | "
                f"Episodes: {total_episodes} | Stage: {stage_name}"
            )
            logger.info(
                f"Mean Reward (last {reward_deque.maxlen}): {mean_reward:.2f} | "
                f"Mean Length (last {length_deque.maxlen}): {mean_length:.1f}"
            )
            logger.info(f"FPS: {fps} | LR: {current_lr:.2e} | Elapsed Time: {elapsed_time:.2f}s")

        # ##>: Save models periodically.
        if config.logging.save_path and update_cycle % config.logging.save_interval == 0:
            save_file_path = f"{config.logging.save_path}_ts{total_steps_overall}_stage{current_curriculum_index}"
            logger.info(f"Saving models to {save_file_path}...")
            agent.save_models(save_file_path)

    logger.info("Training finished (total timesteps reached).")
    env.close()

    # ##>: Final model save.
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

    # ##>: Override arguments for all hyperparameters.
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total training timesteps")
    parser.add_argument(
        "--steps-per-update", type=int, default=None, help="Override steps collected per environment per agent update"
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--gamma", type=float, default=None, help="Override discount factor")
    parser.add_argument("--lam", type=float, default=None, help="Override GAE lambda parameter")
    parser.add_argument("--clip", type=float, default=None, help="Override PPO clip parameter")
    parser.add_argument("--entropy", type=float, default=None, help="Override entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=None, help="Override value function loss coefficient")
    parser.add_argument("--epochs", type=int, default=None, help="Override PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=None, help="Override PPO batch size")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Override max gradient norm")
    parser.add_argument("--no-lr-annealing", action="store_true", help="Disable learning rate annealing")
    parser.add_argument("--use-value-clipping", action="store_true", help="Enable value function clipping")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log progress interval (in updates)")
    parser.add_argument("--save-interval", type=int, default=None, help="Override model save interval (in updates)")
    parser.add_argument("--save-path", type=str, default=None, help="Override path prefix to save models")
    parser.add_argument("--load-path", type=str, default=None, help="Override path prefix to load pre-trained models")

    args = parser.parse_args()
    configuration = load_config(config_path=args.config, args=args)

    train(configuration)
