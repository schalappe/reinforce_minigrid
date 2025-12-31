"""
Main training script for RL agents on the MiniGrid Maze environment.

Supports multiple algorithms:
- PPO: On-policy with curriculum learning, RND, and hybrid exploration
- DQN: Off-policy Rainbow DQN with six enhancements
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
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from maze.envs.base_maze import BaseMaze
from maze.envs.easy_maze import EasyMaze
from maze.envs.hard_maze import HardMaze
from maze.envs.maze import Maze
from maze.envs.medium_maze import MediumMaze
from reinforce import setup_logger
from reinforce.config.config_loader import load_config
from reinforce.config.training_config import Algorithm, MainConfig
from reinforce.factory import create_agent


class CurriculumStage(TypedDict):
    """Type definition for a curriculum stage configuration."""

    name: str
    env_class: type[Maze]
    threshold: float | None
    max_steps: int | None
    entropy_multiplier: float


# ##>: Define the curriculum with per-stage entropy multipliers.
CURRICULUM: list[CurriculumStage] = [
    {"name": "BaseMaze", "env_class": BaseMaze, "threshold": 0.5, "max_steps": 500_000, "entropy_multiplier": 1.0},
    {"name": "EasyMaze", "env_class": EasyMaze, "threshold": 1.0, "max_steps": 500_000, "entropy_multiplier": 1.5},
    {
        "name": "MediumMaze",
        "env_class": MediumMaze,
        "threshold": 2.0,
        "max_steps": 1_000_000,
        "entropy_multiplier": 2.0,
    },
    {"name": "HardMaze", "env_class": HardMaze, "threshold": None, "max_steps": None, "entropy_multiplier": 3.0},
]


def create_env(env_class: type[Maze]) -> Callable[[], Env[Any, Any]]:
    """
    Create and wrap the environment for RL training.

    Uses RGBImgPartialObsWrapper to convert observations to 56x56x3 RGB pixels
    suitable for CNN processing.

    Parameters
    ----------
    env_class : type[Maze]
        The environment class to instantiate.

    Returns
    -------
    Callable[[], Env]
        Factory function that creates a wrapped environment.
    """

    def make_env() -> Env[Any, Any]:
        env = env_class(render_mode="rgb_array")
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

    return make_env


def _train_ppo(config: MainConfig) -> None:
    """
    PPO training loop with curriculum learning and exploration enhancements.

    Parameters
    ----------
    config : MainConfig
        Training configuration.
    """
    from reinforce.ppo.agent import PPOAgent
    from reinforce.ppo.exploration import ExplorationManager
    from reinforce.ppo.rnd import RNDModule

    num_envs = config.training.num_envs
    logger.info(f"Starting PPO training with {num_envs} parallel environments...")

    # ##>: Curriculum setup.
    current_curriculum_index = 0
    current_stage = CURRICULUM[current_curriculum_index]
    logger.info(f"Starting curriculum stage 1: {current_stage['name']}")

    # ##>: Create vectorized environment.
    env_fns = [create_env(current_stage["env_class"]) for _ in range(num_envs)]
    context_method = "fork"
    try:
        mp.get_context("fork")
    except ValueError:
        context_method = "spawn"
        logger.warning("Fork context not available, falling back to spawn.")

    env = AsyncVectorEnv(env_fns, context=context_method)

    # ##>: Initialize agent.
    agent: PPOAgent = create_agent(config, env.single_observation_space, env.single_action_space)  # type: ignore

    # ##>: Initialize RND module.
    rnd_module: RNDModule | None = None
    if config.rnd.enabled:
        obs_shape = env.single_observation_space.shape
        if obs_shape is not None:
            rnd_module = RNDModule(
                input_shape=obs_shape,
                feature_dim=config.rnd.feature_dim,
                learning_rate=config.rnd.learning_rate,
                intrinsic_reward_scale=config.rnd.intrinsic_reward_scale,
                update_proportion=config.rnd.update_proportion,
            )
            logger.info("RND module initialized.")

    # ##>: Initialize exploration manager.
    exploration_manager = ExplorationManager(
        num_actions=agent.num_actions,
        num_envs=num_envs,
        config=config.exploration,
        base_entropy_coef=config.ppo.entropy_coef,
    )

    # ##>: Load pre-trained models if specified.
    if config.logging.load_path:
        agent.load_models(config.logging.load_path)

    # ##>: Prepare for saving.
    if config.logging.save_path:
        save_dir = os.path.dirname(config.logging.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # ##>: Training loop.
    current_obs, _ = env.reset(seed=config.environment.seed)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    total_episodes = 0
    update_cycle = 0
    total_steps_overall = 0
    steps_in_current_stage = 0

    reward_deque: deque[float] = deque(maxlen=100 * num_envs)
    length_deque: deque[int] = deque(maxlen=100 * num_envs)
    start_time = time.time()

    while total_steps_overall < config.training.total_timesteps:
        update_cycle += 1
        steps_per_env = config.training.steps_per_update
        collected_steps_this_update = 0
        last_obs_for_bootstrap = None
        intrinsic_rewards_this_update: list[float] = []

        for step in range(steps_per_env):
            if total_steps_overall >= config.training.total_timesteps:
                break

            actions, info = agent.get_action(current_obs)
            values = info["values"]
            action_probs = info["log_probs"]

            explored_actions = exploration_manager.apply_exploration(
                action_logits=np.zeros((num_envs, agent.num_actions)),
                sampled_actions=actions,
            )
            exploration_manager.update_action_counts(explored_actions)

            next_obs_batch, rewards_batch, terminated_batch, truncated_batch, _ = env.step(explored_actions)

            total_rewards = rewards_batch.copy()
            if rnd_module is not None:
                intrinsic_rewards = rnd_module.compute_intrinsic_reward(next_obs_batch)
                total_rewards = rewards_batch + config.rnd.intrinsic_reward_coef * intrinsic_rewards
                intrinsic_rewards_this_update.extend(intrinsic_rewards.tolist())

            dones_batch = np.logical_or(terminated_batch, truncated_batch)
            agent.store_transition(current_obs, explored_actions, total_rewards, values, dones_batch, action_probs)

            current_obs = next_obs_batch
            exploration_manager.step(num_envs)

            episode_rewards += rewards_batch
            episode_lengths += 1
            total_steps_overall += num_envs
            steps_in_current_stage += num_envs
            collected_steps_this_update += num_envs

            for i, done in enumerate(dones_batch):
                if done:
                    reward_deque.append(float(episode_rewards[i]))
                    length_deque.append(int(episode_lengths[i]))
                    total_episodes += 1
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    exploration_manager.reset_counts_for_env(i)

                    # ##>: Check curriculum advancement.
                    if len(reward_deque) == reward_deque.maxlen:
                        mean_reward = np.mean(list(reward_deque))
                        stage_threshold = current_stage["threshold"]
                        stage_max_steps = current_stage["max_steps"]

                        advance = False
                        if stage_threshold is not None and mean_reward >= stage_threshold:
                            logger.info(f"Stage threshold met: {mean_reward:.2f} >= {stage_threshold}")
                            advance = True
                        elif stage_max_steps is not None and steps_in_current_stage >= stage_max_steps:
                            logger.info(f"Stage max steps reached: {steps_in_current_stage}")
                            advance = True

                        if advance and current_curriculum_index < len(CURRICULUM) - 1:
                            current_curriculum_index += 1
                            current_stage = CURRICULUM[current_curriculum_index]
                            logger.warning(
                                f"Advancing to stage {current_curriculum_index + 1}: {current_stage['name']}"
                            )

                            env.close()
                            env_fns = [create_env(current_stage["env_class"]) for _ in range(num_envs)]
                            env = AsyncVectorEnv(env_fns, context=context_method)
                            current_obs, _ = env.reset()
                            episode_rewards.fill(0)
                            episode_lengths.fill(0)
                            reward_deque.clear()
                            length_deque.clear()
                            steps_in_current_stage = 0
                            break

            if step == steps_per_env - 1:
                last_obs_for_bootstrap = next_obs_batch

        if collected_steps_this_update > 0:
            if rnd_module is not None:
                collected_obs = agent.buffer.states[: agent.buffer.ptr]
                if len(collected_obs) > 0:
                    obs_flat = collected_obs.reshape(-1, *collected_obs.shape[2:])
                    rnd_module.train_step(obs_flat)

            metrics = agent.learn(last_state=last_obs_for_bootstrap, steps_collected=collected_steps_this_update)

            current_entropy = metrics.get("entropy", 0.0)
            base_entropy_coef = exploration_manager.update_entropy_coef(current_entropy)
            agent.entropy_coef = base_entropy_coef * current_stage["entropy_multiplier"]

            if update_cycle % config.logging.log_interval == 0:
                elapsed = time.time() - start_time
                fps = int(total_steps_overall / elapsed) if elapsed > 0 else 0
                logger.info(
                    f"Update {update_cycle} | Steps: {total_steps_overall} | FPS: {fps} | "
                    f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | "
                    f"Value Loss: {metrics.get('value_loss', 0):.4f}"
                )

        if config.logging.save_path and update_cycle % config.logging.save_interval == 0:
            save_path = f"{config.logging.save_path}_ts{total_steps_overall}_stage{current_curriculum_index}"
            agent.save_models(save_path)

    env.close()
    if config.logging.save_path:
        agent.save_models(f"{config.logging.save_path}_final")
    logger.info("PPO training complete.")


def _train_dqn(config: MainConfig) -> None:
    """
    Rainbow DQN training loop.

    Parameters
    ----------
    config : MainConfig
        Training configuration.
    """
    from reinforce.dqn.agent import RainbowAgent

    num_envs = config.training.num_envs
    logger.info(f"Starting Rainbow DQN training with {num_envs} environments...")

    # ##>: Create environment (DQN typically uses fewer parallel envs).
    env_fns = [create_env(BaseMaze) for _ in range(num_envs)]
    context_method = "fork"
    try:
        mp.get_context("fork")
    except ValueError:
        context_method = "spawn"

    env = AsyncVectorEnv(env_fns, context=context_method)

    # ##>: Initialize agent.
    agent: RainbowAgent = create_agent(config, env.single_observation_space, env.single_action_space)  # type: ignore

    # ##>: Load pre-trained models if specified.
    if config.logging.load_path:
        agent.load_models(config.logging.load_path)

    # ##>: Prepare for saving.
    if config.logging.save_path:
        save_dir = os.path.dirname(config.logging.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # ##>: Training loop.
    current_obs, _ = env.reset(seed=config.environment.seed)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    total_episodes = 0
    total_steps = 0

    reward_deque: deque[float] = deque(maxlen=100)
    length_deque: deque[int] = deque(maxlen=100)
    start_time = time.time()

    while total_steps < config.training.total_timesteps:
        actions, _ = agent.get_action(current_obs)
        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        dones = np.logical_or(terminated, truncated)

        # ##>: Store transitions in batch (vectorized, much faster than loop).
        agent.store_transitions_batch(current_obs, actions, rewards, next_obs, dones)

        # ##>: Learn (may run multiple times based on train_freq).
        metrics = agent.learn()

        current_obs = next_obs
        episode_rewards += rewards
        episode_lengths += 1
        total_steps += num_envs

        # ##>: Handle episode ends.
        for i, done in enumerate(dones):
            if done:
                reward_deque.append(float(episode_rewards[i]))
                length_deque.append(int(episode_lengths[i]))
                total_episodes += 1
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                # ##>: Only flush n-step buffer if using single-transition storage (not batch mode).
                if agent.has_pending_n_step_transitions:
                    agent.on_episode_end()

        # ##>: Logging.
        if total_steps % (config.logging.log_interval * 1000) == 0 and len(reward_deque) > 0:
            mean_reward = np.mean(list(reward_deque))
            elapsed = time.time() - start_time
            fps = int(total_steps / elapsed) if elapsed > 0 else 0
            logger.info(
                f"Steps: {total_steps} | Episodes: {total_episodes} | "
                f"Mean Reward: {mean_reward:.2f} | FPS: {fps} | "
                f"Loss: {metrics.get('loss', 0):.4f}"
            )

        # ##>: Save.
        if config.logging.save_path and total_steps % (config.logging.save_interval * 10000) == 0:
            agent.save_models(f"{config.logging.save_path}_ts{total_steps}")

    env.close()
    if config.logging.save_path:
        agent.save_models(f"{config.logging.save_path}_final")
    logger.info("Rainbow DQN training complete.")


def train(config: MainConfig) -> None:
    """
    Main training function that dispatches to algorithm-specific trainers.

    Parameters
    ----------
    config : MainConfig
        Training configuration including algorithm selection.
    """
    setup_logger()
    logger.info(f"Training with algorithm: {config.algorithm.value}")

    np.random.seed(config.environment.seed)
    tf.random.set_seed(config.environment.seed)

    if config.algorithm == Algorithm.PPO:
        _train_ppo(config)
    elif config.algorithm == Algorithm.DQN:
        _train_dqn(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL agent on MiniGrid Maze.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default_training.yaml", help="Path to YAML config.")
    parser.add_argument("--algorithm", type=str, choices=["ppo", "dqn"], default=None, help="RL algorithm to use.")

    # ##>: Common overrides.
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps.")
    parser.add_argument("--steps-per-update", type=int, default=None, help="Steps per update.")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--log-interval", type=int, default=None, help="Log interval.")
    parser.add_argument("--save-interval", type=int, default=None, help="Save interval.")
    parser.add_argument("--save-path", type=str, default=None, help="Model save path.")
    parser.add_argument("--load-path", type=str, default=None, help="Model load path.")

    # ##>: PPO-specific overrides.
    parser.add_argument("--lr", type=float, default=None, help="PPO learning rate.")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor.")
    parser.add_argument("--lam", type=float, default=None, help="GAE lambda.")
    parser.add_argument("--clip", type=float, default=None, help="PPO clip parameter.")
    parser.add_argument("--entropy", type=float, default=None, help="Entropy coefficient.")
    parser.add_argument("--vf-coef", type=float, default=None, help="Value function coefficient.")
    parser.add_argument("--epochs", type=int, default=None, help="PPO epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Max gradient norm.")
    parser.add_argument("--no-lr-annealing", action="store_true", help="Disable LR annealing.")
    parser.add_argument("--use-value-clipping", action="store_true", help="Enable value clipping.")

    # ##>: DQN-specific overrides.
    parser.add_argument("--dqn-lr", type=float, default=None, help="DQN learning rate.")
    parser.add_argument("--n-step", type=int, default=None, help="N-step returns.")
    parser.add_argument("--buffer-size", type=int, default=None, help="Replay buffer size.")
    parser.add_argument("--target-update-freq", type=int, default=None, help="Target network update frequency.")
    parser.add_argument("--learning-starts", type=int, default=None, help="Steps before learning starts.")
    parser.add_argument("--no-noisy", action="store_true", help="Disable noisy networks.")
    parser.add_argument("--no-dueling", action="store_true", help="Disable dueling architecture.")
    parser.add_argument("--no-double", action="store_true", help="Disable double Q-learning.")
    parser.add_argument("--no-per", action="store_true", help="Disable prioritized replay.")

    args = parser.parse_args()
    configuration = load_config(config_path=args.config, args=args)
    train(configuration)
