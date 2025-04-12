# -*- coding: utf-8 -*-
"""
Evaluate an agent in an environment.
"""

from statistics import mean
from typing import Any, Dict

from loguru import logger

from reinforce.agents import BaseAgent
from reinforce.environments import BaseEnvironment
from reinforce.utils.management import setup_logger

setup_logger()


def evaluate_agent(
    agent: BaseAgent, environment: BaseEnvironment, num_episodes: int, max_steps_per_episode: int
) -> Dict[str, Any]:
    """
    Evaluate the agent in the environment.

    Parameters
    ----------
    agent : BaseAgent
        The agent to evaluate.
    environment : BaseEnvironment
        The environment to evaluate in.
    num_episodes : int
        Number of episodes to run for evaluation.
    max_steps_per_episode : int
        Maximum number of steps allowed per evaluation episode.

    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation metrics including mean_reward, max_reward,
        min_reward, mean_steps, and the list of rewards per episode.
    """
    eval_rewards = []
    eval_steps = []
    logger.info(f"Starting evaluation for {num_episodes} episodes...")

    for episode_num in range(num_episodes):
        observation = environment.reset()
        episode_reward = 0
        episode_steps = 0

        # ##: Run one evaluation episode.
        for _ in range(max_steps_per_episode):
            try:
                action, _ = agent.act(observation, training=False)
            except Exception as e:
                logger.error(f"Error getting action from agent during evaluation: {e}", exc_info=True)
                break

            try:
                next_observation, reward, done, _ = environment.step(action)
            except Exception as e:
                logger.error(f"Error stepping environment during evaluation: {e}", exc_info=True)
                break

            observation = next_observation
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        eval_rewards.append(episode_reward)
        eval_steps.append(episode_steps)
        logger.info(
            f"Evaluation Episode {episode_num + 1}/{num_episodes} finished. "
            f"Reward: {episode_reward}, Steps: {episode_steps}"
        )

    # ##: End of evaluation loop.
    mean_reward = mean(eval_rewards) if eval_rewards else 0
    max_reward = max(eval_rewards) if eval_rewards else 0
    min_reward = min(eval_rewards) if eval_rewards else 0
    mean_steps = mean(eval_steps) if eval_steps else 0

    logger.info(
        f"Evaluation finished. "
        f"Mean Reward: {mean_reward:.2f} (Min: {min_reward:.2f}, "
        f"Max: {max_reward:.2f}), "
        f"Mean Steps: {mean_steps:.1f}"
    )

    return {
        "mean_reward": mean_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
        "mean_steps": mean_steps,
        "rewards": eval_rewards,
    }
