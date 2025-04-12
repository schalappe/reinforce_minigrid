# -*- coding: utf-8 -*-
"""
Trainer for Proximal Policy Optimization (PPO).
"""

from collections import deque
from datetime import datetime
from statistics import mean
from time import time
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from loguru import logger

from reinforce.agents.actor_critic import PPOAgent
from reinforce.configs.models import PPOTrainerConfig
from reinforce.environments import BaseEnvironment
from reinforce.learning.evaluation import evaluate_agent
from reinforce.learning.trainers.base_trainer import BaseTrainer
from reinforce.utils.buffers import RolloutBuffer
from reinforce.utils.logger import BaseLogger, setup_logger
from reinforce.utils.persistence import save_checkpoint

setup_logger()


class PPOTrainer(BaseTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO).

    This trainer implements the PPO training loop:
    1. Collect rollouts (N steps) using the agent and environment.
    2. Compute advantages and returns using GAE via the RolloutBuffer.
    3. Perform multiple epochs of minibatch updates on the collected data.
    4. Repeat.
    """

    def __init__(
        self, *, agent: PPOAgent, environment: BaseEnvironment, config: PPOTrainerConfig, logger_instance: BaseLogger
    ):
        """
        Initialize the PPO trainer.

        Parameters
        ----------
        agent : PPOAgent
            The PPO agent to train.
        environment : BaseEnvironment
            The environment to train in.
        config : PPOTrainerConfig
            Pydantic configuration model for the PPO trainer.
        logger_instance : BaseLogger
            Logger instance for experiment tracking.
        """
        self.agent = agent
        self.environment = environment
        self.logger = logger_instance
        self.config = config

        # ##: Ensure the save directory exists (keep this).
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        # ##: Initialize training state.
        self.total_steps = 0
        self.episode = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps_deque = deque(maxlen=100)

        # ##: Initialize the Rollout Buffer.
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.config.n_steps,
            observation_shape=self.environment.observation_space.shape,
            action_shape=(),
            gamma=self.config.gamma,
            gae_lambda=self.agent.hyperparameters.gae_lambda,
        )

        # ##: Timestamp for saving models.
        self.timestamp = int(datetime.now().timestamp())

        # ##: Log hyperparameters using injected logger.
        self.logger.log_params(
            self.config.model_dump(exclude={"trial_info", "pruning_callback", "trainer_type"}), prefix="trainer"
        )
        # ##: Also log agent hyperparameters.
        self.logger.log_params(self.agent.hyperparameters.model_dump(), prefix="agent")

    def train(self) -> Dict[str, Any]:
        """
        Train the PPO agent in the environment.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics.
        """
        start_time = time()
        max_total_steps = self.config.max_total_steps

        # ##: Initial environment reset.
        observation = self.environment.reset()
        current_episode_reward = 0
        current_episode_steps = 0

        while self.total_steps < max_total_steps:
            # ##: --- Rollout Phase ---
            self.rollout_buffer.clear()

            for _ in range(self.config.n_steps):
                self.total_steps += 1
                current_episode_steps += 1

                # ##: Get action, value, log_prob from agent.
                action, agent_info = self.agent.act(observation, training=True)
                value = agent_info["value"]
                log_prob = agent_info["log_prob"]

                # ##: Step the environment.
                next_observation, reward, done, _ = self.environment.step(action)

                # ##: Store experience in the buffer (raw observation).
                self.rollout_buffer.add(observation, action, reward, done, value, log_prob)

                # ##: Update current state.
                observation = next_observation
                current_episode_reward += reward

                # ##: Handle episode termination within the rollout.
                if done:
                    self.episode += 1
                    self.episode_rewards.append(current_episode_reward)
                    self.episode_steps_deque.append(current_episode_steps)
                    mean_reward_100 = mean(self.episode_rewards) if self.episode_rewards else 0

                    # ##: Log episode metrics directly using logger.
                    self.logger.log_metrics(
                        {
                            "episode_reward": current_episode_reward,
                            "episode_steps": current_episode_steps,
                            "mean_reward_100": mean_reward_100,
                        },
                        step=self.total_steps,
                        epoch=self.episode,  # Use current episode count
                        context={"subset": "train"},
                    )
                    # ##: Log progress to console periodically.
                    if self.episode % self.config.log_frequency == 0:
                        elapsed_time = time() - start_time
                        logger.info(
                            f"Step {self.total_steps}/{self.config.max_total_steps} | Ep {self.episode} "
                            f"| Ep Steps: {current_episode_steps} "
                            f"| Ep Reward: {current_episode_reward:.2f} "
                            f"| Mean Rwd (100): {mean_reward_100:.2f} "
                            f"| Time: {elapsed_time:.2f}s"
                        )

                    # ##: Handle evaluation and pruning.
                    if self.episode > 0 and self.episode % self.config.eval_frequency == 0:
                        eval_metrics = evaluate_agent(
                            agent=self.agent,
                            environment=self.environment,
                            num_episodes=self.config.num_eval_episodes,
                            max_steps_per_episode=self.config.max_steps_per_episode,
                        )
                        mean_eval_reward = eval_metrics["mean_reward"]
                        min_r, max_r = eval_metrics["min_reward"], eval_metrics["max_reward"]

                        logger.info(
                            f"Evaluation (Ep {self.episode}, Step {self.total_steps}) Mean Reward: {mean_eval_reward:.2f} "
                            f"(Min: {min_r:.2f}, Max: {max_r:.2f})"
                        )

                        # ##: Log evaluation metrics using injected logger.
                        self.logger.log_metrics(
                            {
                                "eval_mean_reward": mean_eval_reward,
                                "eval_max_reward": eval_metrics["max_reward"],
                                "eval_min_reward": eval_metrics["min_reward"],
                            },
                            step=self.total_steps,
                            epoch=self.episode,
                            context={"subset": "eval"},
                        )

                        # ##: Report to Optuna for pruning (keep pruning logic here).
                        if self.config.pruning_callback is not None and self.config.trial_info is not None:
                            try:
                                self.config.pruning_callback(self.episode, mean_eval_reward)  # Use episode count
                            except Exception as exc:
                                logger.warning(f"Trial pruned at episode {self.episode}: {exc}")
                                return {"pruned": True, "total_steps": self.total_steps, "episode": self.episode}

                    # ##: Reset environment and episode trackers.
                    observation = self.environment.reset()
                    current_episode_reward = 0
                    current_episode_steps = 0

                # ##: Check if max total steps reached during rollout.
                if self.total_steps >= max_total_steps:
                    break

            # ##: --- Update Phase ---
            # ##: Compute GAE and returns. Need value of the last state s_N.
            with tf.device("/cpu:0"):
                _, last_agent_info = self.agent.act(observation, training=False)
            last_value = last_agent_info["value"]
            last_done = False

            self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

            # ##: Perform PPO updates over multiple epochs.
            all_update_metrics = []
            for epoch in range(self.config.n_epochs):
                for batch_data in self.rollout_buffer.sample_mini_batches(
                    batch_size=self.config.batch_size, n_epochs=1
                ):
                    update_metrics = self.agent.learn(batch_data)
                    all_update_metrics.append(update_metrics)

            # ##: Log aggregated update metrics for the rollout using logger.
            if all_update_metrics:
                aggregated_metrics = {}
                keys = all_update_metrics[0].keys()
                for key in keys:
                    # ##: Handle potential tensor values by converting to numpy.
                    values = [
                        m[key].numpy() if hasattr(m[key], "numpy") else m[key] for m in all_update_metrics if key in m
                    ]
                    if values:
                        aggregated_metrics[f"update_{key}_mean"] = np.mean(values)
                # ##: Log aggregated metrics to AIM.
                self.logger.log_metrics(
                    aggregated_metrics,
                    step=self.total_steps,
                    epoch=self.episode,
                    context={"subset": "train_update"},
                )

            # ##: Save checkpoint periodically using CheckpointManager.
            # Consider adding step-based saving frequency config option if needed.
            if self.episode > 0 and self.episode % self.config.save_frequency == 0:
                checkpoint_path_base = (
                    self.config.save_dir
                    / f"{self.agent.name}_{self.environment.name}_ppo_{self.timestamp}_{self.episode}"
                )
                trainer_state = {
                    "episode": self.episode,
                    "total_steps": self.total_steps,
                    "timestamp": self.timestamp,
                }
                save_checkpoint(
                    agent=self.agent,
                    save_path_base=checkpoint_path_base,
                    trainer_state=trainer_state,
                    logger_instance=self.logger,
                )

        # ##: Final evaluation after training loop using Evaluator.
        final_eval_metrics = evaluate_agent(
            agent=self.agent,
            environment=self.environment,
            num_episodes=self.config.num_eval_episodes,
            max_steps_per_episode=self.config.max_steps_per_episode,
        )
        logger.info(f"Final Evaluation Mean Reward: {final_eval_metrics['mean_reward']:.2f}")
        self.logger.log_metrics(
            {f"final_{k}": v for k, v in final_eval_metrics.items() if k != "rewards"},
            step=self.total_steps,
            epoch=self.episode,
            context={"subset": "final_eval"},  # Add context
        )
        self.logger.log_params({"final_total_steps": self.total_steps, "final_episodes": self.episode})

        return {
            "episodes": self.episode,
            "total_steps": self.total_steps,
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
            "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": min(self.episode_rewards) if self.episode_rewards else 0,
        }
