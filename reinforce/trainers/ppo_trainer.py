# -*- coding: utf-8 -*-
"""
Trainer for Proximal Policy Optimization (PPO).
"""

from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import time
from typing import Any, Dict, Union

import numpy as np
import tensorflow as tf
from loguru import logger

from reinforce.agents.actor_critic import PPOAgent
from reinforce.configs.models import PPOTrainerConfig
from reinforce.environments import BaseEnvironment
from reinforce.trainers import BaseTrainer
from reinforce.utils.aim_logger import AimLogger
from reinforce.utils.buffers.rollout_buffer import RolloutBuffer
from reinforce.utils.logging_setup import setup_logger

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
        self, *, agent: PPOAgent, environment: BaseEnvironment, config: PPOTrainerConfig, aim_logger: AimLogger
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
        aim_logger : AimLogger
            AIM logger instance for experiment tracking.
        """
        self.agent = agent
        self.environment = environment
        self.aim_logger = aim_logger
        self.config = config

        # ##: Ensure the save directory exists.
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        # ##: Initialize training state.
        self.total_steps = 0
        self.episode = 0  # Track episodes for logging/evaluation purposes
        self.episode_rewards = deque(maxlen=100)  # Track recent episode rewards
        self.episode_steps_deque = deque(maxlen=100)  # Track recent episode steps

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

        # ##: Log hyperparameters to AIM.
        self.aim_logger.log_params(
            self.config.model_dump(exclude={"trial_info", "pruning_callback", "trainer_type"}), prefix="trainer"
        )
        # ##: Also log agent hyperparameters.
        self.aim_logger.log_params(self.agent.hyperparameters.model_dump(), prefix="agent")

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

                    # ##: Log episode metrics.
                    self._log_episode_metrics(
                        current_episode_reward, current_episode_steps, mean_reward_100, start_time
                    )

                    # ##: Handle evaluation and pruning (similar to EpisodeTrainer).
                    pruned = self._handle_evaluation_and_pruning()
                    if pruned:
                        logger.warning(f"Trial pruned at step {self.total_steps}.")
                        return {"pruned": True, "total_steps": self.total_steps}

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
                _, last_agent_info = self.agent.act(observation, training=False)  # Use non-training mode
            last_value = last_agent_info["value"]
            last_done = False

            self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

            # ##: Perform PPO updates over multiple epochs.
            all_update_metrics = []
            for epoch in range(self.config.n_epochs):
                for batch_data in self.rollout_buffer.sample_mini_batches(
                    batch_size=self.config.batch_size, n_epochs=1  # Sample once per epoch
                ):
                    update_metrics = self.agent.learn(batch_data)
                    all_update_metrics.append(update_metrics)

            # ##: Log aggregated update metrics for the rollout.
            if all_update_metrics:
                self._log_update_metrics(all_update_metrics)

            # ##: Save checkpoint periodically (based on total steps or episodes).
            if self.episode > 0 and self.episode % self.config.save_frequency == 0:
                # Avoid saving too frequently if episodes are short
                # Maybe add a step-based save frequency as well?
                checkpoint_path = (
                    self.config.save_dir
                    / f"{self.agent.name}_{self.environment.name}_ppo_{self.timestamp}_{self.episode}"
                )
                self.save_checkpoint(checkpoint_path)

        # ##: Final evaluation after training loop.
        final_eval_metrics = self.evaluate(self.config.num_eval_episodes)
        logger.info(f"Final Evaluation Mean Reward: {final_eval_metrics['mean_reward']:.2f}")
        self.aim_logger.log_metrics(
            {f"final_{k}": v for k, v in final_eval_metrics.items() if k != "rewards"},
            step=self.total_steps,
            epoch=self.episode,
        )
        self.aim_logger.log_params({"final_total_steps": self.total_steps, "final_episodes": self.episode})

        return {
            "episodes": self.episode,
            "total_steps": self.total_steps,
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
            "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": min(self.episode_rewards) if self.episode_rewards else 0,
        }

    def _log_episode_metrics(
        self, episode_reward: float, episode_steps: int, mean_reward_100: float, start_time: float
    ) -> None:
        """Logs episode metrics to console and AIM."""
        # ##: Log to console periodically.
        if self.episode % self.config.log_frequency == 0:
            elapsed_time = time() - start_time
            logger.info(
                f"Step {self.total_steps}/{self.config.max_total_steps} | Ep {self.episode} "
                f"| Ep Steps: {episode_steps} "
                f"| Ep Reward: {episode_reward:.2f} "
                f"| Mean Rwd (100): {mean_reward_100:.2f} "
                f"| Time: {elapsed_time:.2f}s"
            )

        # ##: Log to AIM.
        self.aim_logger.log_metrics(
            {"episode_reward": episode_reward, "episode_steps": episode_steps, "mean_reward_100": mean_reward_100},
            step=self.total_steps,
            epoch=self.episode,
            context={"subset": "train"},
        )

    def _log_update_metrics(self, all_update_metrics: list) -> None:
        """Logs aggregated metrics from the PPO update phase."""
        if not all_update_metrics:
            return

        # ##: Aggregate metrics across all mini-batches in the update phase.
        aggregated_metrics = {}
        keys = all_update_metrics[0].keys()
        for key in keys:
            # ##: Handle potential tensor values by converting to numpy.
            values = [m[key].numpy() if hasattr(m[key], "numpy") else m[key] for m in all_update_metrics if key in m]
            if values:
                aggregated_metrics[f"update_{key}_mean"] = np.mean(values)

        # ##: Log aggregated metrics to AIM.
        self.aim_logger.log_metrics(
            aggregated_metrics,
            step=self.total_steps,
            epoch=self.episode,
            context={"subset": "train_update"},
        )

    def _handle_evaluation_and_pruning(self) -> bool:
        """Handles periodic evaluation and Optuna pruning. Returns True if pruned."""
        # ##: Evaluate based on episode count for consistency.
        if self.episode > 0 and self.episode % self.config.eval_frequency == 0:
            eval_metrics = self.evaluate(self.config.num_eval_episodes)
            mean_eval_reward = eval_metrics["mean_reward"]
            min_r, max_r = eval_metrics["min_reward"], eval_metrics["max_reward"]

            logger.info(
                f"Evaluation (Ep {self.episode}, Step {self.total_steps}) Mean Reward: {mean_eval_reward:.2f} "
                f"(Min: {min_r:.2f}, Max: {max_r:.2f})"
            )

            # ##: Log evaluation metrics to AIM.
            self.aim_logger.log_metrics(
                {
                    "eval_mean_reward": mean_eval_reward,
                    "eval_max_reward": eval_metrics["max_reward"],
                    "eval_min_reward": eval_metrics["min_reward"],
                },
                step=self.total_steps,
                epoch=self.episode,
                context={"subset": "eval"},
            )

            # ##: Report to Optuna for pruning.
            if self.config.pruning_callback is not None and self.config.trial_info is not None:
                try:
                    self.config.pruning_callback(self.episode, mean_eval_reward)  # Use episode count
                except Exception as exc:
                    logger.warning(f"Trial pruned at episode {self.episode}: {exc}")
                    return True
        return False

    def evaluate(self, num_episodes: int = 1) -> Dict[str, Any]:
        """
        Evaluate the agent in the environment (using non-training mode).
        (Adapted from EpisodeTrainer)

        Parameters
        ----------
        num_episodes : int, optional
            Number of episodes to evaluate, by default 1.

        Returns
        -------
        Dict[str, Any]
            Dictionary of evaluation metrics.
        """
        eval_rewards = []
        eval_steps = []
        max_steps_per_eval_episode = self.config.max_steps_per_episode  # Use trainer config

        for _ in range(num_episodes):
            observation = self.environment.reset()
            episode_reward = 0
            episode_steps = 0

            for _ in range(max_steps_per_eval_episode):
                # ##: Use agent in non-training mode for evaluation.
                action, _ = self.agent.act(observation, training=False)
                next_observation, reward, done, _ = self.environment.step(action)

                observation = next_observation
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            eval_rewards.append(episode_reward)
            eval_steps.append(episode_steps)

        return {
            "mean_reward": mean(eval_rewards) if eval_rewards else 0,
            "max_reward": max(eval_rewards) if eval_rewards else 0,
            "min_reward": min(eval_rewards) if eval_rewards else 0,
            "mean_steps": mean(eval_steps) if eval_steps else 0,
            "rewards": eval_rewards,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save the training state (agent model and trainer state).
        (Adapted from EpisodeTrainer)

        Parameters
        ----------
        path : str | Path
            Base path for the checkpoint (directory will be created).
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        agent_path = path_obj / "agent"

        # ##: Save the agent.
        self.agent.save(str(agent_path))

        # ##: Save trainer state (episode, total_steps).
        trainer_state = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "timestamp": self.timestamp,
        }

        # ##: Log checkpoint artifact info to AIM.
        self.aim_logger.log_artifact(
            artifact_data=trainer_state,
            name=f"checkpoint_episode_{self.episode}",
            path=str(path_obj.resolve()),
            meta={
                "episode": self.episode,
                "total_steps": self.total_steps,
                "agent_save_path": str(agent_path.resolve()),
            },
        )
        logger.info(f"Checkpoint saved to: {path_obj}")
