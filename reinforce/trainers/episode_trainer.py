# -*- coding: utf-8 -*-
"""
Episode-based trainer for reinforcement learning agents.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import time
from typing import Any, Dict, List, Union

import numpy as np
import tensorflow as tf
from aim import Distribution
from loguru import logger

from reinforce.configs.models import EpisodeTrainerConfig
from reinforce.core.base_agent import BaseAgent
from reinforce.core.base_environment import BaseEnvironment
from reinforce.core.base_trainer import BaseTrainer
from reinforce.utils.aim_logger import AimLogger
from reinforce.utils.logging_setup import setup_logger
from reinforce.utils.preprocessing import preprocess_observation

setup_logger()


class EpisodeTrainer(BaseTrainer):
    """
    Episode-based trainer for reinforcement learning agents.

    This trainer trains an agent in an environment for a number of episodes. It collects experiences during
    each episode and updates the agent after each episode or after a specified number of steps.
    """

    def __init__(
        self, *, agent: BaseAgent, environment: BaseEnvironment, config: EpisodeTrainerConfig, aim_logger: AimLogger
    ):
        """
        Initialize the episode trainer.

        Parameters
        ----------
        agent : BaseAgent
            The agent to train.
        environment : BaseEnvironment
            The environment to train in.
        config : EpisodeTrainerConfig
            Pydantic configuration model for the trainer.
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
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)

        # ##: Timestamp for saving models (can be useful even with AIM).
        self.timestamp = int(datetime.now().timestamp())

        # ##: Log hyperparameters to AIM.
        self.aim_logger.log_params(
            self.config.model_dump(exclude={"trial_info", "pruning_callback", "trainer_type"}), prefix="trainer"
        )

    def train(self) -> Dict[str, Any]:
        """
        Train the agent in the environment.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics.
        """
        start_time = time()
        max_episodes = self.config.max_episodes

        for self.episode in range(self.episode, max_episodes):
            observation = self.environment.reset()

            # ##: Initialize episode state.
            episode_reward = 0
            episode_steps = 0

            # ##: Store experiences for batch updates (store raw obs).
            observations, actions, rewards, next_observations, dones, agent_infos = [], [], [], [], [], []

            # ##: Run one episode.
            for step in range(self.config.max_steps_per_episode):
                # ##: Run one step in the environment (using raw observation).
                next_observation, reward, done, action, agent_info = self._run_episode_step(observation, step)

                # ##: Store raw experience.
                observations.append(observation)  # Store raw observation
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_observation)  # Store raw next_observation
                dones.append(done)
                agent_infos.append(agent_info)  # Store agent info

                # ##: Update state (use raw observation for next step).
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                # ##: Update the agent if it's time.
                if self.total_steps % self.config.update_frequency == 0 and len(observations) > 0:
                    self._handle_agent_update(observations, actions, rewards, next_observations, dones, agent_infos)
                    observations, actions, rewards, next_observations, dones, agent_infos = [], [], [], [], [], []

                if done:
                    break

            # ##: Log episode metrics.
            self._log_episode_metrics(episode_reward, episode_steps, start_time)

            # ##: Handle evaluation and pruning.
            pruned = self._handle_evaluation_and_pruning()
            if pruned:
                return {
                    "pruned": True,
                    "episodes": self.episode + 1,
                    "total_steps": self.total_steps,
                    "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
                    "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
                }

            # ##: Save the agent checkpoint periodically.
            if (self.episode + 1) % self.config.save_frequency == 0:
                checkpoint_path = (
                    self.config.save_dir
                    / f"{self.agent.name}_{self.environment.name}_{self.timestamp}_{self.episode + 1}"
                )
                self.save_checkpoint(checkpoint_path)

        # ##: Final evaluation after training loop.
        final_eval_metrics = self.evaluate(self.config.num_eval_episodes)
        logger.info(f"Final Evaluation Mean Reward: {final_eval_metrics['mean_reward']:.2f}")
        self.aim_logger.log_metrics(
            {f"final_{k}": v for k, v in final_eval_metrics.items() if k != "rewards"},
            step=self.total_steps,
            epoch=self.episode + 1,
        )
        self.aim_logger.log_params({"final_total_steps": self.total_steps, "final_episodes": self.episode + 1})

        return {
            "episodes": self.episode + 1,
            "total_steps": self.total_steps,
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
            "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": min(self.episode_rewards) if self.episode_rewards else 0,
        }

    def _run_episode_step(self, observation: Any, step: int) -> tuple:
        """Runs a single step within an episode."""
        # ##: Log environment image periodically.
        if self.config.log_env_image_frequency > 0 and self.total_steps % self.config.log_env_image_frequency == 0:
            try:
                processed_observation = observation
                if isinstance(observation, np.ndarray) and getattr(observation, "dtype", "N/A") == np.float32:
                    processed_observation = (observation * 255).clip(0, 255).astype(np.uint8)

                self.aim_logger.log_image(
                    processed_observation,
                    name="environment_observation",
                    step=self.total_steps,
                    epoch=self.episode,
                    caption=f"Episode {self.episode}, Step {step}",
                )
            except Exception as exc:
                logger.warning(f"Could not log environment image: {exc}")

        # ##: Get action from agent and step environment.
        action, agent_info = self.agent.act(observation)
        next_observation, reward, done, _ = self.environment.step(action)
        return next_observation, reward, done, action, agent_info

    def _handle_agent_update(
        self,
        observations: List,
        actions: List,
        rewards: List,
        next_observations: List,
        dones: List,
        agent_infos: List[Dict],
    ) -> None:
        """Handles the agent learning step and associated logging using tf.data."""
        # ##: Preprocess observations and next_observations here.
        # ##: Use tf.stack to handle the list of numpy arrays correctly.
        processed_observations = tf.stack([preprocess_observation(obs) for obs in observations])
        processed_next_observations = tf.stack([preprocess_observation(n_obs) for n_obs in next_observations])

        # ##: Convert other lists to tensors.
        tf_actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        tf_rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        tf_dones = tf.convert_to_tensor(dones, dtype=tf.float32)  # Use float for TF compatibility

        # ##: Pass tensors directly to agent.learn.
        logger.debug("Calling agent.learn with prepared tensors...")
        batch_tensors = (processed_observations, tf_actions, tf_rewards, processed_next_observations, tf_dones)
        learn_info = self.agent.learn(batch_tensors)
        logger.debug("agent.learn call completed.")

        # ##: Logging needs access to the last agent_info from the batch.
        last_agent_info = agent_infos[-1] if agent_infos else {}

        # ##: Log agent learning info.
        if learn_info and isinstance(learn_info, dict):
            self.aim_logger.log_metrics(
                learn_info, step=self.total_steps, epoch=self.episode, context={"subset": "train_update"}
            )

        # ##: Log agent action info (use last_agent_info).
        if last_agent_info and isinstance(last_agent_info, dict):
            if "action_probs" in last_agent_info:
                try:
                    self.aim_logger.log_metric(
                        name="action_probabilities",
                        value=Distribution(last_agent_info["action_probs"]),
                        step=self.total_steps,
                        epoch=self.episode,
                        context={"subset": "train"},
                    )
                except Exception as exc:
                    logger.warning(f"Could not log action probabilities distribution: {exc}")

            # ##: Log other scalar metrics from last_agent_info.
            other_agent_metrics = {k: v for k, v in last_agent_info.items() if k != "action_probs"}
            scalar_agent_metrics = {k: v for k, v in other_agent_metrics.items() if np.isscalar(v)}
            if scalar_agent_metrics:
                self.aim_logger.log_metrics(
                    scalar_agent_metrics,
                    step=self.total_steps,
                    epoch=self.episode,
                    context={"subset": "train_action"},
                )

    def _log_episode_metrics(self, episode_reward: float, episode_steps: int, start_time: float) -> float:
        """Logs episode metrics to console and AIM."""
        self.episode_rewards.append(episode_reward)
        mean_reward = mean(self.episode_rewards) if self.episode_rewards else 0

        # ##: Log to console.
        if (self.episode + 1) % self.config.log_frequency == 0:
            self._log_progress(episode_reward, episode_steps, mean_reward, start_time)

        # ##: Log to AIM.
        self.aim_logger.log_metrics(
            {"episode_reward": episode_reward, "episode_steps": episode_steps, "mean_reward_100": mean_reward},
            step=self.total_steps,
            epoch=self.episode + 1,
            context={"subset": "train"},
        )

        return mean_reward

    def _handle_evaluation_and_pruning(self) -> bool:
        """Handles periodic evaluation and Optuna pruning. Returns True if pruned."""
        if (self.episode + 1) % self.config.eval_frequency == 0:
            eval_metrics = self.evaluate(self.config.num_eval_episodes)
            mean_eval_reward = eval_metrics["mean_reward"]
            min_r, max_r = eval_metrics["min_reward"], eval_metrics["max_reward"]

            logger.info(
                f"Evaluation (Ep {self.episode + 1}) Mean Reward: {mean_eval_reward:.2f} "
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
                epoch=self.episode + 1,
                context={"subset": "eval"},
            )

            # ##: Report to Optuna for pruning.
            if self.config.pruning_callback is not None and self.config.trial_info is not None:
                try:
                    self.config.pruning_callback(self.episode + 1, mean_eval_reward)
                except Exception as exc:
                    logger.warning(f"Trial pruned at episode {self.episode + 1}: {exc}")
                    return True

        return False

    def evaluate(self, num_episodes: int = 1) -> Dict[str, Any]:
        """
        Evaluate the agent in the environment.

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

        for _ in range(num_episodes):
            observation = self.environment.reset()
            episode_reward = 0
            episode_steps = 0

            # ##: Run one evaluation episode.
            for _ in range(self.config.max_steps_per_episode):
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
        Save the training state to the specified path.

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

        # ##: Save trainer state.
        trainer_state = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "timestamp": self.timestamp,
        }

        # ##: Log checkpoint artifact info to AIM.
        self.aim_logger.log_artifact(
            artifact_data=trainer_state,
            name=f"checkpoint_episode_{self.episode + 1}",
            path=str(path_obj.resolve()),
            meta={
                "episode": self.episode + 1,
                "total_steps": self.total_steps,
                "agent_save_path": str(agent_path.resolve()),
            },
        )
        logger.info(f"Checkpoint saved to: {path_obj}")

    def _log_progress(
        self, episode_reward: float, episode_steps: int, mean_reward_100: float, start_time: float
    ) -> None:
        """
        Log training progress to the console.

            Reward of the current episode.
        episode_steps : int
            Number of steps in the current episode.
        mean_reward_100 : float
            Mean reward over the last 100 episodes.
        start_time : float
            Time when training started.
        """
        elapsed_time = time() - start_time

        logger.info(
            f"Ep {self.episode + 1}/{self.config.max_episodes} "
            f"| Steps: {episode_steps} "
            f"| Reward: {episode_reward:.2f} "
            f"| Mean Rwd (100): {mean_reward_100:.2f} "
            f"| Total Steps: {self.total_steps} "
            f"| Time: {elapsed_time:.2f}s"
        )
