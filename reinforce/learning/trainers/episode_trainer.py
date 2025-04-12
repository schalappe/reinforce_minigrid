# -*- coding: utf-8 -*-
"""
Episode-based trainer for reinforcement learning agents.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import time
from typing import Any, Dict, List

import tensorflow as tf
from aim import Distribution
from loguru import logger
from numpy import isscalar

from reinforce.agents import BaseAgent
from reinforce.configs.models import EpisodeTrainerConfig
from reinforce.environments import BaseEnvironment
from reinforce.learning.evaluation import evaluate_agent
from reinforce.learning.trainers.base_trainer import BaseTrainer
from reinforce.utils.logger import AimTracker, setup_logger
from reinforce.utils.persistence import save_checkpoint
from reinforce.utils.preprocessing import preprocess_observation

setup_logger()


class EpisodeTrainer(BaseTrainer):
    """
    Episode-based trainer for reinforcement learning agents.

    This trainer trains an agent in an environment for a number of episodes. It collects experiences during
    each episode and updates the agent after each episode or after a specified number of steps.
    """

    def __init__(
        self, *, agent: BaseAgent, environment: BaseEnvironment, config: EpisodeTrainerConfig, tracker: AimTracker
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
        tracker : AimTracker
            Tracker instance for experiment tracking.
        """
        self.agent = agent
        self.environment = environment
        self.tracker = tracker
        self.config = config

        # ##: Ensure the save directory exists (keep this logic here).
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        # ##: Initialize training state.
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)

        # ##: Timestamp for saving models (can be useful even with AIM).
        self.timestamp = int(datetime.now().timestamp())  # Keep timestamp generation

        # ##: Log hyperparameters using the injected logger.
        self.tracker.log_params(
            self.config.model_dump(exclude={"trial_info", "pruning_callback", "trainer_type"}), prefix="trainer"
        )
        self.tracker.log_params(self.agent.hyperparameters.model_dump(), prefix="agent")

    def _run_episode_step(self, observation: Any) -> tuple:
        """Runs a single step within an episode."""
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
            self.tracker.log_metrics(
                learn_info, step=self.total_steps, epoch=self.episode, context={"subset": "train_update"}
            )

        # ##: Log agent action info (use last_agent_info).
        if last_agent_info and isinstance(last_agent_info, dict):
            if "action_probs" in last_agent_info:
                try:
                    self.tracker.log_metric(
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
            scalar_agent_metrics = {k: v for k, v in other_agent_metrics.items() if isscalar(v)}
            if scalar_agent_metrics:
                self.tracker.log_metrics(
                    scalar_agent_metrics,
                    step=self.total_steps,
                    epoch=self.episode,
                    context={"subset": "train_action"},
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
            for _ in range(self.config.max_steps_per_episode):
                # ##: Run one step in the environment (using raw observation).
                next_observation, reward, done, action, agent_info = self._run_episode_step(observation)

                # ##: Store raw experience.
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_observation)
                dones.append(done)
                agent_infos.append(agent_info)

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

            # ##: Log episode metrics directly using the logger.
            self.episode_rewards.append(episode_reward)
            mean_reward_100 = mean(self.episode_rewards) if self.episode_rewards else 0
            self.tracker.log_metrics(
                {"episode_reward": episode_reward, "episode_steps": episode_steps, "mean_reward_100": mean_reward_100},
                step=self.total_steps,
                epoch=self.episode + 1,
                context={"subset": "train"},
            )

            # ##: Log progress to console periodically.
            if (self.episode + 1) % self.config.log_frequency == 0:
                elapsed_time = time() - start_time
                logger.info(
                    f"Ep {self.episode + 1}/{self.config.max_episodes} "
                    f"| Steps: {episode_steps} "
                    f"| Reward: {episode_reward:.2f} "
                    f"| Mean Rwd (100): {mean_reward_100:.2f} "
                    f"| Total Steps: {self.total_steps} "
                    f"| Time: {elapsed_time:.2f}s"
                )

            # ##: Handle evaluation and pruning.
            if (self.episode + 1) % self.config.eval_frequency == 0:
                eval_metrics = evaluate_agent(
                    agent=self.agent,
                    environment=self.environment,
                    num_episodes=self.config.num_eval_episodes,
                    max_steps_per_episode=self.config.max_steps_per_episode,
                )
                mean_eval_reward = eval_metrics["mean_reward"]
                min_r, max_r = eval_metrics["min_reward"], eval_metrics["max_reward"]

                logger.info(
                    f"Evaluation (Ep {self.episode + 1}) Mean Reward: {mean_eval_reward:.2f} "
                    f"(Min: {min_r:.2f}, Max: {max_r:.2f})"
                )

                # ##: Log evaluation metrics using the injected logger.
                self.tracker.log_metrics(
                    {
                        "eval_mean_reward": mean_eval_reward,
                        "eval_max_reward": eval_metrics["max_reward"],
                        "eval_min_reward": eval_metrics["min_reward"],
                    },
                    step=self.total_steps,
                    epoch=self.episode + 1,
                    context={"subset": "eval"},
                )

                # ##: Report to Optuna for pruning (keep pruning logic here for now).
                if self.config.pruning_callback is not None and self.config.trial_info is not None:
                    try:
                        self.config.pruning_callback(self.episode + 1, mean_eval_reward)
                    except Exception as exc:
                        logger.warning(f"Trial pruned at episode {self.episode + 1}: {exc}")
                        return {
                            "pruned": True,
                            "episodes": self.episode + 1,
                            "total_steps": self.total_steps,
                            "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
                            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
                        }

            # ##: Save the agent checkpoint periodically using CheckpointManager.
            if (self.episode + 1) % self.config.save_frequency == 0:
                checkpoint_path_base = (
                    self.config.save_dir
                    / f"{self.agent.name}_{self.environment.name}_{self.timestamp}_{self.episode + 1}"
                )
                trainer_state = {
                    "episode": self.episode + 1,
                    "total_steps": self.total_steps,
                    "timestamp": self.timestamp,
                }
                save_checkpoint(
                    agent=self.agent,
                    save_path_base=checkpoint_path_base,
                    trainer_state=trainer_state,
                    tracker=self.tracker,
                )

        # ##: Final evaluation after training loop.
        final_eval_metrics = evaluate_agent(
            agent=self.agent,
            environment=self.environment,
            num_episodes=self.config.num_eval_episodes,
            max_steps_per_episode=self.config.max_steps_per_episode,
        )
        logger.info(f"Final Evaluation Mean Reward: {final_eval_metrics['mean_reward']:.2f}")
        self.tracker.log_metrics(
            {f"final_{k}": v for k, v in final_eval_metrics.items() if k != "rewards"},
            step=self.total_steps,
            epoch=self.episode + 1,
            context={"subset": "final_eval"},
        )
        self.tracker.log_params({"final_total_steps": self.total_steps, "final_episodes": self.episode + 1})

        # ##: Save the final model if a path is specified in the config (Keep this logic).
        if self.config.save_path:
            try:
                final_save_path = Path(self.config.save_path)
                final_save_path.parent.mkdir(parents=True, exist_ok=True)
                self.agent.save(str(final_save_path))
                logger.info(f"Final agent model saved successfully to: {final_save_path}")
            except Exception as e:
                logger.error(f"Failed to save final agent model to {self.config.save_path}: {e}")
                logger.exception("Final model saving failed.")

        # ##: Return final training summary metrics.
        return {
            "episodes": self.episode + 1,
            "total_steps": self.total_steps,
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
            "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": min(self.episode_rewards) if self.episode_rewards else 0,
        }
