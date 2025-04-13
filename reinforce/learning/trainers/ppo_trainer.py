# -*- coding: utf-8 -*-
"""
Trainer for Proximal Policy Optimization (PPO) using Rollout Buffers.
"""

from time import time
from typing import Any, Dict, Tuple

import tensorflow as tf
from loguru import logger
from numpy import mean, ndarray

from reinforce.agents.actor_critic import PPOAgent
from reinforce.configs.models.trainer import PPOTrainerConfig
from reinforce.environments import BaseEnvironment
from reinforce.learning.trainers.ac_trainer import ActorCriticTrainer
from reinforce.utils.buffers import RolloutBuffer
from reinforce.utils.management import AimTracker, setup_logger

setup_logger()


class PPOTrainer(ActorCriticTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO).

    Inherits common logic from ActorCriticTrainer and implements the PPO-specific rollout collection
    and batched update strategy.
    """

    def __init__(
        self, *, agent: PPOAgent, environment: BaseEnvironment, config: PPOTrainerConfig, tracker: AimTracker
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
        tracker : AimTracker
            Tracker instance for experiment tracking.
        """
        # ##: Call the base class initializer first.
        super().__init__(agent=agent, environment=environment, config=config, tracker=tracker)

        # ##: Initialize the Rollout Buffer (Specific to PPOTrainer).
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.config.n_steps,
            observation_shape=self.environment.observation_space["image"].shape,
            action_shape=(),
            gamma=self.agent.hyperparameters.discount_factor,
            gae_lambda=self.agent.hyperparameters.gae_lambda,
        )
        logger.info(f"Initialized Rollout Buffer with size {self.config.n_steps}")

    def _collect_rollouts(
        self, observation: Any, current_episode_reward: float, current_episode_steps: int, start_time: float
    ) -> Tuple[ndarray, float, int, float]:
        """
        Collects rollouts for n_steps.

        Parameters
        ----------
        observation : Any
            The current observation.
        current_episode_reward : float
            The current episode reward.
        current_episode_steps : int
            The current episode steps.
        start_time : float
            The start time of the training.

        Returns
        -------
        Tuple[np.ndarray, float, int, float]
            The next observation, the current episode reward, the current episode steps, and the start time.
        """
        self.rollout_buffer.clear()

        for _ in range(self.config.n_steps):
            self.total_steps += 1
            current_episode_steps += 1

            # ##: Run one step in the environment.
            next_observation, reward, done, action, agent_info = self._run_environment_step(observation)
            value = agent_info["value"]
            log_prob = agent_info["log_prob"]

            # ##: Store experience in the buffer (raw observation).
            self.rollout_buffer.add(observation, action, reward, done, value, log_prob)

            # ##: Update current state.
            observation = next_observation
            current_episode_reward += reward

            # ##: Handle episode termination within the rollout.
            if done:
                self.episode += 1

                # ##: Log episode metrics using base class method.
                self._log_episode_metrics(current_episode_reward, current_episode_steps)

                # ##: Log progress to console using base class method.
                self._log_console_progress(start_time, current_episode_reward, current_episode_steps)

                # ##: Handle evaluation and pruning using base class method.
                pruned = self._run_evaluation_and_pruning()
                if pruned:
                    return observation, current_episode_reward, current_episode_steps, True

                # ##: Save checkpoint using base class method (based on episode frequency).
                self._save_checkpoint()

                # ##: Reset environment and episode trackers.
                observation = self.environment.reset()
                current_episode_reward = 0
                current_episode_steps = 0

            # ##: Check if max total steps reached during rollout.
            if self.total_steps >= self.config.max_total_steps:
                break

        return observation, current_episode_reward, current_episode_steps, False

    def _update_agent_ppo(self, last_observation: ndarray):
        """
        Performs the PPO update phase.

        Parameters
        ----------
        last_observation : ndarray
            The last observation from the environment.
        """
        with tf.device("/cpu:0"):  # Ensure value prediction is on CPU if needed
            _, last_agent_info = self.agent.act(last_observation, training=False)
        last_value = last_agent_info["value"]
        last_done = False

        self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

        # ##: Perform PPO updates over multiple epochs.
        all_update_metrics = []
        for _ in range(self.config.n_epochs):
            for batch_data in self.rollout_buffer.sample_mini_batches(batch_size=self.config.batch_size, n_epochs=1):
                update_metrics = self.agent.learn(batch_data)
                if update_metrics:
                    all_update_metrics.append(update_metrics)

        # ##: Log aggregated update metrics for the rollout.
        if all_update_metrics:
            aggregated_metrics = {}
            keys = set(k for d in all_update_metrics for k in d)
            for key in keys:
                values = [
                    m[key].numpy() if hasattr(m[key], "numpy") else m[key]
                    for m in all_update_metrics
                    if key in m and m[key] is not None
                ]
                if values:
                    try:
                        aggregated_metrics[f"update_{key}_mean"] = mean(values)
                    except TypeError as e:
                        logger.warning(f"Could not aggregate metric '{key}': {e}. Values: {values}")

            # ##: Log aggregated metrics using base class method (or dedicated tracker call).
            if aggregated_metrics:
                self.tracker.log_metrics(
                    aggregated_metrics,
                    step=self.total_steps,
                    epoch=self.episode,
                    context={"subset": "train_update"},
                )

    def train(self) -> Dict[str, Any]:
        """
        Train the PPO agent by collecting rollouts and performing batched updates.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics from the base class finalization.
        """
        start_time = time()
        max_total_steps = self.config.max_total_steps
        observation = self.environment.reset()
        current_episode_reward = 0.0
        current_episode_steps = 0

        # ##: Main training loop until max_total_steps reached.
        while self.total_steps < max_total_steps:
            last_observation, current_episode_reward, current_episode_steps, pruned = self._collect_rollouts(
                observation, current_episode_reward, current_episode_steps, start_time
            )
            observation = last_observation

            if pruned:
                return {"pruned": True, "episodes": self.episode, "total_steps": self.total_steps}

            if self.rollout_buffer.is_full():
                self._update_agent_ppo(last_observation)
            else:
                position = self.rollout_buffer.ptr
                buffer_size = self.rollout_buffer.buffer_size
                logger.warning(
                    f"Rollout ended before buffer was full ({position}/{buffer_size}). Skipping update phase."
                )

        # ##: Final evaluation, logging, and saving handled by base class method.
        final_metrics = self._finalize_training()
        return final_metrics
