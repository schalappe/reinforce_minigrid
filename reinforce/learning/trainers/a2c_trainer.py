# -*- coding: utf-8 -*-
"""
Episode-based trainer for reinforcement learning agents using a Replay Buffer.
"""

from time import time
from typing import Any, Dict

import tensorflow as tf
from loguru import logger

from reinforce.agents.actor_critic import A2CAgent
from reinforce.configs.models.trainer import A2CTrainerConfig
from reinforce.environments import BaseEnvironment
from reinforce.learning.trainers.ac_trainer import ActorCriticTrainer
from reinforce.utils.buffers import RolloutBuffer
from reinforce.utils.management import AimTracker, setup_logger
from reinforce.utils.preprocessing import preprocess_observation

setup_logger()


class A2CTrainer(ActorCriticTrainer):
    """
    Trainer for A2C using a Rollout Buffer and GAE.

    Inherits common logic from ActorCriticTrainer and implements on-policy data collection
    and updates using Generalized Advantage Estimation (GAE).
    """

    def __init__(
        self, *, agent: A2CAgent, environment: BaseEnvironment, config: A2CTrainerConfig, tracker: AimTracker
    ):
        """
        Initialize the episode trainer.

        Parameters
        ----------
        agent : BaseAgent
            The agent to train.
        environment : BaseEnvironment
            The environment to train in.
        config : A2CTrainerConfig
            Pydantic configuration model for the trainer.
        tracker : AimTracker
            Tracker instance for experiment tracking.
        """
        # ##: Call the base class initializer first.
        super().__init__(agent=agent, environment=environment, config=config, tracker=tracker)

        # ##: Initialize Rollout Buffer.
        self.buffer = RolloutBuffer(
            buffer_size=self.config.buffer_capacity,
            observation_shape=self.environment.observation_space["image"].shape,
            action_shape=(),
            gamma=self.agent.hyperparameters.discount_factor,
            gae_lambda=self.agent.hyperparameters.gae_lambda,
        )
        logger.info(
            f"Initialized Rollout Buffer with size {self.config.buffer_capacity}, "
            f"gamma={self.agent.hyperparameters.discount_factor}, "
            f"gae_lambda={self.agent.hyperparameters.gae_lambda}"
        )

    def train(self) -> Dict[str, Any]:
        """
        Train the agent by collecting rollouts and updating the policy.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics from the base class finalization.
        """
        start_time = time()
        max_steps = self.config.get_max_steps()
        observation = self.environment.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while self.total_steps < max_steps:
            for _ in range(self.config.buffer_capacity):
                # ##: Run one step in the environment.
                next_observation, reward, done, action, agent_info = self._run_environment_step(observation)
                value = agent_info["value"]
                log_prob = agent_info["log_prob"]

                # ##: Store experience in the buffer (raw observation).
                self.buffer.add(observation, action, reward, done, value, log_prob)

                # ##: Update state.
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                if done or episode_steps >= self.config.max_steps_per_episode:
                    # ##: Log episode metrics.
                    self.episode += 1
                    self._log_episode_metrics(episode_reward, episode_steps)
                    self._log_console_progress(start_time, episode_reward, episode_steps)

                    # ##: Handle evaluation and pruning.
                    pruned = self._run_evaluation_and_pruning()
                    if pruned:
                        return {"pruned": True, "episodes": self.episode, "total_steps": self.total_steps}

                    # ##: Save checkpoint.
                    self._save_checkpoint()

                    # ##: Reset episode state.
                    observation = self.environment.reset()
                    episode_reward = 0
                    episode_steps = 0

                # ##: Check if training should stop based on total steps.
                if self.total_steps >= max_steps:
                    break

            # ##: Compute returns and advantages after collecting the rollout.
            _, last_value = self.agent.model(tf.expand_dims(preprocess_observation(observation), axis=0))
            last_value = last_value[0, 0].numpy()
            self.buffer.compute_returns_and_advantages(last_value=last_value, last_done=done)

            # ##: Get the full batch from the rollout buffer.
            rollout_data = self.buffer.get_batch()

            # ##: Update the agent using the collected rollout data.
            learn_info = self.agent.learn(rollout_data)
            self._log_agent_update_metrics(learn_info)

            # ##: Clear the buffer for the next rollout.
            self.buffer.clear()

        # ##: Final evaluation, logging, and saving handled by base class method.
        final_metrics = self._finalize_training()
        return final_metrics
