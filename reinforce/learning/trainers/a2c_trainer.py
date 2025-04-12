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
from reinforce.utils.buffers import ReplayBuffer
from reinforce.utils.management import AimTracker, setup_logger
from reinforce.utils.preprocessing import preprocess_observation

setup_logger()


class A2CTrainer(ActorCriticTrainer):
    """
    Episode-based trainer using a Replay Buffer.

    Inherits common logic from ActorCriticTrainer and implements episode-specific data collection
    and agent updates based on sampled experiences.
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

        # ##: Initialize Replay Buffer.
        self.buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            observation_shape=self.environment.observation_space.shape,
            action_shape=self.environment.action_space.shape,
        )
        logger.info(f"Initialized Replay Buffer with capacity {self.config.buffer_capacity}")

    def _update_agent(self):
        """Samples from buffer and updates the agent."""
        if self.total_steps % self.config.update_frequency == 0 and self.buffer.can_sample(self.config.batch_size):
            # ##: Sample batch from buffer.
            batch = self.buffer.sample(self.config.batch_size)

            # ##: Preprocess and convert to tensors.
            processed_observations = tf.stack([preprocess_observation(obs) for obs in batch["observations"]])
            processed_next_observations = tf.stack(
                [preprocess_observation(n_obs) for n_obs in batch["next_observations"]]
            )
            tf_actions = tf.convert_to_tensor(batch["actions"], dtype=tf.int32)
            tf_rewards = tf.convert_to_tensor(batch["rewards"], dtype=tf.float32)
            tf_dones = tf.convert_to_tensor(batch["dones"], dtype=tf.float32)

            # ##: Pass tensors directly to agent.learn.
            batch_tensors = (
                processed_observations,
                tf_actions,
                tf_rewards,
                processed_next_observations,
                tf_dones,
            )
            learn_info = self.agent.learn(batch_tensors)

            # ##: Log agent learning info using base class method.
            self._log_agent_update_metrics(learn_info)

    def train(self) -> Dict[str, Any]:
        """
        Train the agent by running episodes and updating from a replay buffer.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics from the base class finalization.
        """
        start_time = time()
        max_episodes = self.config.max_episodes
        observation = self.environment.reset()

        while self.episode < max_episodes:
            self.episode += 1

            # ##: Initialize episode state.
            episode_reward = 0
            episode_steps = 0
            done = False

            # ##: Run one episode.
            while not done and episode_steps < self.config.max_steps_per_episode:
                # ##: Run one step in the environment.
                next_observation, reward, step_done, action, _ = self._run_environment_step(observation)

                # ##: Store experience in the buffer (using raw observations).
                experience = {
                    "observation": observation,
                    "action": action,
                    "reward": reward,
                    "next_observation": next_observation,
                    "done": step_done,
                }
                self.buffer.add(experience)

                # ##: Update state.
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                done = step_done

                # ##: Update the agent periodically.
                self._update_agent()

            # ##: Log episode metrics using base class method.
            self._log_episode_metrics(episode_reward, episode_steps)

            # ##: Log progress to console using base class method.
            self._log_console_progress(start_time, episode_reward, episode_steps)

            # ##: Handle evaluation and pruning using base class method.
            pruned = self._run_evaluation_and_pruning()
            if pruned:
                return {"pruned": True, "episodes": self.episode, "total_steps": self.total_steps}

            # ##: Save checkpoint using base class method.
            self._save_checkpoint()

            # ##: Reset environment for the next episode (if not done by env itself).
            if not done:
                observation = self.environment.reset()

        # ##: Final evaluation, logging, and saving handled by base class method.
        final_metrics = self._finalize_training()
        return final_metrics
