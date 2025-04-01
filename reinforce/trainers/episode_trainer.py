# -*- coding: utf-8 -*-
"""
Episode-based trainer for reinforcement learning agents.
"""

import time
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from reinforce.core.base_agent import BaseAgent
from reinforce.core.base_environment import BaseEnvironment
from reinforce.core.base_trainer import BaseTrainer


class EpisodeTrainer(BaseTrainer):
    """
    Episode-based trainer for reinforcement learning agents.

    This trainer trains an agent in an environment for a number of episodes. It collects experiences during
    each episode and updates the agent after each episode or after a specified number of steps.
    """

    def __init__(
        self,
        agent: BaseAgent,
        environment: BaseEnvironment,
        config: Dict[str, Any],
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize the episode trainer.

        Parameters
        ----------
        agent : BaseAgent
            The agent to train.
        environment : BaseEnvironment
            The environment to train in.
        config : dict
            Configuration parameters for training.
        callbacks : list of callable, optional
            Optional callbacks to invoke during training.
        """
        self.agent = agent
        self.environment = environment
        self.callbacks = callbacks or []

        # ##: Extract configuration parameters.
        self.max_episodes = config.get("max_episodes", 1000)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 100)
        self.update_frequency = config.get("update_frequency", 1)
        self.eval_frequency = config.get("eval_frequency", 10)
        self.num_eval_episodes = config.get("num_eval_episodes", 5)
        self.gamma = config.get("gamma", 0.99)
        self.log_frequency = config.get("log_frequency", 1)
        self.save_frequency = config.get("save_frequency", 100)
        self.save_dir = Path(config.get("save_dir", Path("outputs") / "models"))

        # ##: Optuna trial information for pruning.
        self._trial_info = config.get("_trial_info", None)
        self._pruning_callback = config.get("_pruning_callback", None)

        # ##: Ensure the save directory exists.
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ##: Initialize training state.
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)

        # ##: Timestamp for saving models.
        self.timestamp = int(datetime.now().timestamp())

    def train(self) -> Dict[str, Any]:
        """
        Train the agent in the environment.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics.
        """
        start_time = time.time()

        for self.episode in range(self.episode, self.max_episodes):
            observation = self.environment.reset()

            # ##: Initialize episode state.
            episode_reward = 0
            episode_steps = 0

            # ##: Store experiences for batch updates.
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []

            # ##: Run one episode.
            for step in range(self.max_steps_per_episode):
                action, _ = self.agent.act(observation)
                next_observation, reward, done, _ = self.environment.step(action)

                # ##: Store the experience.
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_observation)
                dones.append(done)

                # ##: Update episode state.
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                # ##: Update the agent if it's time.
                if self.total_steps % self.update_frequency == 0 and len(observations) > 0:
                    np_observations = np.array(observations)
                    np_actions = np.array(actions)
                    np_rewards = np.array(rewards)
                    np_next_observations = np.array(next_observations)
                    np_dones = np.array(dones)

                    self.agent.learn(np_observations, np_actions, np_rewards, np_next_observations, np_dones)

                    observations = []
                    actions = []
                    rewards = []
                    next_observations = []
                    dones = []

                if done:
                    break

            # ##: Store the episode reward.
            self.episode_rewards.append(episode_reward)

            # ##: Log training progress.
            if (self.episode + 1) % self.log_frequency == 0:
                self._log_progress(episode_reward, episode_steps, start_time)

            # ##: Evaluate the agent.
            if (self.episode + 1) % self.eval_frequency == 0:
                eval_rewards = self.evaluate(self.num_eval_episodes)
                mean_reward = eval_rewards["mean_reward"]

                print(
                    f"Evaluation reward: {mean_reward:.2f} "
                    f"(min: {eval_rewards['min_reward']:.2f}, max: {eval_rewards['max_reward']:.2f})"
                )

                # ##: Report to Optuna for pruning if this is a trial.
                if self._pruning_callback is not None and self._trial_info is not None:
                    try:
                        self._pruning_callback(self.episode + 1, mean_reward)
                    except Exception as e:
                        print(f"Trial pruned at episode {self.episode + 1}: {e}")
                        # Return current results if pruned
                        return {
                            "pruned": True,
                            "episodes": self.episode + 1,
                            "total_steps": self.total_steps,
                            "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
                            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
                        }

            # ##: Save the agent.
            if (self.episode + 1) % self.save_frequency == 0:
                self.save_checkpoint(
                    self.save_dir / f"{self.agent.name}_{self.environment.name}_{self.timestamp}_{self.episode + 1}"
                )

            # ##: Invoke callbacks.
            for callback in self.callbacks:
                callback(self, self.episode, episode_reward, episode_steps)

        return {
            "episodes": self.episode + 1,
            "total_steps": self.total_steps,
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
            "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": min(self.episode_rewards) if self.episode_rewards else 0,
        }

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

        for episode in range(num_episodes):
            observation = self.environment.reset()

            episode_reward = 0

            # ##: Run one episode.
            for step in range(self.max_steps_per_episode):
                action, _ = self.agent.act(observation, training=False)

                next_observation, reward, done, _ = self.environment.step(action)

                observation = next_observation
                episode_reward += reward

                if done:
                    break

            # ##: Store the episode reward.
            eval_rewards.append(episode_reward)

        return {
            "mean_reward": mean(eval_rewards) if eval_rewards else 0,
            "max_reward": max(eval_rewards) if eval_rewards else 0,
            "min_reward": min(eval_rewards) if eval_rewards else 0,
            "rewards": eval_rewards,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save the training state to the specified path.

        Parameters
        ----------
        path : str | Path
            Directory path to save the training state.
        """
        # ##: Create directory if it doesn't exist.
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # ##: Save the agent.
        self.agent.save(str(path_obj / "agent"))

        # ##: Save trainer state.
        trainer_state = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "timestamp": self.timestamp,
        }
        np.save(path_obj / "trainer_state.npy", np.array([trainer_state], dtype=object))

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load the training state from the specified path.

        Parameters
        ----------
        path : str or Path
            Directory path to load the training state from.
        """
        path_obj = Path(path)
        self.agent.load(str(path_obj / "agent"))

        # ##: Load trainer state.
        trainer_state = np.load(path_obj / "trainer_state.npy", allow_pickle=True).item()
        self.episode = trainer_state["episode"]
        self.total_steps = trainer_state["total_steps"]
        self.timestamp = trainer_state["timestamp"]

    def _log_progress(self, episode_reward: float, episode_steps: int, start_time: float) -> None:
        """
        Log training progress.

        Parameters
        ----------
        episode_reward : float
            Reward of the current episode.
        episode_steps : int
            Number of steps in the current episode.
        start_time : float
            Time when training started.
        """
        elapsed_time = time.time() - start_time
        mean_reward = mean(self.episode_rewards) if self.episode_rewards else 0

        print(
            f"Episode {self.episode + 1}/{self.max_episodes} "
            f"| Steps: {episode_steps} "
            f"| Reward: {episode_reward:.2f} "
            f"| Mean Reward: {mean_reward:.2f} "
            f"| Total Steps: {self.total_steps} "
            f"| Time: {elapsed_time:.2f}s"
        )
