# -*- coding: utf-8 -*-
"""
Episode-based trainer for reinforcement learning agents.
"""

from collections import deque
from datetime import datetime
from logging import getLogger
from pathlib import Path
from statistics import mean
from time import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from aim import Distribution

from reinforce.core.base_agent import BaseAgent
from reinforce.core.base_environment import BaseEnvironment
from reinforce.core.base_trainer import BaseTrainer
from reinforce.utils.aim_logger import AimLogger

logger = getLogger(__name__)


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
        aim_logger: Optional[AimLogger] = None,
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
        aim_logger : AimLogger, optional
            AIM logger instance for experiment tracking.
        """
        self.agent = agent
        self.environment = environment
        self.callbacks = callbacks or []
        self.aim_logger = aim_logger

        # ##: Extract configuration parameters.
        self.max_episodes = config.get("max_episodes", 1000)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 100)
        self.update_frequency = config.get("update_frequency", 1)
        self.eval_frequency = config.get("eval_frequency", 10)
        self.num_eval_episodes = config.get("num_eval_episodes", 5)
        self.gamma = config.get("gamma", 0.99)
        self.log_frequency = config.get("log_frequency", 1)
        self.log_env_image_frequency = config.get("log_env_image_frequency", 0)
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

        # ##: Timestamp for saving models (can be useful even with AIM).
        self.timestamp = int(datetime.now().timestamp())

        # ##: Log hyperparameters to AIM if logger is provided.
        if self.aim_logger:
            trainer_hparams = {
                "max_episodes": self.max_episodes,
                "max_steps_per_episode": self.max_steps_per_episode,
                "update_frequency": self.update_frequency,
                "eval_frequency": self.eval_frequency,
                "num_eval_episodes": self.num_eval_episodes,
                "gamma": self.gamma,
                "log_frequency": self.log_frequency,
                "log_env_image_frequency": self.log_env_image_frequency,
                "save_frequency": self.save_frequency,
                "save_dir": str(self.save_dir),
            }
            self.aim_logger.log_params(trainer_hparams, prefix="trainer")

    def train(self) -> Dict[str, Any]:
        """
        Train the agent in the environment.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics.
        """
        start_time = time()

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
                # ##: Log environment image periodically.
                if (
                    self.aim_logger
                    and self.log_env_image_frequency > 0
                    and self.total_steps % self.log_env_image_frequency == 0
                ):
                    try:
                        self.aim_logger.log_image(
                            observation,
                            name="environment_observation",
                            step=self.total_steps,
                            epoch=self.episode,
                            caption=f"Episode {self.episode}, Step {step}",
                        )
                    except Exception as exc:
                        logger.warning("Could not log environment image: %s", exc)

                # ##: Get action from agent.
                action, agent_info = self.agent.act(observation)
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

                    # ##: Agent learning step.
                    learn_info = self.agent.learn(
                        np_observations, np_actions, np_rewards, np_next_observations, np_dones
                    )

                    # ##: Log agent learning info if available and logger exists.
                    if self.aim_logger and learn_info and isinstance(learn_info, dict):
                        self.aim_logger.log_metrics(
                            learn_info, step=self.total_steps, epoch=self.episode, context={"subset": "train_update"}
                        )

                    # ##: Log action distribution if available.
                    if self.aim_logger and agent_info and isinstance(agent_info, dict):
                        if "action_probs" in agent_info:
                            try:
                                self.aim_logger.log_metric(
                                    name="action_probabilities",
                                    value=Distribution(agent_info["action_probs"]),
                                    step=self.total_steps,
                                    epoch=self.episode,
                                    context={"subset": "train"},
                                )
                            except Exception as exc:
                                logger.warning("Could not log action probabilities distribution: %s", exc)

                        other_agent_metrics = {k: v for k, v in agent_info.items() if k != "action_probs"}
                        scalar_agent_metrics = {k: v for k, v in other_agent_metrics.items() if np.isscalar(v)} # Filter for scalars
                        if scalar_agent_metrics: # Log only scalar metrics
                            self.aim_logger.log_metrics(
                                scalar_agent_metrics,
                                step=self.total_steps,
                                epoch=self.episode,
                                context={"subset": "train_action"},
                            )

                    observations = []
                    actions = []
                    rewards = []
                    next_observations = []
                    dones = []

                if done:
                    break

            # ##: Store the episode reward for running mean calculation.
            self.episode_rewards.append(episode_reward)
            mean_reward_100 = mean(self.episode_rewards) if self.episode_rewards else 0

            # ##: Log training progress (console).
            if (self.episode + 1) % self.log_frequency == 0:
                self._log_progress(episode_reward, episode_steps, mean_reward_100, start_time)

            # ##: Log metrics to AIM.
            if self.aim_logger:
                self.aim_logger.log_metric(
                    "episode_reward",
                    episode_reward,
                    step=self.total_steps,
                    epoch=self.episode + 1,
                    context={"subset": "train"},
                )
                self.aim_logger.log_metric(
                    "episode_steps",
                    episode_steps,
                    step=self.total_steps,
                    epoch=self.episode + 1,
                    context={"subset": "train"},
                )
                self.aim_logger.log_metric(
                    "mean_reward_100",
                    mean_reward_100,
                    step=self.total_steps,
                    epoch=self.episode + 1,
                    context={"subset": "train"},
                )

            # ##: Evaluate the agent periodically.
            if (self.episode + 1) % self.eval_frequency == 0:
                eval_metrics = self.evaluate(self.num_eval_episodes)
                mean_eval_reward = eval_metrics["mean_reward"]

                print(
                    f"Evaluation (Ep {self.episode + 1}) Mean Reward: {mean_eval_reward:.2f} "
                    f"(Min: {eval_metrics['min_reward']:.2f}, Max: {eval_metrics['max_reward']:.2f})"
                )

                # ##: Log evaluation metrics to AIM.
                if self.aim_logger:
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

                # ##: Report to Optuna for pruning if this is a trial.
                if self._pruning_callback is not None and self._trial_info is not None:
                    try:
                        self._pruning_callback(self.episode + 1, mean_eval_reward)
                    except Exception as e:
                        print(f"Trial pruned at episode {self.episode + 1}: {e}")
                        return {
                            "pruned": True,
                            "episodes": self.episode + 1,
                            "total_steps": self.total_steps,
                            "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
                            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
                        }

            # ##: Save the agent checkpoint.
            if (self.episode + 1) % self.save_frequency == 0:
                checkpoint_path = (
                    self.save_dir / f"{self.agent.name}_{self.environment.name}_{self.timestamp}_{self.episode + 1}"
                )
                self.save_checkpoint(checkpoint_path)

            # ##: Invoke callbacks.
            for callback in self.callbacks:
                callback(
                    trainer=self, episode=self.episode, episode_reward=episode_reward, episode_steps=episode_steps
                )

        # ##: Final evaluation after training loop
        final_eval_metrics = self.evaluate(self.num_eval_episodes)
        print(f"Final Evaluation Mean Reward: {final_eval_metrics['mean_reward']:.2f}")
        if self.aim_logger:
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
            for _ in range(self.max_steps_per_episode):
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
        state_path = path_obj / "trainer_state.npy"
        np.save(state_path, np.array([trainer_state], dtype=object))

        # ##: Log checkpoint artifact info to AIM.
        if self.aim_logger:
            self.aim_logger.log_artifact(
                artifact_data=trainer_state,
                name=f"checkpoint_ep{self.episode + 1}",
                path=str(path_obj.resolve()),
                meta={
                    "episode": self.episode + 1,
                    "total_steps": self.total_steps,
                    "agent_save_path": str(agent_path.resolve()),
                    "trainer_state_path": str(state_path.resolve()),
                },
            )
        print(f"Checkpoint saved to: {path_obj}")

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
        self.timestamp = trainer_state.get("timestamp", self.timestamp)

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

        print(
            f"Ep {self.episode + 1}/{self.max_episodes} "
            f"| Steps: {episode_steps} "
            f"| Reward: {episode_reward:.2f} "
            f"| Mean Rwd (100): {mean_reward_100:.2f} "
            f"| Total Steps: {self.total_steps} "
            f"| Time: {elapsed_time:.2f}s"
        )
