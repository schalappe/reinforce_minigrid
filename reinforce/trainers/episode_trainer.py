# -*- coding: utf-8 -*-
"""
Episode-based trainer for reinforcement learning agents.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from aim import Distribution
from loguru import logger

from reinforce.core.base_agent import BaseAgent
from reinforce.core.base_environment import BaseEnvironment
from reinforce.core.base_trainer import BaseTrainer
from reinforce.utils.aim_logger import AimLogger
from reinforce.utils.logging_setup import setup_logger

setup_logger()


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
        *,
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
        self.config = config

        # ##: Extract specific config values needed frequently or for state.
        self.save_dir = Path(self.config.get("save_dir", Path("outputs") / "models"))
        self._trial_info = self.config.get("_trial_info", None)
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
            # ##: Log relevant trainer parameters from the config.
            trainer_hparams_to_log = {
                key: value
                for key, value in self.config.items()
                if key
                in [
                    "max_episodes",
                    "max_steps_per_episode",
                    "update_frequency",
                    "eval_frequency",
                    "num_eval_episodes",
                    "gamma",
                    "log_frequency",
                    "log_env_image_frequency",
                    "save_frequency",
                ]
            }
            trainer_hparams_to_log["save_dir"] = str(self.save_dir)
            self.aim_logger.log_params(trainer_hparams_to_log, prefix="trainer")

    def train(self) -> Dict[str, Any]:
        """
        Train the agent in the environment.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics.
        """
        start_time = time()
        max_episodes = self.config.get("max_episodes", 1000)

        for self.episode in range(self.episode, max_episodes):
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
            max_steps_per_episode = self.config.get("max_steps_per_episode", 100)

            # ##: Run one episode.
            for step in range(max_steps_per_episode):
                # ##: Log environment image periodically.
                log_env_image_frequency = self.config.get("log_env_image_frequency", 0)
                if self.aim_logger and log_env_image_frequency > 0 and self.total_steps % log_env_image_frequency == 0:
                    try:
                        self.aim_logger.log_image(
                            observation,
                            name="environment_observation",
                            step=self.total_steps,
                            epoch=self.episode,
                            caption=f"Episode {self.episode}, Step {step}",
                        )
                    except Exception as exc:
                        logger.warning(f"Could not log environment image: {exc}")

                # ##: Run one step in the environment.
                next_observation, reward, done, action, agent_info = self._run_episode_step(observation, step)

                # ##: Store experience.
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_observation)
                dones.append(done)

                # ##: Update state.
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                # ##: Update the agent if it's time.
                update_frequency = self.config.get("update_frequency", 1)
                if self.total_steps % update_frequency == 0 and len(observations) > 0:
                    self._handle_agent_update(observations, actions, rewards, next_observations, dones, agent_info)
                    observations, actions, rewards, next_observations, dones = [], [], [], [], []

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
            save_frequency = self.config.get("save_frequency", 100)
            if (self.episode + 1) % save_frequency == 0:
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
        num_eval_episodes = self.config.get("num_eval_episodes", 5)
        final_eval_metrics = self.evaluate(num_eval_episodes)
        logger.info(f"Final Evaluation Mean Reward: {final_eval_metrics['mean_reward']:.2f}")
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

    def _run_episode_step(self, observation: Any, step: int) -> tuple:
        """Runs a single step within an episode."""
        # ##: Log environment image periodically.
        log_env_image_frequency = self.config.get("log_env_image_frequency", 0)
        if self.aim_logger and log_env_image_frequency > 0 and self.total_steps % log_env_image_frequency == 0:
            try:
                self.aim_logger.log_image(
                    observation,
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
        agent_info: Dict,
    ) -> None:
        """Handles the agent learning step and associated logging."""
        np_observations = np.array(observations)
        np_actions = np.array(actions)
        np_rewards = np.array(rewards)
        np_next_observations = np.array(next_observations)
        np_dones = np.array(dones)

        experience_batch = {
            "observations": np_observations,
            "actions": np_actions,
            "rewards": np_rewards,
            "next_observations": np_next_observations,
            "dones": np_dones,
        }

        learn_info = self.agent.learn(experience_batch)

        if not self.aim_logger:
            return

        # ##: Log agent learning info.
        if learn_info and isinstance(learn_info, dict):
            self.aim_logger.log_metrics(
                learn_info, step=self.total_steps, epoch=self.episode, context={"subset": "train_update"}
            )

        # ##: Log agent action info (e.g., action probabilities).
        if agent_info and isinstance(agent_info, dict):
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
                    logger.warning(f"Could not log action probabilities distribution: {exc}")

            # ##: Log other scalar metrics from agent_info.
            other_agent_metrics = {k: v for k, v in agent_info.items() if k != "action_probs"}
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
        mean_reward_100 = mean(self.episode_rewards) if self.episode_rewards else 0
        log_frequency = self.config.get("log_frequency", 1)

        # ##: Log to console.
        if (self.episode + 1) % log_frequency == 0:
            self._log_progress(episode_reward, episode_steps, mean_reward_100, start_time)

        # ##: Log to AIM.
        if self.aim_logger:
            metrics_to_log = {
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "mean_reward_100": mean_reward_100,
            }
            self.aim_logger.log_metrics(
                metrics_to_log, step=self.total_steps, epoch=self.episode + 1, context={"subset": "train"}
            )

        return mean_reward_100

    def _handle_evaluation_and_pruning(self) -> bool:
        """Handles periodic evaluation and Optuna pruning. Returns True if pruned."""
        eval_frequency = self.config.get("eval_frequency", 10)
        num_eval_episodes = self.config.get("num_eval_episodes", 5)

        if (self.episode + 1) % eval_frequency == 0:
            eval_metrics = self.evaluate(num_eval_episodes)
            mean_eval_reward = eval_metrics["mean_reward"]
            min_r, max_r = eval_metrics["min_reward"], eval_metrics["max_reward"]

            logger.info(
                f"Evaluation (Ep {self.episode + 1}) Mean Reward: {mean_eval_reward:.2f} "
                f"(Min: {min_r:.2f}, Max: {max_r:.2f})"
            )

            # ##: Log evaluation metrics to AIM.
            if self.aim_logger:
                aim_eval_metrics = {
                    "eval_mean_reward": mean_eval_reward,
                    "eval_max_reward": eval_metrics["max_reward"],
                    "eval_min_reward": eval_metrics["min_reward"],
                }
                self.aim_logger.log_metrics(
                    aim_eval_metrics, step=self.total_steps, epoch=self.episode + 1, context={"subset": "eval"}
                )

            # ##: Report to Optuna for pruning.
            if self._pruning_callback is not None and self._trial_info is not None:
                try:
                    self._pruning_callback(self.episode + 1, mean_eval_reward)
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
            max_steps_per_episode = self.config.get("max_steps_per_episode", 100)  # Access config

            # ##: Run one evaluation episode.
            for _ in range(max_steps_per_episode):
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

        # ##: Use save instead of savez for single dictionary object.
        state_path = path_obj / "trainer_state.npy"
        np.save(state_path, trainer_state)

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
        logger.info(f"Checkpoint saved to: {path_obj}")

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
        # ##: Use .item() to extract dictionary if saved within a 0-d array previously, or load directly.
        try:
            trainer_state = np.load(path_obj / "trainer_state.npy", allow_pickle=True).item()
        except AttributeError:
            trainer_state = np.load(path_obj / "trainer_state.npy", allow_pickle=True)

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
        max_episodes = self.config.get("max_episodes", 1000)

        logger.info(
            f"Ep {self.episode + 1}/{max_episodes} "
            f"| Steps: {episode_steps} "
            f"| Reward: {episode_reward:.2f} "
            f"| Mean Rwd (100): {mean_reward_100:.2f} "
            f"| Total Steps: {self.total_steps} "
            f"| Time: {elapsed_time:.2f}s"
        )
