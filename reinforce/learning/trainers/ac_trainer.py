# -*- coding: utf-8 -*-
"""
Base trainer for Actor-Critic style algorithms (A2C, PPO).
"""

from abc import abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import time
from typing import Any, Dict, Tuple, Union

from aim import Distribution
from loguru import logger
from numpy import isscalar, ndarray

from reinforce.agents.actor_critic import A2CAgent, PPOAgent
from reinforce.configs.models.trainer import A2CTrainerConfig, PPOTrainerConfig
from reinforce.environments import BaseEnvironment
from reinforce.learning.evaluation import evaluate_agent
from reinforce.learning.trainers.base_trainer import BaseTrainer
from reinforce.utils.management import AimTracker
from reinforce.utils.persistence import save_checkpoint

ACAgent = Union[A2CAgent, PPOAgent]
ACConfig = Union[A2CTrainerConfig, PPOTrainerConfig]


class ActorCriticTrainer(BaseTrainer):
    """
    Base class for trainers implementing Actor-Critic style algorithms.

    Handles common logic like initialization, state tracking, logging, evaluation, pruning,
    checkpointing, and finalization.
    """

    def __init__(self, *, agent: ACAgent, environment: BaseEnvironment, config: ACConfig, tracker: AimTracker):
        """
        Initialize the base Actor-Critic trainer.

        Parameters
        ----------
        agent : A2CAgent | PPOAgent
            The agent to train.
        environment : BaseEnvironment
            The environment to train in.
        config : A2CTrainerConfig | PPOTrainerConfig
            Pydantic configuration model for the trainer.
        tracker : AimTracker
            Tracker instance for experiment tracking.
        """
        self.agent = agent
        self.environment = environment
        self.tracker = tracker
        self.config = config

        # ##: Ensure the save directory exists.
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        # ##: Initialize training state.
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps_deque = deque(maxlen=100)

        # ##: Timestamp for saving models.
        self.timestamp = int(datetime.now().timestamp())

        # ##: Log hyperparameters.
        self._log_hyperparameters()

    def _log_hyperparameters(self):
        """Logs trainer and agent hyperparameters."""
        self.tracker.log_params(
            self.config.model_dump(exclude={"trial_info", "pruning_callback", "trainer_type"}), prefix="trainer"
        )
        self.tracker.log_params(self.agent.hyperparameters.model_dump(), prefix="agent")

    def _log_episode_metrics(self, episode_reward: float, episode_steps: int):
        """
        Logs metrics at the end of an episode.

        Parameters
        ----------
        episode_reward : float
            Total reward accumulated in the episode.
        episode_steps : int
            Number of steps taken in the episode.
        """
        self.episode_rewards.append(episode_reward)
        self.episode_steps_deque.append(episode_steps)
        mean_reward_100 = mean(self.episode_rewards) if self.episode_rewards else 0
        self.tracker.log_metrics(
            {"episode_reward": episode_reward, "episode_steps": episode_steps, "mean_reward_100": mean_reward_100},
            step=self.total_steps,
            epoch=self.episode,
            context={"subset": "train"},
        )

    def _log_console_progress(self, start_time: float, episode_reward: float, episode_steps: int):
        """
        Logs progress to the console periodically.

        Parameters
        ----------
        start_time : float
            Start time of the training.
        episode_reward : float
            Total reward accumulated in the episode.
        episode_steps : int
            Number of steps taken in the episode.
        """
        if self.episode % self.config.log_frequency == 0:
            elapsed_time = time() - start_time
            mean_reward_100 = mean(self.episode_rewards) if self.episode_rewards else 0
            logger.info(
                f"Step {self.total_steps}/{self.config.get_max_steps()} | Ep {self.episode} "
                f"| Ep Steps: {episode_steps} "
                f"| Ep Reward: {episode_reward:.2f} "
                f"| Mean Rwd (100): {mean_reward_100:.2f} "
                f"| Time: {elapsed_time:.2f}s"
            )

    def _run_evaluation_and_pruning(self) -> bool:
        """Runs evaluation and handles Optuna pruning. Returns True if pruned."""
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

            # ##: Log evaluation metrics.
            self.tracker.log_metrics(
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
                    self.config.pruning_callback(self.episode, mean_eval_reward)
                except Exception as exc:
                    logger.warning(f"Trial pruned at episode {self.episode}: {exc}")
                    return True
        return False

    def _save_checkpoint(self):
        """Saves agent and trainer state checkpoint."""
        if self.episode > 0 and self.episode % self.config.save_frequency == 0:
            checkpoint_path_base = (
                self.config.save_dir
                / f"{self.agent.name}_{self.environment.name}_{self.config.trainer_type}_{self.timestamp}_{self.episode}"
            )
            trainer_state = {"episode": self.episode, "total_steps": self.total_steps, "timestamp": self.timestamp}
            save_checkpoint(
                agent=self.agent,
                save_path_base=checkpoint_path_base,
                trainer_state=trainer_state,
                tracker=self.tracker,
            )

    def _finalize_training(self) -> Dict[str, Any]:
        """
        Performs final evaluation, logging, and model saving.

        Returns
        --------
        Dict[str, Any]
            Final evaluation metrics.
        """
        # ##: Final evaluation.
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
            epoch=self.episode,
            context={"subset": "final_eval"},
        )
        self.tracker.log_params({"final_total_steps": self.total_steps, "final_episodes": self.episode})

        # ##: Save the final model if path specified.
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
            "episodes": self.episode,
            "total_steps": self.total_steps,
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
            "mean_reward": mean(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": min(self.episode_rewards) if self.episode_rewards else 0,
        }

    def _log_agent_update_metrics(self, learn_info: Dict[str, Any]):
        """
        Logs metrics returned by agent.learn().

        Parameters
        ----------
        learn_info : Dict[str, Any]
            Dictionary of metrics returned by agent.learn().
        """
        loggable_info = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in learn_info.items()}
        self.tracker.log_metrics(
            loggable_info, step=self.total_steps, epoch=self.episode, context={"subset": "train_update"}
        )

    def _log_agent_action_metrics(self, agent_info: Dict[str, Any]):
        """
        Logs metrics returned by agent.act() (e.g., action probabilities).

        Parameters
        ----------
        agent_info : Dict[str, Any]
            Dictionary of metrics returned by agent.act().
        """
        if "action_probs" in agent_info:
            probs = agent_info["action_probs"]
            if hasattr(probs, "numpy"):
                probs = probs.numpy()

            self.tracker.log_metric(
                name="action_probabilities",
                value=Distribution(probs),
                step=self.total_steps,
                epoch=self.episode,
                context={"subset": "train_action"},
            )

        other_agent_metrics = {k: v for k, v in agent_info.items() if k != "action_probs"}
        scalar_agent_metrics = {
            k: v.numpy() if hasattr(v, "numpy") else v
            for k, v in other_agent_metrics.items()
            if isscalar(v) or hasattr(v, "numpy") and v.ndim == 0
        }
        if scalar_agent_metrics:
            self.tracker.log_metrics(
                scalar_agent_metrics,
                step=self.total_steps,
                epoch=self.episode,
                context={"subset": "train_action"},
            )

    def _run_environment_step(self, observation: ndarray) -> Tuple[ndarray, float, bool, int, Dict[str, Any]]:
        """
        Runs a single step in the environment.

        Parameters
        ----------
        observation : ndarray
            The current observation from the environment.

        Returns
        -------
        Tuple[np.ndarray, float, bool, int, Dict[str, Any]]
            The next observation, reward, done flag, action, and agent info.
        """
        action, agent_info = self.agent.act(observation, training=True)
        self._log_agent_action_metrics(agent_info)

        next_observation, reward, done, _ = self.environment.step(action)
        return next_observation, reward, done, action, agent_info

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the agent in the environment. Must be implemented by subclasses.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics.
        """
        raise NotImplementedError
