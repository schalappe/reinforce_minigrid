# -*- coding: utf-8 -*-
"""
Experiment runner for reinforcement learning experiments.

This module provides the `ExperimentRunner` class, which orchestrates the execution of reinforcement
learning experiments. It handles configuration loading, environment setup, agent initialization,
training, and logging (via AIM).
"""

import sys
from time import time
from traceback import format_exc
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from reinforce.agents.factory import AgentFactory
from reinforce.configs.models import ExperimentConfig
from reinforce.environments.factory import EnvironmentFactory
from reinforce.learning.trainers import BaseTrainer
from reinforce.learning.trainers.factory import TrainerFactory
from reinforce.utils.logger import AimTracker, setup_logger

setup_logger()


class ExperimentRunner:
    """
    Runner for reinforcement learning experiments.

    This class sets up and runs reinforcement learning experiments based on configuration files.
    It supports logging via AIM, Optuna pruning, and saving experiment results.
    """

    @classmethod
    def _setup_experiment(
        cls,
        experiment_config: ExperimentConfig,
        pruning_callback: Optional[Callable[[int, float], None]] = None,
        aim_tags: Optional[List[str]] = None,
    ) -> Tuple[BaseTrainer, AimTracker]:
        """
        Set up the experiment components: config, logger, evaluator, checkpoint_manager, env, agent, trainer.

        Initializes logger, evaluator, checkpoint manager, environment, and agent based on the configuration,
        then creates the trainer with injected dependencies.

        Parameters
        ----------
        experiment_config : ExperimentConfig
            The experiment configuration object.
        pruning_callback : Callable[[int, float], None, optional
            Callback function for Optuna pruning. Takes (step, value) parameters.
        aim_tags : List[str], optional
            Additional tags for AIM logging.

        Returns
        -------
        Tuple[BaseTrainer, AimTracker]
            A tuple containing the initialized trainer and the tracker instance.
        """
        # ##: Initialize Tracker.
        base_tags = ["reinforce", experiment_config.agent.agent_type]
        if aim_tags:
            base_tags.extend(aim_tags)
        tracker = AimTracker(experiment_name=experiment_config.aim_experiment_name, tags=list(set(base_tags)))

        # ##: Log Pydantic models by dumping them to dicts using the logger instance.
        tracker.log_params(
            experiment_config.model_dump(exclude={"agent", "trainer", "environment"}), prefix="experiment"
        )
        tracker.log_params(experiment_config.environment.model_dump(), prefix="environment")

        # ##: Set up the environment and agent using factories.
        environment = EnvironmentFactory.create(experiment_config.environment)
        agent = AgentFactory.create(experiment_config.agent, environment.action_space.n)

        # ##: Pass pruning callback directly if Optuna trial info exists in the config.
        if experiment_config.trial_info is not None:
            experiment_config.trainer.pruning_callback = pruning_callback

        # ##: Create the trainer using the factory, injecting all dependencies.
        trainer = TrainerFactory.create(
            trainer_config=experiment_config.trainer, agent=agent, environment=environment, tracker=tracker
        )

        return trainer, tracker

    def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        pruning_callback: Optional[Callable[[int, float], None]] = None,
        aim_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run an experiment using the provided configuration.

        This method sets up an experiment, trains an agent in the environment, logs the results, and saves
        the experiment results to a file if specified in the configuration.

        Parameters
        ----------
        experiment_config : ExperimentConfig
            ExperimentConfig object containing all configuration parameters for the experiment.
        pruning_callback : callable, optional
            Callback function for Optuna pruning. Takes (step, value) parameters.
        aim_tags : List[str], optional
            Additional tags for AIM logging.

        Returns
        -------
        Dict[str, Any]
            Dictionary of experiment results, including:
            - episodes: Total number of episodes run.
            - total_steps: Total steps taken.
            - mean_reward: Average reward per episode.
            - max_reward: Maximum reward achieved.

        Raises
        ------
        Exception
            If any error occurs during experiment execution.
        """
        start_time = time()
        logger_instance: Optional[AimTracker] = None

        try:
            # ##: Setup experiment components using the helper method.
            trainer, logger_instance = self._setup_experiment(experiment_config, pruning_callback, aim_tags)

            # ##: Run the training.
            results = trainer.train()

            # ##: Calculate and log duration using the logger instance.
            end_time = time()
            duration_seconds = end_time - start_time
            if logger_instance:
                logger_instance.log_metric("experiment_duration_seconds", duration_seconds)
                logger_instance.log_params({"experiment_duration_readable": f"{duration_seconds:.2f}s"})
            logger_instance.close()
            logger.info(f"Experiment duration: {duration_seconds:.2f} seconds")

            return results

        except Exception as exc:
            logger.error(f"An error occurred during the experiment: {exc}")
            if logger_instance:
                logger_instance.log_text(f"Experiment failed: {exc}\n{format_exc()}", name="error_log")
                logger_instance.close()
            sys.exit(1)
