# -*- coding: utf-8 -*-
"""
Experiment runner for reinforcement learning experiments.

This module provides the `ExperimentRunner` class, which orchestrates the execution of reinforcement
learning experiments. It handles configuration loading, environment setup, agent initialization,
training, and logging (via AIM).
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
from time import time
from traceback import format_exc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic import ValidationError

from reinforce.agents import A2CAgent
from reinforce.configs import ConfigManager
from reinforce.configs.models import (
    A2CConfig,
    AgentConfigUnion,
    EnvironmentConfig,
    EpisodeTrainerConfig,
    ExperimentConfig,
    TrainerConfigUnion,
)
from reinforce.core import BaseAgent, BaseEnvironment, BaseTrainer
from reinforce.environments import MazeEnvironment
from reinforce.trainers import EpisodeTrainer
from reinforce.utils import AimLogger
from reinforce.utils.logging_setup import setup_logger

setup_logger()


class ExperimentRunner:
    """
    Runner for reinforcement learning experiments.

    This class sets up and runs reinforcement learning experiments based on configuration files.
    It supports logging via AIM, Optuna pruning, and saving experiment results.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the experiment runner.

        Parameters
        ----------
        config_dir : str, optional
            Directory containing configuration files. If None, defaults to built-in configurations.
        """
        self.config_manager = ConfigManager(config_dir)

    def _setup_experiment(
        self,
        experiment_config_path: Union[str, Path],
        pruning_callback: Optional[Callable[[int, float], None]] = None,
        aim_tags: Optional[List[str]] = None,
    ) -> Tuple[ExperimentConfig, BaseTrainer, Optional[AimLogger]]:
        """
        Set up the experiment components: configuration, environment, agent, trainer and logger.

        This method loads and validates the experiment configuration using Pydantic, initializes the AIM logger,
        creates the environment and agent based on the configuration, and sets up the trainer with the appropriate
        configurations.

        Parameters
        ----------
        experiment_config_path : Union[str, Path]
            Path to the experiment configuration file (YAML or JSON).
        pruning_callback : Callable[[int, float], None, optional
            Callback function for Optuna pruning. Takes (step, value) parameters.
        aim_tags : List[str], optional
            Additional tags for AIM logging.

        Returns
        -------
        Tuple[ExperimentConfig, BaseTrainer, Optional[AimLogger]]
            A tuple containing the validated Pydantic configuration object, the initialized trainer,
            and the AIM logger.

        Raises
        ------
        ValueError
            If configuration loading or validation fails.
        """
        try:
            config = self.config_manager.load_experiment_config(str(experiment_config_path))
        except (FileNotFoundError, ValueError, ValidationError) as e:
            logger.error(f"Failed to load or validate experiment config '{experiment_config_path}': {e}")
            raise ValueError(f"Configuration error: {e}") from e

        # ##: Initialize AIM Logger.
        experiment_name = Path(experiment_config_path).stem
        base_tags = ["reinforce", config.agent.agent_type]
        if aim_tags:
            base_tags.extend(aim_tags)

        aim_logger = AimLogger(
            experiment_name=config.aim_experiment_name or experiment_name, tags=list(set(base_tags))
        )

        if not aim_logger.run:
            logger.info("Failed to initialize AIM logger. Proceeding without AIM tracking.")
        else:
            # ##: Log Pydantic models by dumping them to dicts.
            aim_logger.log_params(config.model_dump(exclude={"agent", "trainer", "environment"}), prefix="experiment")
            aim_logger.log_params(config.agent.model_dump(), prefix="agent")
            aim_logger.log_params(config.environment.model_dump(), prefix="environment")
            aim_logger.log_params(
                config.trainer.model_dump(exclude={"_trial_info", "_pruning_callback"}), prefix="trainer"
            )

        # ##: Set up the environment, agent, and trainer using Pydantic config objects.
        environment = self._create_environment(config.environment)
        agent = self._create_agent(config.agent, environment.action_space.n)

        # ##: Pass pruning callback directly if Optuna trial info exists in the config.
        if config.trial_info is not None:
            config.trainer.pruning_callback = pruning_callback

        trainer = self._create_trainer(config.trainer, agent, environment, aim_logger)

        return config, trainer, aim_logger

    def run_experiment(
        self,
        experiment_config_path: Union[str, Path],
        pruning_callback: Optional[Callable[[int, float], None]] = None,
        aim_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run an experiment using the provided configuration.

        This method sets up an experiment, trains an agent in the environment, logs the results, and saves
        the experiment results to a file if specified in the configuration.

        Parameters
        ----------
        experiment_config_path : str | Path
            Path to the experiment configuration file (YAML or JSON).
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
        aim_logger = None

        try:
            # ##: Setup experiment components using the helper method.
            config, trainer, aim_logger = self._setup_experiment(experiment_config_path, pruning_callback, aim_tags)

            # ##: Run the training.
            results = trainer.train()

            # ##: Calculate and log duration.
            end_time = time()
            duration_seconds = end_time - start_time
            if aim_logger and aim_logger.run:
                aim_logger.log_metric("experiment_duration_seconds", duration_seconds)
                aim_logger.log_params({"experiment_duration_readable": f"{duration_seconds:.2f}s"})
            logger.info(f"Experiment duration: {duration_seconds:.2f} seconds")

            # ##: Save results if specified in the Pydantic config.
            if config.save_results:
                config.results_dir.mkdir(parents=True, exist_ok=True)

                experiment_name = Path(experiment_config_path).stem
                results_path = config.results_dir / f"{experiment_name}_results.json"

                # ##: save_config now expects a dict, results is already a dict.
                self.config_manager.save_config(results, str(results_path))

            return results

        except Exception as exc:
            logger.error(f"An error occurred during the experiment: {exc}")
            if aim_logger and aim_logger.run:
                aim_logger.log_text(f"Experiment failed: {exc}\n{format_exc()}", name="error_log")
                aim_logger.close()

            # ##: Ensure AIM logger is closed on error.
            if aim_logger and aim_logger.run:
                aim_logger.close()
            sys.exit(1)

    @staticmethod
    def _create_environment(env_config: EnvironmentConfig) -> MazeEnvironment:
        """
        Create an environment based on the Pydantic configuration model.

        This method creates and returns a `MazeEnvironment` instance based on the provided config model.

        Parameters
        ----------
        env_config : Dict[str, Any]
            Environment configuration.

        Returns
        -------
        MazeEnvironment
            Created environment instance.

        Notes
        -----
        Currently only supports `MazeEnvironment`. Needs update if more env types are added.
        """
        if env_config.env_type == "MazeEnvironment":
            return MazeEnvironment(use_image_obs=env_config.use_image_obs)
        raise ValueError(f"Unsupported environment type: {env_config.env_type}")

    @staticmethod
    def _create_agent(agent_config: AgentConfigUnion, env_action_space_size: int) -> BaseAgent:
        """
        Create an agent based on the Pydantic configuration model.

        This method creates and returns an agent instance based on the provided config model.
        It uses the `agent_type` field to determine which agent class to instantiate.

        Parameters
        ----------
        agent_config : AgentConfigUnion
            Pydantic agent configuration model (e.g., A2CConfig).

        Returns
        -------
        BaseAgent
            Created agent instance.

        Raises
        ------
        ValueError
            If the agent type specified in the config is not supported.
        """
        if agent_config.agent_type == "A2C":
            if isinstance(agent_config, A2CConfig):
                return A2CAgent(action_space=env_action_space_size, hyperparameters=agent_config)
            raise TypeError(f"Expected A2CConfig, got {type(agent_config)}")

        raise ValueError(f"Unsupported agent type: {agent_config.agent_type}")

    @staticmethod
    def _create_trainer(
        trainer_config: TrainerConfigUnion,
        agent: BaseAgent,
        environment: BaseEnvironment,
        aim_logger: Optional[AimLogger] = None,
    ) -> BaseTrainer:
        """
        Create a trainer based on the Pydantic configuration model.

        This method creates and returns a trainer instance based on the provided config model.
        It uses the `trainer_type` field to determine which trainer class to instantiate.

        Parameters
        ----------
        trainer_config : TrainerConfigUnion
            Pydantic trainer configuration model (e.g., EpisodeTrainerConfig).
        agent : BaseAgent
            The agent instance to be trained.
        environment : BaseEnvironment
            The environment instance to train in.
        aim_logger : AimLogger, optional
            AIM logger instance for experiment tracking.

        Returns
        -------
        BaseTrainer
            Created trainer instance.

        Raises
        ------
        ValueError
            If the trainer type specified in the config is not supported.
        """
        if trainer_config.trainer_type == "EpisodeTrainer":
            if isinstance(trainer_config, EpisodeTrainerConfig):
                return EpisodeTrainer(
                    agent=agent, environment=environment, config=trainer_config, aim_logger=aim_logger
                )
            raise TypeError(f"Expected EpisodeTrainerConfig, got {type(trainer_config)}")

        raise ValueError(f"Unsupported trainer type: {trainer_config.trainer_type}")


def main():
    """
    Run an experiment from the command line.

    Usage:
        python experiment_runner.py <config_path> [--config-dir <dir>] [--run-name <name>] [--tags <tag1> <tag2> ...]

    Example:
        python experiment_runner.py examples/configs/a2c_maze.yaml --tags baseline test
    """
    # ##: Parse command line arguments.
    parser = ArgumentParser(description="Run a reinforcement learning experiment")
    parser.add_argument(
        "config", help="Path to the experiment configuration file (e.g., examples/configs/a2c_maze.yaml)"
    )
    parser.add_argument("--config-dir", help="Directory containing base configuration files")
    parser.add_argument("--run-name", help="Custom name for the AIM run")
    parser.add_argument("--tags", nargs="+", help="Additional tags for the AIM run (e.g., --tags baseline test)")
    args = parser.parse_args()

    # ##: Create experiment runner.
    runner = ExperimentRunner(args.config_dir)

    # ##: Run the experiment with optional AIM args.
    results = runner.run_experiment(args.config, aim_tags=args.tags)

    # ##: Print summary of results.
    logger.info("Experiment complete!")
    logger.info(f"Episodes: {results.get('episodes', 0)}")
    logger.info(f"Total steps: {results.get('total_steps', 0)}")
    logger.info(f"Mean reward: {results.get('mean_reward', 0):.2f}")
    logger.info(f"Max reward: {results.get('max_reward', 0):.2f}")


if __name__ == "__main__":
    main()
