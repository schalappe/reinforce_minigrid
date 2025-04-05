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

from reinforce.agents import A2CAgent
from reinforce.configs import ConfigManager
from reinforce.core import BaseAgent, BaseEnvironment
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
    ) -> Tuple[Dict[str, Any], EpisodeTrainer, Optional[AimLogger]]:
        """
        Set up the experiment components: configuration, environment, agent, trainer and logger.

        This method loads the experiment configuration, initializes the AIM logger, creates the environment and agent
        based on the configuration, and sets up the trainer with the appropriate configurations.

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
        Tuple[Dict[str, Any], EpisodeTrainer, Optional[AimLogger]]
            A tuple containing the experiment configuration, the initialized trainer, and the AIM logger.
        """
        config = self.config_manager.load_experiment_config(str(experiment_config_path))
        experiment_name = Path(experiment_config_path).stem

        # ##: Initialize AIM Logger.
        base_tags = ["reinforce", config.get("agent", {}).get("agent_type", "unknown_agent")]
        if aim_tags:
            base_tags.extend(aim_tags)

        aim_logger = AimLogger(
            experiment_name=config.get("aim_experiment_name", experiment_name), tags=list(set(base_tags))
        )

        if not aim_logger.run:
            logger.info("Failed to initialize AIM logger. Proceeding without AIM tracking.")
        else:
            aim_logger.log_params(config, prefix="experiment_config")
            aim_logger.log_params(config.get("agent", {}), prefix="agent_config")
            aim_logger.log_params(config.get("environment", {}), prefix="env_config")

        # ##: Check if this is an Optuna trial.
        trial_info = config.get("_trial_info", None)

        # ##: Set up the environment, agent, and trainer.
        environment = self._create_environment(config.get("environment", {}))
        agent = self._create_agent(config.get("agent", {}), environment)

        # ## Add pruning support to trainer config if needed.
        trainer_config = config.get("trainer", {}).copy()
        if trial_info is not None:
            trainer_config["_trial_info"] = trial_info
            trainer_config["_pruning_callback"] = pruning_callback

        trainer = self._create_trainer(trainer_config, agent, environment, aim_logger)

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

            # ##: Save results if specified.
            if "save_results" in config and config["save_results"]:
                results_dir = Path(config.get("results_dir", "outputs/results"))
                results_dir.mkdir(parents=True, exist_ok=True)

                experiment_name = Path(experiment_config_path).stem
                results_path = results_dir / f"{experiment_name}_results.json"

                self.config_manager.save_config(results, str(results_path))

            return results

        except Exception as exc:
            logger.error(f"An error occurred during the experiment: {exc}")
            if aim_logger and aim_logger.run:
                aim_logger.log_text(f"Experiment failed: {exc}\n{format_exc()}", name="error_log")
                aim_logger.close()
            sys.exit(1)

    @staticmethod
    def _create_environment(env_config: Dict[str, Any]) -> MazeEnvironment:
        """
        Create an environment based on the configuration.

        This method creates and returns a `MazeEnvironment` instance based on the provided configuration.

        Parameters
        ----------
        env_config : Dict[str, Any]
            Environment configuration.

        Returns
        -------
        MazeEnvironment
            Created environment.

        Notes
        -----
        Currently only supports `MazeEnvironment`.
        """
        use_image_obs = env_config.get("use_image_obs", True)
        return MazeEnvironment(use_image_obs=use_image_obs)

    @staticmethod
    def _create_agent(agent_config: Dict[str, Any], environment) -> A2CAgent:
        """
        Create an agent based on the configuration.

        This method creates and returns an agent instance based on the provided configuration.
        Currently, only A2C agents are supported.

        Parameters
        ----------
        agent_config : Dict[str, Any]
            Agent configuration.
        environment : Environment
            Environment to use.

        Returns
        -------
        A2CAgent
            Created agent.

        Raises
        ------
        ValueError
            If the agent type is not supported.
        """
        agent_type = agent_config.get("agent_type", "A2C")

        if agent_type == "A2C":
            action_space = agent_config.get("action_space", environment.action_space.n)

            # ##: Extract hyperparameters expected by A2CAgent from the config.
            hyperparams = {
                "embedding_size": agent_config.get("embedding_size", 128),
                "learning_rate": agent_config.get("learning_rate", 0.001),
                "discount_factor": agent_config.get("discount_factor", 0.99),
                "entropy_coef": agent_config.get("entropy_coef", 0.01),
                "value_coef": agent_config.get("value_coef", 0.5),
            }

            return A2CAgent(action_space=action_space, hyperparameters=hyperparams)

        raise ValueError(f"Unsupported agent type: {agent_type}")

    @staticmethod
    def _create_trainer(
        trainer_config: Dict[str, Any],
        agent: BaseAgent,
        environment: BaseEnvironment,
        aim_logger: Optional[AimLogger] = None,
    ) -> EpisodeTrainer:
        """
        Create a trainer based on the configuration.

        This method creates and returns a trainer instance based on the provided configuration.
        Currently, only `EpisodeTrainer` is supported.

        Parameters
        ----------
        trainer_config : Dict[str, Any]
            Trainer configuration.
        agent : Agent
            Agent to train.
        environment : Environment
            Environment to train in.
        aim_logger : AimLogger, optional
            Logger for tracking metrics.

        Returns
        -------
        EpisodeTrainer
            Created trainer.

        Raises
        ------
        ValueError
            If the trainer type is not supported.
        """
        trainer_type = trainer_config.get("trainer_type", "EpisodeTrainer")

        if trainer_type == "EpisodeTrainer":
            return EpisodeTrainer(agent=agent, environment=environment, config=trainer_config, aim_logger=aim_logger)

        raise ValueError(f"Unsupported trainer type: {trainer_type}")


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
