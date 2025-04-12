# -*- coding: utf-8 -*-
"""
Script to run a single reinforcement learning experiment using ExperimentRunner.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from reinforce.configs.manager import ConfigManager
from reinforce.configs.models.experiment import ExperimentConfig
from reinforce.experiments import ExperimentRunner
from reinforce.utils.management import setup_logger

# Setup logger
setup_logger()


def main(config_path: str):
    """
    Loads an experiment configuration and runs it using ExperimentRunner.

    Parameters
    ----------
    config_path : str
        Path to the experiment configuration file.
    """
    logger.info(f"Starting experiment run with configuration: {config_path}")

    runner = ExperimentRunner()

    # ##: Load and Validate Experiment Configuration using Pydantic model.
    try:
        experiment_config = ConfigManager.load_and_validate(config_path, ExperimentConfig)
        logger.info("Experiment configuration loaded and validated successfully.")
        logger.info(
            f"Agent Type: {experiment_config.agent.agent_type}, Trainer Type: {experiment_config.trainer.trainer_type}"
        )
    except ValueError as e:
        logger.error(f"Failed to load or validate configuration from {config_path}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # ##: Run the Experiment.
    logger.info("Running experiment...")
    results = runner.run_experiment(experiment_config=experiment_config)

    logger.info("Experiment finished successfully.")
    logger.info("--- Experiment Results ---")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("-------------------------")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run a single RL experiment using a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration YAML file (e.g., training/configs/a2c_train.yaml).",
    )
    args = parser.parse_args()

    config_file_path = Path(args.config)
    if not config_file_path.is_file():
        logger.error(f"Configuration file not found: {config_file_path}")
        logger.info("Please provide a valid path to an experiment configuration YAML file.")
        logger.info("Example: python training/train_agent.py --config training/configs/a2c_train.yaml")
        sys.exit(1)
    else:
        main(str(config_file_path))
