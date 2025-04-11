# -*- coding: utf-8 -*-
"""
Script to run a single reinforcement learning experiment using ExperimentRunner.
"""

import argparse
from pathlib import Path
import sys

from loguru import logger

from reinforce.configs import ConfigManager
from reinforce.experiments import ExperimentRunner
from reinforce.utils import setup_logger

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

    # ##: Load Experiment Configuration.
    experiment_config = ConfigManager.load_experiment_config(config_path)
    logger.info("Experiment configuration loaded successfully.")
    logger.info(f"Agent Type: {experiment_config.agent.agent_type}, Trainer Type: {experiment_config.trainer.trainer_type}")

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
    parser = argparse.ArgumentParser(description="Run a single RL experiment using a config file.")
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
