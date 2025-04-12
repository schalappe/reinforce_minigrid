# -*- coding: utf-8 -*-
"""
Generic script for running hyperparameter search using Optuna.

This script loads a search configuration from a YAML file and runs an Optuna-based hyperparameter
search for the specified agent and trainer.
"""

import os
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from reinforce.configs.manager.reader import YamlReader
from reinforce.experiments import HyperparameterSearch
from reinforce.utils.management import setup_logger

setup_logger()


def main(search_config_path: str, n_trials: int):
    """
    Run a hyperparameter search using Optuna based on a config file.

    Parameters
    ----------
    search_config_path: Path
        Path to the search configuration YAML file.
    n_trials: int
        Number of trials to run.
    """
    # Load search config directly using YamlReader
    try:
        search_config = YamlReader().read(Path(search_config_path))
        base_config = search_config.get("base_config", {})
        agent_type = base_config.get("agent", {}).get("agent_type", "unknown_agent")
        logger.info(f"Loaded search configuration for agent: {agent_type}")
    except FileNotFoundError:
        logger.error(f"Search configuration file not found: {search_config_path}")
        return
    except Exception as e:
        logger.error(f"Error loading search configuration from {search_config_path}: {e}")
        return

    # ##: Create and run hyperparameter search.
    search = HyperparameterSearch()

    logger.info(f"Running Optuna-based hyperparameter search with configuration: {search_config_path}")
    logger.info(f"Number of trials: {n_trials}")

    try:
        results = search.run_search(search_config_path, n_trials=n_trials)
    except Exception as e:
        logger.error(f"Hyperparameter search failed: {e}")
        return

    if results:
        logger.info(f"\n{agent_type} Hyperparameter Search Results:")
        logger.info(f"Number of completed trials: {results.get('num_completed_trials', 'N/A')}")
        logger.info(f"Best trial number: {results.get('best_trial_number', 'N/A')}")
        best_reward = results.get("best_mean_reward")
        if best_reward is not None:
            logger.info(f"Best mean reward: {best_reward:.4f}")
        else:
            logger.info("Best mean reward: N/A")

        logger.info("\nBest hyperparameters:")
        best_params = results.get("best_hyperparameters", {})
        if best_params:
            for param, value in best_params.items():
                logger.info(f"  {param}: {value}")
        else:
            logger.info("  No best hyperparameters found.")

        logger.info(f"\nStudy database saved in {search.results_dir}")
    else:
        logger.warning("Hyperparameter search did not return results.")


if __name__ == "__main__":
    # ##: Parse command line arguments.
    parser = ArgumentParser(description="Run an Optuna-based hyperparameter search from a configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the search configuration YAML file (e.g., examples/configs/a2c_search.yaml)",
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    args = parser.parse_args()

    # ##: Check if the specified config file exists.
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Please provide a valid path to a search configuration YAML file.")
        logger.info(
            "Example: python examples/hyperparameter_search.py --config examples/configs/a2c_search.yaml --trials 10"
        )
    else:
        main(args.config, args.trials)
