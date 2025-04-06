# -*- coding: utf-8 -*-
"""
Example of running a hyperparameter search using Optuna for the A2C agent.

This script adapts the general hyperparameter search example to specifically
optimize hyperparameters for the A2C agent using the EpisodeTrainer.
"""

import argparse
import os

from loguru import logger

from reinforce.configs import ConfigManager
from reinforce.experiments import HyperparameterSearch
from reinforce.utils import setup_logger

setup_logger()


def main(search_config_path, n_trials=20):
    """Run an A2C hyperparameter search using Optuna.

    Args:
        search_config_path: Path to the A2C search configuration file.
        n_trials: Number of trials to run.
    """
    # Create output directories
    output_dir = "outputs/hyperparameter_search/a2c"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)  # Ensure viz dir exists

    # Create hyperparameter search, specifying the output directory
    search = HyperparameterSearch(output_dir=output_dir)

    # Run the search
    logger.info(f"Running Optuna-based hyperparameter search for A2C with configuration: {search_config_path}")
    logger.info(f"Number of trials: {n_trials}")

    results = search.run_search(search_config_path, n_trials=n_trials)

    # Print results
    logger.info("\nA2C Hyperparameter Search Results:")
    logger.info(f"Number of completed trials: {results['num_completed_trials']}")
    logger.info(f"Best trial number: {results['best_trial_number']}")
    logger.info(f"Best mean reward: {results['best_mean_reward']:.4f}")
    logger.info("\nBest hyperparameters:")
    for param, value in results["best_hyperparameters"].items():
        logger.info(f"  {param}: {value}")

    logger.info(f"\nVisualization plots saved to {os.path.join(output_dir, 'visualizations')}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run an Optuna-based hyperparameter search for the A2C agent")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/configs/a2c_search.yaml",
        help="Path to the A2C search configuration file",
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    args = parser.parse_args()

    # Create example search configuration if it doesn't exist
    if not os.path.exists(args.config):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.config), exist_ok=True)

        # Create configuration manager
        config_manager = ConfigManager()

        # Create base experiment configuration for A2C
        base_config = {
            "agent": {
                "agent_type": "A2C",
                "action_space": 7,  # Assuming MiniGrid default
                "embedding_size": 128,  # Default, will be overridden by search if included
                "learning_rate": 0.001,  # Default, will be overridden by search
                "discount_factor": 0.99,  # Default, will be overridden by search
                "entropy_coef": 0.01,  # Default, will be overridden by search
                "value_coef": 0.5,  # Default, will be overridden by search
            },
            "environment": {"use_image_obs": True},  # Assuming image observations
            "trainer": {
                "trainer_type": "EpisodeTrainer",
                "max_episodes": 100,  # Reduced for faster search trials
                "max_steps_per_episode": 100,  # Reduced for faster search trials
                "update_frequency": 1,  # A2C typically updates frequently
                "eval_frequency": 20,  # Evaluate more often during search
                "num_eval_episodes": 5,
                "gamma": 0.99,  # Often tied to agent's discount_factor
                "log_frequency": 10,
                "log_env_image_frequency": 0,  # Disable image logging during search
                "save_frequency": 0,  # Disable model saving during search
                "save_dir": "outputs/models/a2c_search_temp",  # Temporary dir
            },
        }

        # Create A2C search configuration using Optuna format
        search_config = {
            "base_config": base_config,
            "hyperparameters": {
                "agent.learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log_scale": True},
                "agent.entropy_coef": {"type": "float", "low": 1e-4, "high": 0.1, "log_scale": True},
                "agent.discount_factor": {"type": "categorical", "values": [0.9, 0.95, 0.99, 0.995]},
                "agent.embedding_size": {
                    "type": "int",
                    "low": 64,
                    "high": 256,
                    "step": 64,  # Suggest powers of 2 or similar steps
                },
                "agent.value_coef": {"type": "float", "low": 0.25, "high": 0.75},
            },
        }

        # Save the search configuration
        config_manager.save_config(search_config, args.config)
        logger.info(f"Created example A2C Optuna search configuration: {args.config}")

    # Run the A2C search example
    main(args.config, args.trials)
