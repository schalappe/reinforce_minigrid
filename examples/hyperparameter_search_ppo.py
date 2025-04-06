# -*- coding: utf-8 -*-
"""
Example of running a hyperparameter search using Optuna for the PPO agent.

This script adapts the general hyperparameter search example to specifically
optimize hyperparameters for the PPO agent using the PPOTrainer.
"""

import argparse
import os

from loguru import logger

from reinforce.configs import ConfigManager
from reinforce.experiments import HyperparameterSearch
from reinforce.utils import setup_logger

setup_logger()


def main(search_config_path, n_trials=20):
    """Run a PPO hyperparameter search using Optuna.

    Args:
        search_config_path: Path to the PPO search configuration file.
        n_trials: Number of trials to run.
    """
    # Create output directories
    output_dir = "outputs/hyperparameter_search/ppo"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)  # Ensure viz dir exists

    # Create hyperparameter search, specifying the output directory
    search = HyperparameterSearch(output_dir=output_dir)

    # Run the search
    logger.info(f"Running Optuna-based hyperparameter search for PPO with configuration: {search_config_path}")
    logger.info(f"Number of trials: {n_trials}")

    results = search.run_search(search_config_path, n_trials=n_trials)

    # Print results
    logger.info("\nPPO Hyperparameter Search Results:")
    logger.info(f"Number of completed trials: {results['num_completed_trials']}")
    logger.info(f"Best trial number: {results['best_trial_number']}")
    logger.info(f"Best mean reward: {results['best_mean_reward']:.4f}")
    logger.info("\nBest hyperparameters:")
    for param, value in results["best_hyperparameters"].items():
        logger.info(f"  {param}: {value}")

    logger.info(f"\nVisualization plots saved to {os.path.join(output_dir, 'visualizations')}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run an Optuna-based hyperparameter search for the PPO agent")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/configs/ppo_search.yaml",
        help="Path to the PPO search configuration file",
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    args = parser.parse_args()

    # Create example search configuration if it doesn't exist
    if not os.path.exists(args.config):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.config), exist_ok=True)

        # Create configuration manager
        config_manager = ConfigManager()

        # Create base experiment configuration for PPO
        base_config = {
            "agent": {
                "agent_type": "PPO",
                "action_space": 7,  # Assuming MiniGrid default
                "embedding_size": 128,  # Default, will be overridden by search if included
                "learning_rate": 3e-4,  # Default, will be overridden by search
                "discount_factor": 0.99,  # Default, will be overridden by search
                "gae_lambda": 0.95,  # Default, will be overridden by search
                "clip_range": 0.2,  # Default, will be overridden by search
                "entropy_coef": 0.01,  # Default, will be overridden by search
                "value_coef": 0.5,  # Default, will be overridden by search
                "max_grad_norm": 0.5,  # Default, maybe search this too?
            },
            "environment": {"use_image_obs": True},  # Assuming image observations
            "trainer": {
                "trainer_type": "PPOTrainer",  # Use PPOTrainer
                "n_steps": 2048,  # Default, will be overridden by search
                "n_epochs": 10,  # Default, will be overridden by search
                "batch_size": 64,  # Default, will be overridden by search
                "max_total_steps": 50000,  # Reduced total steps for faster search trials
                "max_steps_per_episode": 100,  # Reduced for faster search trials
                "eval_frequency": 5,  # Evaluate more often during search (based on episodes/rollouts)
                "num_eval_episodes": 5,
                "log_frequency": 1,
                "log_env_image_frequency": 0,  # Disable image logging during search
                "save_frequency": 0,  # Disable model saving during search
                "save_dir": "outputs/models/ppo_search_temp",  # Temporary dir
            },
        }

        # Create PPO search configuration using Optuna format
        search_config = {
            "base_config": base_config,
            "hyperparameters": {
                # Agent Hyperparameters
                "agent.learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log_scale": True},
                "agent.entropy_coef": {"type": "float", "low": 0.0, "high": 0.05},  # Can be zero for PPO
                "agent.value_coef": {"type": "float", "low": 0.3, "high": 0.7},
                "agent.discount_factor": {"type": "categorical", "values": [0.95, 0.99, 0.995]},
                "agent.embedding_size": {"type": "int", "low": 64, "high": 256, "step": 64},
                "agent.gae_lambda": {"type": "float", "low": 0.9, "high": 0.99},
                "agent.clip_range": {"type": "float", "low": 0.1, "high": 0.3},
                # Trainer Hyperparameters
                "trainer.n_steps": {"type": "int", "low": 512, "high": 4096, "step": 512},  # Rollout buffer size
                "trainer.n_epochs": {"type": "int", "low": 5, "high": 20},  # Optimization epochs per rollout
                "trainer.batch_size": {"type": "categorical", "values": [32, 64, 128]},  # Minibatch size
            },
        }

        # Save the search configuration
        config_manager.save_config(search_config, args.config)
        logger.info(f"Created example PPO Optuna search configuration: {args.config}")

    # Run the PPO search example
    main(args.config, args.trials)
