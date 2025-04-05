#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of running a hyperparameter search using Optuna for the A2C agent.

This example demonstrates how to use the Optuna-based hyperparameter search functionality
to efficiently optimize different combinations of hyperparameters and find
the best configuration for an agent, with support for parallel execution,
pruning of underperforming trials, and visualization.
"""

import os
import argparse

from reinforce.configs import ConfigManager
from reinforce.experiments import HyperparameterSearch


def main(search_config_path, n_trials=20, timeout=None):
    """Run a hyperparameter search example using Optuna.
    
    Args:
        search_config_path: Path to the search configuration file
        n_trials: Number of trials to run
        timeout: Optional timeout in seconds
    """
    # Create output directories
    os.makedirs("outputs/hyperparameter_search", exist_ok=True)
    
    # Create hyperparameter search
    search = HyperparameterSearch()
    
    # Run the search
    print(f"Running Optuna-based hyperparameter search with configuration: {search_config_path}")
    print(f"Number of trials: {n_trials}")
    
    results = search.run_search(
        search_config_path,
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Print results
    print("\nHyperparameter Search Results:")
    print(f"Number of experiments: {results['num_completed_trials']}")
    print(f"Best experiment: {results['best_trial_number']}")
    print(f"Best mean reward: {results['best_mean_reward']:.4f}")
    print("\nBest hyperparameters:")
    for param, value in results['best_hyperparameters'].items():
        print(f"  {param}: {value}")
    
    print("\nVisualization plots saved to outputs/hyperparameter_search/visualizations/")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run an Optuna-based hyperparameter search for an agent")
    parser.add_argument("--config", type=str, default="examples/configs/a2c_search.yaml",
                        help="Path to the search configuration file")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials to run")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds (optional)")
    args = parser.parse_args()
    
    # Create example search configuration if it doesn't exist
    if not os.path.exists(args.config):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        
        # Create configuration manager
        config_manager = ConfigManager()
        
        # Create base experiment configuration
        base_config = {
            "agent": {
                "agent_type": "A2C",
                "action_space": 7,
                "embedding_size": 128,
                "learning_rate": 0.001,
                "discount_factor": 0.99,
                "entropy_coef": 0.01,
                "value_coef": 0.5
            },
            "environment": {
                "use_image_obs": True
            },
            "trainer": {
                "trainer_type": "EpisodeTrainer",
                "max_episodes": 500,  # Reduced for the search
                "max_steps_per_episode": 100,
                "update_frequency": 1,
                "eval_frequency": 100,
                "num_eval_episodes": 5,
                "gamma": 0.99,
                "log_frequency": 100,
                "save_frequency": 500,
                "save_dir": "outputs/models"
            },
            "save_results": True,
            "results_dir": "outputs/results"
        }
        
        # Create search configuration using Optuna format
        search_config = {
            "base_config": base_config,
            "hyperparameters": {
                "agent.learning_rate": {
                    "type": "float",
                    "low": 0.0001,
                    "high": 0.01,
                    "log_scale": True
                },
                "agent.entropy_coef": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.1,
                    "log_scale": True
                },
                "agent.discount_factor": {
                    "type": "categorical",
                    "values": [0.9, 0.95, 0.99]
                },
                "agent.embedding_size": {
                    "type": "int",
                    "low": 64,
                    "high": 256
                }
            }
        }
        
        # Save the search configuration
        config_manager.save_config(search_config, args.config)
        print(f"Created example Optuna search configuration: {args.config}")
    
    # Run the example
    main(args.config, args.trials, args.timeout)
