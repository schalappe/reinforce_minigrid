#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of running a hyperparameter search for the A2C agent.

This example demonstrates how to use the hyperparameter search functionality
to automatically test different combinations of hyperparameters and find
the best configuration for an agent.
"""

import os
import argparse
import json
import yaml

from reinforce.configs import ConfigManager
from reinforce.experiments import HyperparameterSearch


def main(search_config_path):
    """Run a hyperparameter search example.
    
    Args:
        search_config_path: Path to the search configuration file
    """
    # Create output directories
    os.makedirs("outputs/hyperparameter_search", exist_ok=True)
    
    # Create hyperparameter search
    search = HyperparameterSearch()
    
    # Run the search
    print(f"Running hyperparameter search with configuration: {search_config_path}")
    results = search.run_search(search_config_path)
    
    # Print results
    print("\nHyperparameter Search Results:")
    print(f"Number of experiments: {results['num_experiments']}")
    print(f"Best experiment: {results['best_experiment']}")
    print(f"Best mean reward: {results['best_mean_reward']:.2f}")
    print("\nBest hyperparameters:")
    for param, value in results['best_hyperparameters'].items():
        print(f"  {param}: {value}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a hyperparameter search for an agent")
    parser.add_argument("--config", type=str, default="examples/configs/a2c_search.yaml",
                        help="Path to the search configuration file")
    args = parser.parse_args()
    
    # Create example search configuration if it doesn't exist
    if not os.path.exists(args.config):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        
        # Create configuration manager
        config_manager = ConfigManager()
        
        # Create base experiment configuration (same as in config_training.py)
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
                "max_episodes": 1000,  # Reduced for the search
                "max_steps_per_episode": 100,
                "update_frequency": 1,
                "eval_frequency": 100,
                "num_eval_episodes": 5,
                "gamma": 0.99,
                "log_frequency": 100, # Less logging during search
                "save_frequency": 1000, # Less saving during search
                "save_dir": "outputs/models"
            },
            "save_results": True,
            "results_dir": "outputs/results"
        }
        
        # Create search configuration
        search_config = {
            "base_config": base_config,
            "hyperparameters": {
                "agent.learning_rate": [0.0001, 0.0005, 0.001, 0.005],
                "agent.entropy_coef": [0.001, 0.01, 0.05],
                "agent.discount_factor": [0.95, 0.99]
            }
        }
        
        # Save the search configuration
        config_manager.save_config(search_config, args.config)
        print(f"Created example search configuration: {args.config}")
    
    # Run the example
    main(args.config)
