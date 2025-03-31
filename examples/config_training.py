#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of training an A2C agent using configuration files.

This example demonstrates how to use the configuration management system
to load and run an experiment defined in a YAML file.
"""

import os
import argparse

from reinforce.configs import ConfigManager
from reinforce.experiments import ExperimentRunner


def main(config_path):
    """Run a training example using configuration files.
    
    Args:
        config_path: Path to the experiment configuration file
    """
    # Create output directories
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Run the experiment
    print(f"Running experiment with configuration: {config_path}")
    results = runner.run_experiment(config_path)
    
    # Print results
    print("Experiment Results:")
    print(f"Episodes: {results['episodes']}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Mean reward: {results['mean_reward']:.2f}")
    print(f"Max reward: {results['max_reward']:.2f}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train an agent using configuration files")
    parser.add_argument("--config", type=str, default="examples/configs/a2c_maze.yaml",
                        help="Path to the experiment configuration file")
    args = parser.parse_args()
    
    # Create example configuration if it doesn't exist
    if not os.path.exists(args.config):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        
        # Create configuration manager
        config_manager = ConfigManager()
        
        # Create experiment configuration
        config = {
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
                "max_episodes": 1000,
                "max_steps_per_episode": 100,
                "update_frequency": 1,
                "eval_frequency": 100,
                "num_eval_episodes": 5,
                "gamma": 0.99,
                "log_frequency": 10,
                "save_frequency": 500,
                "save_dir": "outputs/models"
            },
            "save_results": True,
            "results_dir": "outputs/results"
        }
        
        # Save the configuration
        config_manager.save_config(config, args.config)
        print(f"Created example configuration: {args.config}")
    
    # Run the example
    main(args.config)
