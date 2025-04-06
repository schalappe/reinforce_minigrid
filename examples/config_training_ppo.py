#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of training a PPO agent using configuration files.

This example demonstrates how to use the configuration management system
to load and run an experiment defined in a YAML file using the PPO algorithm.
"""

import argparse
import os
from pathlib import Path

from reinforce.configs import ConfigManager
from reinforce.experiments import ExperimentRunner


def main(config_path):
    """Run a PPO training example using configuration files.

    Args:
        config_path: Path to the experiment configuration file
    """
    # Create output directories (ensure trainer's save_dir is handled)
    # The trainer config now specifies save_dir, so direct creation might be redundant
    # but doesn't hurt.
    os.makedirs("outputs/models/ppo", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    # Create experiment runner
    runner = ExperimentRunner()

    # Run the experiment
    print(f"Running PPO experiment with configuration: {config_path}")
    results = runner.run_experiment(config_path)

    # Print results
    print("\nPPO Experiment Results:")
    if results:
        print(f"  Episodes: {results.get('episodes', 'N/A')}")
        print(f"  Total steps: {results.get('total_steps', 'N/A')}")
        print(f"  Mean reward (last 100): {results.get('mean_reward', 'N/A'):.2f}")
        print(f"  Max reward (last 100): {results.get('max_reward', 'N/A'):.2f}")
    else:
        print("  Experiment did not return results (possibly pruned or error).")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a PPO agent using configuration files")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/configs/ppo_maze.yaml",
        help="Path to the PPO experiment configuration file",
    )
    args = parser.parse_args()

    config_file_path = Path(args.config)

    # Create example PPO configuration if it doesn't exist
    if not config_file_path.exists():
        # Create directory if it doesn't exist
        config_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create configuration manager
        config_manager = ConfigManager()

        # Create default PPO experiment configuration dictionary
        # Note: action_space will be overridden by the environment if not specified or incorrect
        config = {
            "agent": {
                "agent_type": "PPO",  # Use PPO agent
                "action_space": 7,  # Placeholder, will be set by env
                "embedding_size": 128,
                "learning_rate": 3e-4,
                "discount_factor": 0.99,  # gamma
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "max_grad_norm": 0.5,
                "use_adaptive_kl": False,
                "target_kl": 0.01,
                "initial_kl_coeff": 1.0,
            },
            "environment": {
                "env_id": "MiniGrid-Maze-9x9-v0",  # Example environment ID
                "use_image_obs": True,
                "max_steps": 500,  # Match trainer's max_steps_per_episode
            },
            "trainer": {
                "trainer_type": "PPOTrainer",  # Use PPOTrainer
                "n_steps": 2048,  # Rollout length
                "n_epochs": 10,  # Update epochs per rollout
                "batch_size": 64,  # Minibatch size
                "max_total_steps": 1_000_000,  # Total training steps
                "max_steps_per_episode": 500,  # Max steps in env episode
                "eval_frequency": 50,  # Evaluate every N episodes
                "num_eval_episodes": 10,  # Num episodes per evaluation
                "log_frequency": 1,  # Log progress every N episodes
                "save_frequency": 100,  # Save checkpoint every N episodes
                "save_dir": "outputs/models/ppo",  # PPO specific save dir
                # gamma is usually taken from agent config
            },
            "aim_experiment_name": "ppo_maze_example"  # Optional AIM name
            # Removed save_results and results_dir as AIM handles logging
        }

        # Save the configuration
        config_manager.save_config(config, str(config_file_path))
        print(f"Created example PPO configuration: {config_file_path}")

    # Run the example
    main(args.config)
