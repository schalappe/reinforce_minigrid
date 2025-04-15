# -*- coding: utf-8 -*-
"""Configuration management for training and evaluation."""

from typing import Any, Dict

# Default hyperparameters for PPO training
DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "env_id": "Maze-v0",  # Environment ID (though Maze is hardcoded for now)
    "epochs": 50,
    "steps_per_epoch": 4000,
    "save_freq": 10,  # Save weights every N epochs
    "render": False,  # Render environment during training
    "seed": 42,
    "save_dir": "checkpoints",
    # PPO Agent Hyperparameters
    "gamma": 0.99,
    "lam": 0.97,  # GAE lambda
    "clip_ratio": 0.2,
    "policy_learning_rate": 3e-4,
    "value_function_learning_rate": 1e-3,
    "train_policy_iterations": 80,
    "train_value_iterations": 80,
    "target_kl": 0.01,
    # Network Parameters
    "conv_filters": 32,
    "conv_kernel_size": 3,
    "dense_units": 128,
}

# Default configuration for evaluation
DEFAULT_EVAL_CONFIG: Dict[str, Any] = {
    "env_id": "Maze-v0",
    "num_episodes": 10,  # Number of episodes to evaluate
    "render": False,  # Render environment during evaluation (usually False, but can be overridden)
    "seed": 123,  # Use a different seed for evaluation than training
    "load_dir": "checkpoints",  # Directory containing saved weights
    "weights_prefix": "ppo_maze_epoch_50",  # Specific weights file prefix (e.g., from last epoch)
    "max_episode_steps": 200,  # Maximum steps per evaluation episode (Note: Maze might have its own limit)
    "save_gif": False,  # Whether to save a GIF of the evaluation episodes
    "gif_path": "evaluation.gif",
    # Network parameters must match the saved model
    "conv_filters": 32,
    "conv_kernel_size": 3,
    "dense_units": 128,
}

# TODO: Add functions for parsing command-line arguments and merging with defaults
# def parse_train_args() -> Dict[str, Any]:
#     ...

# def parse_eval_args() -> Dict[str, Any]:
#     ...
