# -*- coding: utf-8 -*-
"""
Configuration management for training and evaluation.

Handles default configurations, loading from YAML files, and merging with command-line arguments.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger

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


def load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Parameters
    ----------
    config_path : Optional[str]
        Path to the YAML configuration file. If None or the file doesn't exist,
        returns an empty dictionary.

    Returns
    -------
    Dict[str, Any]
        The configuration loaded from the YAML file, or an empty dict if no path is provided
        or the file is not found.
    """
    if config_path is None:
        logger.info("No YAML config path provided.")
        return {}

    path = Path(config_path)
    if not path.is_file():
        logger.warning(f"YAML config file not found at: {config_path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from YAML: {config_path}")
            return yaml_config if yaml_config else {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error reading YAML file {config_path}: {e}")
        return {}


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Creates a base argument parser with common arguments like --config.

    Parameters
    ----------
    description : str
       Description of the parser.

    Returns
    -------
    argparse.ArgumentParser
        The base argument parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    return parser


def get_train_config() -> Dict[str, Any]:
    """
    Parses args, loads YAML, merges, and returns the final training config.

    Returns
    -------
    Dict[str, Any]
        The final training configuration.
    """
    parser = create_base_parser("Train PPO agent on MiniGrid Maze")
    args = parser.parse_args()

    cli_config = vars(args)
    yaml_config_path = cli_config.pop("config", None)
    return load_yaml_config(yaml_config_path)


def get_eval_config() -> Dict[str, Any]:
    """
    Parses args, loads YAML, merges, and returns the final evaluation config.

    Returns
    -------
    Dict[str, Any]
        The final evaluation configuration.
    """
    parser = create_base_parser("Evaluate PPO agent on MiniGrid Maze")
    args = parser.parse_args()

    cli_config = vars(args)
    yaml_config_path = cli_config.pop("config", None)
    return load_yaml_config(yaml_config_path)
