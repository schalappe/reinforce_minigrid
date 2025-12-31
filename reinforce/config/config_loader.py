"""Load configuration from YAML files with CLI overrides."""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .training_config import MainConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file, returning empty dict if file is empty."""
    if not path.is_file():
        logger.error(f"Configuration file not found: {path}")
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {path}: {e}")
        sys.exit(1)


def _apply_cli_overrides(config_dict: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to the config dictionary in-place."""
    # ##>: Algorithm selection.
    if getattr(args, "algorithm", None) is not None:
        config_dict["algorithm"] = args.algorithm
        logger.debug(f"CLI override: algorithm = {args.algorithm}")

    # ##>: Maps CLI arg name -> (section, config_key).
    arg_mapping = {
        "total_timesteps": ("training", "total_timesteps"),
        "steps_per_update": ("training", "steps_per_update"),
        "num_envs": ("training", "num_envs"),
        "lr": ("ppo", "learning_rate"),
        "gamma": ("ppo", "gamma"),
        "lam": ("ppo", "lambda"),
        "clip": ("ppo", "clip_param"),
        "entropy": ("ppo", "entropy_coef"),
        "vf_coef": ("ppo", "vf_coef"),
        "epochs": ("ppo", "epochs"),
        "batch_size": ("ppo", "batch_size"),
        "max_grad_norm": ("ppo", "max_grad_norm"),
        "seed": ("environment", "seed"),
        "log_interval": ("logging", "log_interval"),
        "save_interval": ("logging", "save_interval"),
        "save_path": ("logging", "save_path"),
        "load_path": ("logging", "load_path"),
    }

    # ##>: DQN-specific argument mappings.
    dqn_arg_mapping = {
        "dqn_lr": ("dqn", "learning_rate"),
        "dqn_gamma": ("dqn", "gamma"),
        "n_step": ("dqn", "n_step"),
        "num_atoms": ("dqn", "num_atoms"),
        "buffer_size": ("dqn", "buffer_size"),
        "dqn_batch_size": ("dqn", "batch_size"),
        "target_update_freq": ("dqn", "target_update_freq"),
        "learning_starts": ("dqn", "learning_starts"),
        "train_freq": ("dqn", "train_freq"),
    }
    arg_mapping.update(dqn_arg_mapping)

    for arg_name, (section, key) in arg_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config_dict.setdefault(section, {})[key] = value
            logger.debug(f"CLI override: {section}.{key} = {value}")

    # ##>: Handle boolean flags.
    if getattr(args, "no_lr_annealing", False):
        config_dict.setdefault("ppo", {})["use_lr_annealing"] = False
        logger.debug("CLI override: ppo.use_lr_annealing = False")
    if getattr(args, "use_value_clipping", False):
        config_dict.setdefault("ppo", {})["use_value_clipping"] = True
        logger.debug("CLI override: ppo.use_value_clipping = True")

    # ##>: DQN component toggles.
    dqn_toggles = ["no_noisy", "no_dueling", "no_double", "no_per", "no_multistep", "no_categorical"]
    for toggle in dqn_toggles:
        if getattr(args, toggle, False):
            key = toggle.replace("no_", "use_")
            config_dict.setdefault("dqn", {})[key] = False
            logger.debug(f"CLI override: dqn.{key} = False")


def load_config(
    config_path: str | None = None,
    args: argparse.Namespace | None = None,
) -> MainConfig:
    """
    Load configuration from YAML file with optional CLI overrides.

    Parameters
    ----------
    config_path : str | None
        Path to YAML config file. If None, uses defaults.
    args : argparse.Namespace | None
        CLI arguments to override config values.

    Returns
    -------
    MainConfig
        Validated configuration instance.
    """
    config_dict: dict[str, Any] = {}

    if config_path:
        logger.info(f"Loading configuration from: {config_path}")
        config_dict = _load_yaml(Path(config_path))
    else:
        logger.info("No config file specified, using defaults.")

    if args:
        _apply_cli_overrides(config_dict, args)

    try:
        config = MainConfig.model_validate(config_dict)
        logger.success("Configuration loaded and validated successfully.")
        return config
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
