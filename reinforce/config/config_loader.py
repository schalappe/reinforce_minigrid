# -*- coding: utf-8 -*-
"""
Handles loading and merging of configuration from YAML files and command-line arguments.
"""

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml
from loguru import logger

from .training_config import MainConfig

T = TypeVar("T")


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Loads a YAML file."""
    if not path.is_file():
        logger.error(f"Configuration file not found: {path}")
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {path}: {e}")
        sys.exit(1)


def _map_yaml_to_dataclass_field(dataclass_type: Type[Any], yaml_key: str) -> str:
    """Maps a YAML key to a dataclass field name using metadata if necessary."""
    for field in dataclasses.fields(dataclass_type):
        if field.metadata.get("yaml_name") == yaml_key:
            return field.name
    return yaml_key


def _dict_to_dataclass(dataclass_type: Type[T], data: Dict[str, Any]) -> T:
    """Recursively converts a dictionary to a nested dataclass structure."""
    field_types = {f.name: f.type for f in dataclasses.fields(dataclass_type)}
    mapped_data = {}

    for yaml_key, value in data.items():
        field_name = _map_yaml_to_dataclass_field(dataclass_type, yaml_key)
        if field_name not in field_types:
            logger.warning(f"Ignoring unknown config key '{yaml_key}' in section '{dataclass_type.__name__}'")
            continue

        field_type = field_types[field_name]

        # ##: Handle nested dataclasses.
        if dataclasses.is_dataclass(field_type):
            if isinstance(value, dict):
                mapped_data[field_name] = _dict_to_dataclass(field_type, value)
            else:
                logger.warning(
                    f"Expected dict for nested config '{field_name}' ({field_type.__name__}), "
                    f"got {type(value)}. Using defaults."
                )
                mapped_data[field_name] = field_type()
        # ##: Handle Optional types.
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is Optional:
            inner_type = field_type.__args__[0]
            if value is None or isinstance(value, inner_type):
                mapped_data[field_name] = value
            else:
                logger.warning(
                    f"Incorrect type for optional field '{field_name}'. Expected {inner_type}, got {type(value)}."
                )
                mapped_data[field_name] = None
        else:
            # ##: Basic type conversion/assignment.
            if not isinstance(value, field_type):
                try:
                    mapped_data[field_name] = field_type(value)
                except (TypeError, ValueError):
                    logger.warning(
                        f"Type mismatch for '{field_name}'."
                        f"Expected {field_type}, got {type(value)}"
                        "Using value as is."
                    )
                    mapped_data[field_name] = value
            else:
                mapped_data[field_name] = value

    # ##: Instantiate the dataclass with potentially missing fields relying on defaults.
    try:
        valid_keys = {f.name for f in dataclasses.fields(dataclass_type)}
        filtered_data = {k: v for k, v in mapped_data.items() if k in valid_keys}
        return dataclass_type(**filtered_data)
    except TypeError as e:
        logger.error(f"Error instantiating {dataclass_type.__name__}: {e}. Check config structure and types.")
        missing_fields = []
        for f in dataclasses.fields(dataclass_type):
            if (
                f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
                and f.name not in filtered_data
            ):
                missing_fields.append(f.name)
        if missing_fields:
            logger.error(f"Missing required fields without defaults: {', '.join(missing_fields)}")
        sys.exit(1)


def load_config(
    config_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> MainConfig:
    """
    Loads configuration from a YAML file, applies command-line overrides, and returns a validated MainConfig object.

    Args:
        config_path: Path to the YAML configuration file.
        args: Parsed command-line arguments (optional).

    Returns:
        A populated and validated MainConfig object.
    """
    # 1. Start with default config from dataclass definitions
    config_dict = dataclasses.asdict(MainConfig())

    # 2. Load from YAML file if provided
    yaml_config = {}
    if config_path:
        logger.info(f"Loading configuration from: {config_path}")
        yaml_config = _load_yaml(Path(config_path))
    else:
        logger.info("No config file specified, using default settings.")

    # 3. Merge YAML config into the default config dictionary (deep merge might be needed for nested dicts)
    # Basic merge for top-level keys (environment, training, ppo, logging)
    for key, value in yaml_config.items():
        if key in config_dict and isinstance(value, dict) and isinstance(config_dict[key], dict):
            # Merge nested dictionaries
            config_dict[key].update(value)
        elif key in config_dict:
            # Overwrite top-level non-dict values (shouldn't happen with current structure)
            config_dict[key] = value
        else:
            logger.warning(f"Ignoring unknown top-level key '{key}' in YAML config.")

    # 4. Apply command-line overrides if args are provided
    if args:
        logger.info("Applying command-line overrides...")
        # Map argparse dest names to config structure
        # Example: args.lr -> config_dict['ppo']['learning_rate']
        arg_map = {
            "total_timesteps": ("training", "total_timesteps"),
            "steps_per_update": ("training", "steps_per_update"),
            "lr": ("ppo", "learning_rate"),
            "gamma": ("ppo", "gamma"),
            "lam": ("ppo", "lambda_gae"),
            "clip": ("ppo", "clip_param"),
            "entropy": ("ppo", "entropy_coef"),
            "vf_coef": ("ppo", "value_coef"),
            "epochs": ("ppo", "epochs"),
            "batch_size": ("ppo", "batch_size"),
            "seed": ("environment", "seed"),
            "log_interval": ("logging", "log_interval"),
            "save_interval": ("logging", "save_interval"),
            "save_path": ("logging", "save_path"),
            "load_path": ("logging", "load_path"),
        }
        for arg_name, (section, config_key) in arg_map.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                override_value = getattr(args, arg_name)
                if config_key == "lambda_gae":
                    for field in dataclasses.fields(MainConfig.ppo.__class__):
                        if field.name == "lambda_gae":
                            yaml_name = field.metadata.get("yaml_name", "lambda_gae")
                            break
                    if section in config_dict and isinstance(config_dict[section], dict):
                        config_dict[section][yaml_name] = override_value
                        logger.debug(f"Overriding {section}.{yaml_name} with CLI value: {override_value}")

                elif section in config_dict and isinstance(config_dict[section], dict):
                    target_key = config_key
                    dataclass_section = getattr(MainConfig, section, None)
                    if dataclass_section and hasattr(dataclass_section, "__dataclass_fields__"):
                        field = dataclass_section.__dataclass_fields__.get(config_key)
                        if field and field.metadata.get("yaml_name"):
                            target_key = field.metadata["yaml_name"]

                    config_dict[section][target_key] = override_value
                    logger.debug(f"Overriding {section}.{target_key} with CLI value: {override_value}")

    # ##: 5. Convert the final merged dictionary to the MainConfig dataclass structure.
    try:
        final_config = _dict_to_dataclass(MainConfig, config_dict)
        final_config.__post_init__()
        logger.success("Configuration loaded and validated successfully.")
        return final_config
    except (ValueError, TypeError) as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
