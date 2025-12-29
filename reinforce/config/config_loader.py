"""
Handles loading and merging of configuration from YAML files and command-line arguments.
"""

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from .training_config import MainConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load configuration data from a YAML file.

    Parameters
    ----------
    path : Path
        The path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the loaded configuration data.
        Returns an empty dictionary if the YAML file is empty.

    Raises
    ------
    SystemExit
        If the specified path does not point to a file or if there is an error parsing the YAML content.
    """
    if not path.is_file():
        logger.error(f"Configuration file not found: {path}")
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {path}: {e}")
        sys.exit(1)


def _map_yaml_to_dataclass_field(dataclass_type: type[Any], yaml_key: str) -> str:
    """
    Map a YAML key to its corresponding dataclass field name.

    Checks if a field in the dataclass has a 'yaml_name' metadata attribute matching the provided `yaml_key`.
    If a match is found, the field's actual name is returned. Otherwise, the original `yaml_key` is returned.

    Parameters
    ----------
    dataclass_type : Type[Any]
        The dataclass type to inspect for field metadata.
    yaml_key : str
        The key name as it appears in the YAML configuration.

    Returns
    -------
    str
        The corresponding dataclass field name, or the original `yaml_key` if no mapping is found in the metadata.
    """
    for field in dataclasses.fields(dataclass_type):
        if field.metadata.get("yaml_name") == yaml_key:
            return field.name
    return yaml_key


def _dict_to_dataclass[T](dataclass_type: type[T], data: dict[str, Any]) -> T:
    """
    Recursively convert a dictionary to a nested dataclass structure.

    This function takes a dictionary and attempts to map its keys and values to the fields of the
    specified dataclass type. It handles nested dataclasses, optional types, and basic type conversions.
    It logs warnings for unknown keys or type mismatches.

    Parameters
    ----------
    dataclass_type : Type[T]
        The target dataclass type (or nested dataclass type) to instantiate.
    data : Dict[str, Any]
        The dictionary containing data to populate the dataclass instance.
        Keys may be mapped using `_map_yaml_to_dataclass_field`.

    Returns
    -------
    T
        An instance of `dataclass_type` populated with data from the input dictionary.
        Fields not present in the dictionary will use their default values if defined.

    Raises
    ------
    SystemExit
        If instantiation fails due to missing required fields (those without defaults)
        or other `TypeError` during instantiation.
    """
    # ##&: Cast required because dataclasses.fields expects DataclassInstance, but type[T] is compatible at runtime.
    field_types = {f.name: f.type for f in dataclasses.fields(dataclass_type)}  # type: ignore[arg-type]
    mapped_data = {}

    for yaml_key, value in data.items():
        field_name = _map_yaml_to_dataclass_field(dataclass_type, yaml_key)
        if field_name not in field_types:
            logger.warning(f'Ignoring unknown config key "{yaml_key}" in section "{dataclass_type.__name__}"')
            continue

        field_type = field_types[field_name]

        # ##: Handle nested dataclasses.
        if dataclasses.is_dataclass(field_type):
            if isinstance(value, dict):
                mapped_data[field_name] = _dict_to_dataclass(field_type, value)  # type: ignore[arg-type]
            else:
                field_type_name = getattr(field_type, "__name__", str(field_type))
                logger.warning(
                    f'Expected dict for nested config "{field_name}" ({field_type_name}), '
                    f"got {type(value)}. Using defaults."
                )
                mapped_data[field_name] = field_type()  # type: ignore[operator]
        # ##: Handle Optional types.
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is Optional:
            inner_type = field_type.__args__[0]  # type: ignore[union-attr]
            if value is None or isinstance(value, inner_type):  # type: ignore[arg-type]
                mapped_data[field_name] = value
            else:
                logger.warning(
                    f'Incorrect type for optional field "{field_name}". Expected {inner_type}, got {type(value)}.'
                )
                mapped_data[field_name] = None
        else:
            # ##: Basic type conversion/assignment.
            # ##&: field_type may be a string (forward ref) or actual type at runtime.
            if not isinstance(value, field_type):  # type: ignore[arg-type]
                try:
                    mapped_data[field_name] = field_type(value)  # type: ignore[operator]
                except (TypeError, ValueError):
                    logger.warning(
                        f'Type mismatch for "{field_name}". '
                        f"Expected {field_type}, got {type(value)}. "
                        f"Using value as is."
                    )
                    mapped_data[field_name] = value
            else:
                mapped_data[field_name] = value

    # ##: Instantiate the dataclass with potentially missing fields relying on defaults.
    valid_keys = {f.name for f in dataclasses.fields(dataclass_type)}  # type: ignore[arg-type]
    filtered_data = {k: v for k, v in mapped_data.items() if k in valid_keys}
    try:
        return dataclass_type(**filtered_data)
    except TypeError as e:
        logger.error(f"Error instantiating {dataclass_type.__name__}: {e}. Check config structure and types.")
        missing_fields = []
        for f in dataclasses.fields(dataclass_type):  # type: ignore[arg-type]
            if (
                f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
                and f.name not in filtered_data
            ):
                missing_fields.append(f.name)
        if missing_fields:
            separator = ", "
            logger.error(f"Missing required fields without defaults: {separator.join(missing_fields)}")
        sys.exit(1)


def load_config(
    config_path: str | None = None,
    args: argparse.Namespace | None = None,
) -> MainConfig:
    """
    Load configuration from YAML and CLI overrides.

    Loads configuration settings starting from the defaults defined in the `MainConfig` dataclass.
    It then merges settings from a specified YAML file (if provided) and finally applies any
    overrides specified through command-line arguments. The resulting configuration is validated
    before being returned.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file. If None, only default settings and command-line overrides are used.
        Default is None.
    args : argparse.Namespace, optional
        Parsed command-line arguments. If provided, these arguments can override settings from the YAML
        file or defaults. Default is None.

    Returns
    -------
    MainConfig
        A populated and validated instance of the `MainConfig` dataclass, representing the final configuration settings.

    Raises
    ------
    SystemExit
        If the configuration file is not found (via `_load_yaml`), cannot be parsed (via `_load_yaml`), or if the final
        configuration fails validation or instantiation (via `_dict_to_dataclass` or `__post_init__`).
    """
    # ##: 1. Start with default config from dataclass definitions.
    config_dict = dataclasses.asdict(MainConfig())

    # ##: 2. Load from YAML file if provided.
    yaml_config = {}
    if config_path:
        logger.info(f"Loading configuration from: {config_path}")
        yaml_config = _load_yaml(Path(config_path))
    else:
        logger.info("No config file specified, using default settings.")

    # ##: 3. Merge YAML config into the default config dictionary (deep merge might be needed for nested dicts).
    for key, value in yaml_config.items():
        if key in config_dict and isinstance(value, dict) and isinstance(config_dict[key], dict):
            config_dict[key].update(value)
        elif key in config_dict:
            config_dict[key] = value
        else:
            logger.warning(f"Ignoring unknown top-level key {key} in YAML config.")

    # ##: 4. Apply command-line overrides if args are provided.
    if args:
        logger.info("Applying command-line overrides...")
        arg_map = {
            "total_timesteps": ("training", "total_timesteps"),
            "steps_per_update": ("training", "steps_per_update"),
            "num_envs": ("training", "num_envs"),
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
                    yaml_name = "lambda_gae"
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
