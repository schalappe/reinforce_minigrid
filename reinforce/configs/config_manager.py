# -*- coding: utf-8 -*-
"""
Configuration management for the reinforcement learning framework.
"""

import json
import os
from typing import Any, Dict, Optional

import yaml

from reinforce.configs.schemas.agent_schema import validate_agent_config
from reinforce.configs.schemas.trainer_schema import validate_trainer_config

# ##: TODO: Use pathlib for path management.


class ConfigManager:
    """
    Configuration manager for the reinforcement learning framework.

    This class provides functionality for loading, validating, and managing configurations
    for agents, environments, trainers, and experiments.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the configuration manager.

        Parameters
        ----------
        config_dir : str, optional
            Directory containing configuration files
        """
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), "default_configs")

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """
        Load a configuration from a file.

        Parameters
        ----------
        path : str
            Path to the configuration file.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        ValueError
            If the file format is not supported.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        file_ext = os.path.splitext(path)[1].lower()

        if file_ext in (".yaml", ".yml"):
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        elif file_ext == ".json":
            with open(path, "r") as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return config

    @staticmethod
    def save_config(config: Dict[str, Any], path: str) -> None:
        """
        Save a configuration to a file.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.
        path : str
            Path to save the configuration to.

        Raises
        ------
        ValueError
            If the file format is not supported.
        """
        file_ext = os.path.splitext(path)[1].lower()

        # ##: Create directory if it doesn't exist.
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if file_ext in (".yaml", ".yml"):
            with open(path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        elif file_ext == ".json":
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @staticmethod
    def validate_agent_config(config: Dict[str, Any]) -> None:
        """
        Validate an agent configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Agent configuration to validate.

        Raises
        ------
        ValidationError
            If the configuration is invalid.
        """
        validate_agent_config(config)

    @staticmethod
    def validate_trainer_config(config: Dict[str, Any]) -> None:
        """
        Validate a trainer configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Trainer configuration to validate.

        Raises
        ------
        ValidationError
            If the configuration is invalid.
        """
        validate_trainer_config(config)

    def get_default_config(self, config_type: str, name: str) -> Dict[str, Any]:
        """
        Get a default configuration.

        Parameters
        ----------
        config_type : str
            Type of configuration (agent, environment, trainer, etc.).
        name : str
            Name of the configuration.

        Returns
        -------
        Dict[str, Any]
            Default configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the default configuration is not found.
        """
        config_path = os.path.join(self.config_dir, config_type, f"{name}.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Default configuration not found: {config_path}")

        return self.load_config(config_path)

    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations.

        Parameters
        ----------
        base_config : Dict[str, Any]
            Base configuration.
        override_config : Dict[str, Any]
            Configuration to override base configuration.

        Returns
        -------
        Dict[str, Any]
            Merged configuration.
        """
        merged_config = base_config.copy()

        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                merged_config[key] = self.merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value

        return merged_config

    def load_experiment_config(self, path: str) -> Dict[str, Any]:
        """
        Load an experiment configuration.

        An experiment configuration includes configurations for agent, environment, trainer, and other components.

        Parameters
        ----------
        path : str
            Path to the experiment configuration file.

        Returns
        -------
        Dict[str, Any]
            Experiment configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        ValueError
            If the file format is not supported.
        """
        config = self.load_config(path)
        # ##: Validate and process agent configuration.
        if "agent" in config:
            if isinstance(config["agent"], str):
                agent_config = self.get_default_config("agent", config["agent"])
            elif isinstance(config["agent"], dict):
                agent_config = config["agent"]
            else:
                raise ValueError(f"Invalid agent configuration type: {type(config['agent'])}")

            self.validate_agent_config(agent_config)
            config["agent"] = agent_config

        # ##: Validate and process trainer configuration.
        if "trainer" in config:
            if isinstance(config["trainer"], str):
                trainer_config = self.get_default_config("trainer", config["trainer"])
            elif isinstance(config["trainer"], dict):
                trainer_config = config["trainer"]
            else:
                raise ValueError(f"Invalid trainer configuration type: {type(config['trainer'])}")

            self.validate_trainer_config(trainer_config)
            config["trainer"] = trainer_config

        return config
