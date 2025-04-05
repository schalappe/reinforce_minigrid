# -*- coding: utf-8 -*-
"""
Configuration management for the reinforcement learning framework.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, ValidationError

from reinforce.configs.models import ExperimentConfig


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
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "default_configs"

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """
        Load a configuration from a file.

        Parameters
        ----------
        path : str | Path
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
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        file_ext = path_obj.suffix.lower()

        if file_ext in (".yaml", ".yml"):
            with path_obj.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
        elif file_ext == ".json":
            with path_obj.open("r", encoding="utf-8") as file:
                config = json.load(file)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return config

    @staticmethod
    def save_config(config: Union[BaseModel, Dict[str, Any]], path: str) -> None:
        """
        Save a Pydantic configuration model to a file.

        Parameters
        ----------
        config : BaseModel | Dict[str, Any]
            Pydantic configuration model or dictionary to save.
        path : str | Path
            Path to save the configuration to.

        Raises
        ------
        ValueError
            If the file format is not supported.
        """
        path_obj = Path(path)
        file_ext = path_obj.suffix.lower()

        # ##: Create directory if it doesn't exist.
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # ##: Dump Pydantic model to dict before saving.
        config_dict = config.model_dump(mode="json") if isinstance(config, BaseModel) else config

        if file_ext in (".yaml", ".yml"):
            with path_obj.open("w", encoding="utf-8") as file:
                yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)
        elif file_ext == ".json":
            with path_obj.open("w", encoding="utf-8") as file:
                json.dump(config_dict, file, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def load_experiment_config(self, path: str) -> ExperimentConfig:
        """
        Load and validate an experiment configuration using Pydantic.

        Parameters
        ----------
        path : str
            Path to the experiment configuration file.

        Returns
        -------
        ExperimentConfig
            Validated experiment configuration object.

        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        ValueError
            If the file format is not supported or validation fails.
        """
        raw_config = self.load_config(path)

        try:
            # ##: Parse and validate the raw dictionary using the Pydantic model.
            experiment_config = ExperimentConfig(**raw_config)
            return experiment_config
        except ValidationError as exc:
            raise ValueError(f"Configuration validation failed: {exc}") from exc
