# -*- coding: utf-8 -*-
"""
Configuration management for the reinforcement learning framework.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Union

import yaml
from pydantic import BaseModel, ValidationError

from reinforce.configs.models import ExperimentConfig


class ConfigManager:
    """
    Configuration manager for the reinforcement learning framework.

    This class provides functionality for loading, validating, and managing configurations
    for agents, environments, trainers, and experiments.
    """

    @staticmethod
    def _get_reader(file_ext: str) -> Callable:
        """
        Get the appropriate function to read a config file based on extension.

        Parameters
        ----------
        file_ext : str
            The file extension of the config file.

        Returns
        -------
        Callable
            The function to read the config file.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        if file_ext in (".yaml", ".yml"):
            return yaml.safe_load

        if file_ext == ".json":
            return json.load

        raise ValueError(f"Unsupported file format for reading: {file_ext}")

    @staticmethod
    def _get_writer(file_ext: str) -> Callable:
        """
        Get the appropriate function to write a config file based on extension.

        Parameters
        ----------
        file_ext : str
            The file extension of the config file.

        Returns
        -------
        Callable
            The function to write the config file.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        if file_ext in (".yaml", ".yml"):
            return lambda data, file: yaml.dump(data, file, default_flow_style=False, sort_keys=False)

        if file_ext == ".json":
            return lambda data, file: json.dump(data, file, indent=2)

        raise ValueError(f"Unsupported file format for writing: {file_ext}")

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
        try:
            reader = ConfigManager._get_reader(file_ext)
            with path_obj.open("r", encoding="utf-8") as file:
                config = reader(file)
            return config
        except ValueError as exc:
            raise ValueError(f"Unsupported file format: {file_ext}") from exc

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

        try:
            writer = ConfigManager._get_writer(file_ext)
            with path_obj.open("w", encoding="utf-8") as file:
                writer(config_dict, file)
        except ValueError as exc:
            raise ValueError(f"Unsupported file format: {file_ext}") from exc

    @classmethod
    def load_experiment_config(cls, path: str) -> ExperimentConfig:
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
        raw_config = cls.load_config(path)

        try:
            # ##: Parse and validate the raw dictionary using the Pydantic model.
            return ExperimentConfig(**raw_config)
        except ValidationError as exc:
            raise ValueError(f"Configuration validation failed: {exc}") from exc
