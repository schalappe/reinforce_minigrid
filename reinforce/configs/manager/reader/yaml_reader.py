# -*- coding: utf-8 -*-
"""
Concrete implementation for reading configuration from YAML files.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from .base_reader import ConfigReader


class YamlReader(ConfigReader):
    """Reads configuration from YAML files."""

    def read(self, path: Path) -> Dict[str, Any]:
        """
        Read configuration data from the given YAML path.

        Parameters
        ----------
        path : Path
            Path to the YAML configuration file.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        yaml.YAMLError
            If the file is not valid YAML.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
