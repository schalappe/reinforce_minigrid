# -*- coding: utf-8 -*-
"""
Concrete implementation for reading configuration from JSON files.
"""

import json
from pathlib import Path
from typing import Any, Dict

from .base_reader import ConfigReader


class JsonReader(ConfigReader):
    """Reads configuration from JSON files."""

    def read(self, path: Path) -> Dict[str, Any]:
        """
        Read configuration data from the given JSON path.

        Parameters
        ----------
        path : Path
            Path to the JSON configuration file.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
