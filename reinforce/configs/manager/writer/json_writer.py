# -*- coding: utf-8 -*-
"""
Concrete implementation for writing configuration to JSON files.
"""

import json
from pathlib import Path
from typing import Any, Dict

from .base_writer import ConfigWriter


class JsonWriter(ConfigWriter):
    """Writes configuration to JSON files."""

    def write(self, config_dict: Dict[str, Any], path: Path) -> None:
        """
        Write configuration data to the given JSON path.

        Ensures the parent directory exists before writing.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary to save.
        path : Path
            Path to save the JSON configuration to.

        Raises
        ------
        OSError
            If there's an error creating directories or writing the file.
        TypeError
            If the config_dict contains non-serializable types for JSON.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(config_dict, file, indent=2)
