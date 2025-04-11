# -*- coding: utf-8 -*-
"""
Concrete implementation for writing configuration to YAML files.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from .base_writer import ConfigWriter


class YamlWriter(ConfigWriter):
    """Writes configuration to YAML files."""

    def write(self, config_dict: Dict[str, Any], path: Path) -> None:
        """
        Write configuration data to the given YAML path.

        Ensures the parent directory exists before writing. Uses standard YAML
        formatting options (no flow style, don't sort keys).

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary to save.
        path : Path
            Path to save the YAML configuration to.

        Raises
        ------
        OSError
            If there's an error creating directories or writing the file.
        yaml.YAMLError
            If there's an error during YAML serialization.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)
