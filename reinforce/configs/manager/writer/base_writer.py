# -*- coding: utf-8 -*-
"""
Abstract base class for configuration writers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class ConfigWriter(ABC):
    """Abstract base class for configuration writers."""

    @abstractmethod
    def write(self, config_dict: Dict[str, Any], path: Path) -> None:
        """
        Write configuration data to the given path.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary to save.
        path : Path
            Path to save the configuration to.

        Raises
        ------
        Exception
            If there's an error during writing.
        """
        pass
