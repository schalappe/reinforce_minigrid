# -*- coding: utf-8 -*-
"""
Abstract base class for configuration readers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class ConfigReader(ABC):
    """Abstract base class for configuration readers."""

    @abstractmethod
    def read(self, path: Path) -> Dict[str, Any]:
        """
        Read configuration data from the given path.

        Parameters
        ----------
        path : Path
            Path to the configuration file.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        Exception
            If there's an error during reading or parsing.
        """
        pass
