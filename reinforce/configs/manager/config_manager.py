# -*- coding: utf-8 -*-
"""
Core ConfigManager class responsible for orchestrating config loading and saving.
"""

from pathlib import Path
from typing import Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .reader import YamlReader

# ##: Define a generic type for Pydantic models.
T = TypeVar("T", bound=BaseModel)


class ConfigManager:
    """
    Class responsible for loading and validating configuration files.
    """

    @staticmethod
    def load_and_validate(path: Union[str, Path], model_cls: Type[T]) -> T:
        """
        Load a configuration file and validate it using a Pydantic model.

        Parameters
        ----------
        path : str | Path
            Path to the configuration file.
        model_cls : Type[T]
            The Pydantic model class to validate against.

        Returns
        -------
        T
            Validated configuration object (instance of model_cls).

        Raises
        ------
        ValueError
            If the file format is not supported or validation fails.
        """
        raw_config = YamlReader().read(Path(path))

        try:
            return model_cls(**raw_config)
        except ValidationError as exc:
            raise ValueError(f"Configuration validation failed for {path}: {exc}") from exc
