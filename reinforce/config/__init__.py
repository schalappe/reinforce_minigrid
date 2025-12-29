"""
Configuration Management for REINFORCE Training.

This package handles the loading and management of configuration settings for
the REINFORCE agent training process.

Modules
-------
config_loader
    Provides functions to load configuration from files.
training_config
    Defines the data structures for training configurations.
"""

from .config_loader import load_config
from .training_config import MainConfig

__all__ = ["load_config", "MainConfig"]
