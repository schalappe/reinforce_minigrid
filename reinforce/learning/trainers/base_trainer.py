# -*- coding: utf-8 -*-
"""
Base interface for all trainers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTrainer(ABC):
    """
    Base interface for all trainers.

    This abstract class defines the interface that all trainer implementations must adhere to.
    It provides a standard API for training agents in environments.
    """

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the agent in the environment.

        Returns
        -------
        Dict[str, Any]
            Dictionary of training metrics.
        """
