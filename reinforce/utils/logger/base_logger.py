# -*- coding: utf-8 -*-
"""
Base interface for logging components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union


class BaseLogger(ABC):
    """
    Abstract base class for logging implementations.

    Defines the standard interface for logging parameters, metrics, artifacts, images, and text during experiments.
    """

    @abstractmethod
    def log_params(self, params: Dict[str, Any], prefix: Optional[str] = None) -> None:
        """
        Log hyperparameters or configuration parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters to log.
        prefix : Optional[str], optional
            Optional prefix for parameter keys. Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def log_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        *,
        epoch: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a single metric value.

        Parameters
        ----------
        name : str
            Name of the metric.
        value : Any
            Value of the metric.
        step : Optional[int], optional
            Step number. Defaults to None.
        epoch : Optional[int], optional
            Epoch number. Defaults to None.
        context : Optional[Dict[str, Any]], optional
            Additional context. Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log multiple metrics from a dictionary.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Dictionary of metrics.
        step : Optional[int], optional
            Step number. Defaults to None.
        epoch : Optional[int], optional
            Epoch number. Defaults to None.
        context : Optional[Dict[str, Any]], optional
            Additional context. Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def log_artifact(
        self,
        artifact_data: Any,
        name: str,
        path: Optional[Union[str, Path]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metadata about an artifact.

        Parameters
        ----------
        artifact_data : Any
            The artifact object itself (used for type info, not stored).
        name : str
            Unique name for the artifact.
        path : Optional[Union[str, Path]], optional
            Filesystem path to the artifact. Defaults to None.
        meta : Optional[Dict[str, Any]], optional
            Additional metadata. Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def log_text(
        self,
        text_data: str,
        name: str,
        step: Optional[int] = None,
        *,
        epoch: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log text data.

        Parameters
        ----------
        text_data : str
            Text string to log.
        name : str
            Name for the text sequence.
        step : Optional[int], optional
            Step number. Defaults to None.
        epoch : Optional[int], optional
            Epoch number. Defaults to None.
        context : Optional[Dict[str, Any]], optional
            Additional context. Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Close the logger and release resources.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def run_hash(self) -> Optional[str]:
        """
        Get the unique identifier (hash) of the current run, if applicable.

        Returns
        -------
        str | None
            The run hash, or None if not applicable.
        """
        raise NotImplementedError

    def __enter__(self) -> BaseLogger:
        """
        Enter the runtime context.

        Returns
        -------
        BaseLogger
            The logger instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the runtime context and close the logger."""
        self.close()
