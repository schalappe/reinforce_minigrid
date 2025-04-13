# -*- coding: utf-8 -*-
"""
Utility for AIM experiment tracking.

This module provides a wrapper class `AimLogger` for streamlined interaction with the Aim experiment tracking library.
It simplifies initializing runs, logging various data types, and managing the run lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aim.sdk.objects import Text as AimText
from aim.sdk.run import Run
from loguru import logger

from reinforce.utils.management.logging_setup import setup_logger

setup_logger()


class AimTracker:
    """
    An implementation of BaseLogger using AIM experiment tracking.

    Handles AIM Run initialization, logging of hyperparameters, metrics, artifacts (metadata), images, and text, along
    with managing the run lifecycle using a context manager pattern.
    """

    def __init__(self, experiment_name: str, tags: Optional[List[str]] = None, repo_path: Union[str, Path] = ".aim"):
        """
        Initialize the AimLogger and start an AIM Run.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment in AIM. Must not be empty.
        tags : Optional[List[str]], optional
            Tags to associate with the run. Defaults to an empty list.
        repo_path : Union[str, Path], optional
            Path to the AIM repository. Defaults to ".aim".
        """
        if not experiment_name:
            raise ValueError("experiment_name cannot be empty.")

        self._repo_path: Union[str, Path] = repo_path
        self._experiment_name: str = experiment_name
        self._tags: List[str] = tags or []

        self._run = Run(repo=self._repo_path, experiment=self._experiment_name)
        self._add_tags(self._tags)

        logger.info(f"AIM Run initialized: Name='{self._run.name}', Hash='{self.run_hash}'")

    def __enter__(self) -> AimTracker:
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the runtime context and close the AIM run."""
        self.close()

    def _add_tags(self, tags: List[str]):
        """
        Add tags to the AIM run.

        Parameters
        ----------
        tags : List[str]
            List of tags to add to the run.
        """
        new_tags = set(tags) - set(self._run.props.tags)
        for tag in new_tags:
            self._run.add_tag(tag)
            logger.debug(f"Added tag '{tag}' to run {self.run_hash}")

    @property
    def run(self) -> Optional[Run]:
        """
        Get the underlying AIM Run object. Returns None if initialization failed.

        Returns
        -------
        Run | None
            The AIM Run object or None if not initialized.
        """
        return self._run

    @property
    def run_hash(self) -> Optional[str]:
        """
        Get the hash of the current AIM Run. Returns None if not initialized.

        Returns
        -------
        str | None
            The hash of the AIM Run or None if not initialized.
        """
        return self._run.hash if self._run else None

    def log_params(self, params: Dict[str, Any], prefix: Optional[str] = None) -> None:
        """
        Log hyperparameters or configuration parameters to the 'hparams' dictionary.

        Parameters are merged with existing hyperparameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters to log.
        prefix : Optional[str], optional
            If provided, keys in the params dictionary will be prefixed with f"{prefix}.".
            Defaults to None.
        """
        processed_params = {f"{prefix}.{k}": v for k, v in params.items()} if prefix else params
        self._run.set("hparams", {**self._run.get("hparams", {}), **processed_params}, strict=False)

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
        Log a single metric value at a specific step or epoch.

        Parameters
        ----------
        name : str
            Name of the metric (e.g., 'loss', 'accuracy').
        value : Any
            The value of the metric. Must be serializable by Aim.
        step : Optional[int], optional
            The step number for the metric. Defaults to None.
        epoch : Optional[int], optional
            The epoch number for the metric. Defaults to None.
        context : Optional[Dict[str, Any]], optional
            Additional context for the metric (e.g., {'subset': 'train'}). Defaults to None.
        """
        self._run.track(value, name=name, step=step, epoch=epoch, context=context or {})

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log multiple metrics from a dictionary at a specific step or epoch.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Dictionary where keys are metric names and values are metric values.
        step : Optional[int], optional
            The step number for all metrics in the dictionary. Defaults to None.
        epoch : Optional[int], optional
            The epoch number for all metrics in the dictionary. Defaults to None.
        context : Optional[Dict[str, Any]], optional
            Additional context for all metrics. Defaults to None.
        """
        base_context = context or {}
        for name, value in metrics.items():
            self.log_metric(name, value, step=step, epoch=epoch, context=base_context)

    def log_artifact(
        self,
        artifact_data: Any,
        name: str,
        path: Optional[Union[str, Path]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metadata about an artifact (e.g., model checkpoint, dataset).

        Parameters
        ----------
        artifact_data : Any
            The artifact object itself (used to determine its type). Not stored directly.
        name : str
            A unique name for the artifact within the run.
        path : Optional[Union[str, Path]], optional
            Filesystem path to the artifact file or directory. Defaults to None.
        meta : Optional[Dict[str, Any]], optional
            Additional metadata about the artifact. Defaults to None.

        Notes
        -----
        Note: Aim currently doesn't have first-class artifact tracking like MLflow. This method logs
        artifact information (type, path, metadata) into a dictionary structure within the run's
        parameters under `artifacts/{name}`.
        """
        artifact_info: Dict[str, Any] = {"name": name, "type": type(artifact_data).__name__}

        if path:
            artifact_info["path"] = str(path)

        if meta:
            artifact_info["metadata"] = meta

        self._run.set(f"artifacts/{name}", artifact_info, strict=False)

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
            The text string to log.
        name : str
            Name for the text sequence (e.g., 'episode_summary').
        step : Optional[int], optional
            The step number for the text. Defaults to None.
        epoch : Optional[int], optional
            The epoch number for the text. Defaults to None.
        context : Optional[Dict[str, Any]], optional
            Additional context for the text. Defaults to None.
        """
        self._run.track(AimText(text_data), name=name, step=step, epoch=epoch, context=context or {})

    def close(self) -> None:
        """
        Close the AIM Run and release resources.

        This is automatically called when exiting a `with` block.
        """
        if self._run:
            run_hash = self.run_hash
            self._run.close()
            self._run = None
            logger.info(f"AIM Run closed successfully: {run_hash}")
        else:
            logger.info("No active AIM Run to close (already closed or initialization failed).")
