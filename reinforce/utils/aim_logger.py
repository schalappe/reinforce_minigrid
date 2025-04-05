# -*- coding: utf-8 -*-
"""
Utility for AIM experiment tracking.

This module provides a wrapper class `AimLogger` for streamlined interaction with the Aim experiment tracking library.
It simplifies initializing runs, logging various data types (parameters, metrics, artifacts, images, text), and
managing the run lifecycle.
"""

# Removed: import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from aim.sdk.objects import Image as AimImage
from aim.sdk.objects import Text as AimText
from aim.sdk.run import Run
from loguru import logger

from reinforce.utils.logging_setup import setup_logger

setup_logger()


class AimLogger:
    """
    A wrapper class for AIM experiment tracking, promoting DRY principles.

    Handles AIM Run initialization, logging of hyperparameters, metrics, artifacts (metadata), images, and text, along
    with managing the run lifecycle using a context manager pattern.

    Examples
    --------
    >>> logger_config = {
    ...     "experiment_name": "my_rl_experiment",
    ...     "run_name": "a2c_run_1",
    ...     "tags": ["a2c", "baseline"],
    ...     "repo_path": ".aim_results"
    ... }
    >>> with AimLogger(**logger_config) as aim_log:
    ...     if aim_log.run: # Check if run initialized successfully
    ...         hparams = {"learning_rate": 0.001, "gamma": 0.99}
    ...         aim_log.log_params(hparams, prefix="agent")
    ...         for step in range(100):
    ...             metrics = {"reward": step * 0.1, "loss": 1.0 / (step + 1)}
    ...             aim_log.log_metrics(metrics, step=step)
    ...         # Log final artifact metadata
    ...         aim_log.log_artifact("model_weights", name="final_model", path="models/final.pt")
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

        self._run: Optional[Run] = None
        self._repo_path: Union[str, Path] = repo_path
        self._experiment_name: str = experiment_name
        self._tags: List[str] = tags or []

        self._initialize_run()

    def __enter__(self) -> "AimLogger":
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the runtime context and close the AIM run."""
        self.close()

    def _initialize_run(self) -> None:
        """Initialize the AIM Run object and set system params."""
        if self._run:
            logger.warning("AIM Run (%s) already initialized. Skipping re-initialization.", self.run_hash)
            return

        try:
            self._run = Run(
                repo=self._repo_path,
                experiment=self._experiment_name,
                system_tracking_interval=None,
            )
            logger.info("AIM Run initialized: Name='%s', Hash='%s'", self._run.name, self.run_hash)

            # ##: Add tags if provided.
            if self._tags:
                current_tags = set(self._run.props.tags)
                new_tags = set(self._tags)
                for tag in new_tags - current_tags:
                    try:
                        self._run.add_tag(tag)
                        logger.debug("Added tag '%s' to run %s", tag, self.run_hash)
                    except Exception as exc:
                        logger.error("Failed to add tag '%s' to run %s: %s", tag, self.run_hash, exc)

        except Exception as exc:
            logger.error(
                "Fatal error initializing AIM Run (Experiment: '%s', Repo: '%s'): %s",
                self._experiment_name,
                self._repo_path,
                exc,
                exc_info=True,
            )
            self._run = None

    def _safe_run_operation(self, operation: Callable[..., Any], description: str, *args: Any, **kwargs: Any) -> bool:
        """
        Safely execute an operation on the AIM Run object.

        Handles checking if the run exists and catches exceptions during the operation.

        Parameters
        ----------
        operation : Callable[..., Any]
            The AIM Run method or operation to execute (e.g., self._run.track, self._run.set).
        description : str
            A description of the operation for logging purposes (e.g., "logging metric 'reward'").
        *args : Any
            Positional arguments for the operation.
        **kwargs : Any
            Keyword arguments for the operation.

        Returns
        -------
        bool
            True if the operation was successful, False otherwise.
        """
        if not self._run:
            logger.warning("AIM Run not initialized. Cannot perform operation: %s.", description)
            return False
        try:
            operation(*args, **kwargs)
            logger.debug("Successfully performed operation: %s (Run: %s)", description, self.run_hash)
            return True
        except Exception as exc:
            logger.error(
                "Error performing operation '%s' on AIM Run %s: %s", description, self.run_hash, exc, exc_info=False
            )
            return False

    @property
    def run(self) -> Optional[Run]:
        """Get the underlying AIM Run object. Returns None if initialization failed."""
        return self._run

    @property
    def run_hash(self) -> Optional[str]:
        """Get the hash of the current AIM Run. Returns None if not initialized."""
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

        Examples
        --------
        >>> AimLogger.log_params({"learning_rate": 0.01, "optimizer": "Adam"})
        >>> AimLogger.log_params({"gamma": 0.99, "lambda": 0.95}, prefix="ppo")
        """
        if prefix:
            processed_params = {f"{prefix}.{k}": v for k, v in params.items()}
        else:
            processed_params = params

        if not processed_params:
            logger.warning("No parameters provided to log_params after processing prefix.")
            return

        op_desc = f"logging parameters with prefix '{prefix}'" if prefix else "logging parameters"
        self._safe_run_operation(
            lambda p: self._run.set("hparams", {**self._run.get("hparams", {}), **p}, strict=False),
            op_desc,
            processed_params,
        )

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

        Examples
        --------
        >>> AimLogger.log_metric("train_loss", 0.5, step=100, context={"subset": "train"})
        >>> AimLogger.log_metric("val_accuracy", 0.95, epoch=5, context={"subset": "validation"})
        """
        op_desc = f"logging metric '{name}' (Step: {step}, Epoch: {epoch})"
        self._safe_run_operation(
            self._run.track, op_desc, value, name=name, step=step, epoch=epoch, context=context or {}
        )

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

        Examples
        --------
        >>> metrics_dict = {"reward": 10.5, "steps_per_episode": 55}
        >>> AimLogger.log_metrics(metrics_dict, step=500, context={"env": "CartPole-v1"})
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

        op_desc = f"logging artifact info for '{name}'"
        self._safe_run_operation(
            lambda key, value: self._run.set(key, value, strict=False), op_desc, f"artifacts/{name}", artifact_info
        )

    def log_image(
        self,
        image_data: Any,
        name: str,
        step: Optional[int] = None,
        *,
        epoch: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        caption: Optional[str] = None,
    ) -> None:
        """
        Log an image.

        The image data should be in a format supported by `aim.Image`
        (e.g., PIL Image, NumPy array, PyTorch tensor).

        Parameters
        ----------
        image_data : Any
            The image data to log.
        name : str
            Name for the image sequence (e.g., 'environment_observations').
        step : Optional[int], optional
            The step number for the image. Defaults to None.
        epoch : Optional[int], optional
            The epoch number for the image. Defaults to None.
        context : Optional[Dict[str, Any]], optional
            Additional context for the image. Defaults to None.
        caption : Optional[str], optional
            Caption for the image. Defaults to None.

        Examples
        --------
        >>> import numpy as np
        >>> random_image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        >>> AimLogger.log_image(random_image, "random_noise", step=50, caption="Random RGB noise")
        """
        try:
            aim_image = AimImage(image_data, caption=caption)
        except Exception as exc:
            logger.error("Failed to create aim.Image object for '%s': %s. Skipping log_image.", name, exc)
            return

        op_desc = f"logging image '{name}' (Step: {step}, Epoch: {epoch})"
        self._safe_run_operation(
            self._run.track, op_desc, aim_image, name=name, step=step, epoch=epoch, context=context or {}
        )

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

        Examples
        --------
        >>> AimLogger.log_text("Episode finished after 150 steps.", "episode_log", step=150)
        """
        try:
            aim_text = AimText(text_data)
        except Exception as exc:
            logger.error("Failed to create aim.Text object for '%s': %s. Skipping log_text.", name, exc)
            return

        op_desc = f"logging text '{name}' (Step: {step}, Epoch: {epoch})"
        self._safe_run_operation(
            self._run.track, op_desc, aim_text, name=name, step=step, epoch=epoch, context=context or {}
        )

    def close(self) -> None:
        """
        Close the AIM Run and release resources.

        This is automatically called when exiting a `with` block.
        """
        if self._run:
            run_hash = self.run_hash
            try:
                self._run.close()
                logger.info("AIM Run closed successfully: %s", run_hash)
            except Exception as exc:
                logger.error("Error closing AIM Run %s: %s", run_hash, exc, exc_info=True)
            finally:
                self._run = None
        else:
            logger.info("No active AIM Run to close (already closed or initialization failed).")
