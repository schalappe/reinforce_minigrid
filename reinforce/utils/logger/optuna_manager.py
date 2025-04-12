# -*- coding: utf-8 -*-
"""
Optuna Manager for Hyperparameter Search.

This module provides a class to manage Optuna study creation, optimization, and hyperparameter sampling,
separating Optuna logic from the main experiment runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from loguru import logger
from optuna import Study, Trial, create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

from reinforce.utils.logger.logging_setup import setup_logger

setup_logger()


class OptunaManager:
    """
    Manages Optuna study interactions for hyperparameter optimization.

    Handles study creation/loading, running optimization loops, sampling parameters, and pruning logic.
    """

    def __init__(
        self,
        search_name: str,
        results_dir: Union[str, Path] = "outputs/hyperparameter_search",
        direction: str = "maximize",
        pruner: Optional[MedianPruner] = None,
    ):
        """
        Initialize the OptunaManager.

        Parameters
        ----------
        search_name : str
            Name of the Optuna study.
        results_dir : Union[str, Path], optional
            Directory to store Optuna study database. Defaults to "outputs/hyperparameter_search".
        direction : str, optional
            Direction of optimization ('maximize' or 'minimize'). Defaults to "maximize".
        pruner : Optional[MedianPruner], optional
            Optuna pruner instance. Defaults to MedianPruner with default settings.
        """
        self.search_name = search_name
        self.results_dir = Path(results_dir)
        self.direction = direction
        self.pruner = pruner or MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
        self.study: Optional[Study] = None
        self.search_config: Dict[str, Any] = {}

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._create_or_load_study()

    def _create_or_load_study(self):
        """Create or load the Optuna study with storage and pruner."""
        storage_name = f"sqlite:///{self.results_dir}/{self.search_name}.db"
        try:
            self.study = create_study(
                study_name=self.search_name,
                storage=storage_name,
                load_if_exists=True,
                direction=self.direction,
                pruner=self.pruner,
            )
            logger.info(f"Using persistent storage for study: {storage_name}")
        except ImportError as exc:
            logger.error(f"Database backend for Optuna storage not installed: {exc}")
            logger.warning("Using in-memory storage instead.")
            self.study = create_study(study_name=self.search_name, direction=self.direction, pruner=self.pruner)

    def _sample_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters for the current trial based on search config.

        Parameters
        ----------
        trial : Trial
            The Optuna trial object.

        Returns
        -------
        Dict[str, Any]
            A dictionary of sampled hyperparameters.
        """
        params = {}
        if "hyperparameters" not in self.search_config:
            logger.warning("No hyperparameters defined in search config.")
            return params

        for param_path, param_config in self.search_config["hyperparameters"].items():
            param_type = param_config.get("type", "categorical")
            try:
                if param_type == "categorical":
                    params[param_path] = trial.suggest_categorical(param_path, param_config["values"])
                elif param_type == "float":
                    log_scale = param_config.get("log_scale", False)
                    params[param_path] = trial.suggest_float(
                        param_path, param_config["low"], param_config["high"], log=log_scale
                    )
                elif param_type == "int":
                    log_scale = param_config.get("log_scale", False)
                    params[param_path] = trial.suggest_int(
                        param_path, param_config["low"], param_config["high"], log=log_scale
                    )
                else:
                    logger.warning(f"Unsupported parameter type '{param_type}' for {param_path}")
            except KeyError as exc:
                logger.error(f"Missing required key '{exc}' for parameter {param_path} in search config.")
                raise ValueError(f"Invalid config for parameter {param_path}") from exc

        return params

    def run_optimization(
        self,
        objective_func: Callable[[Trial, Dict[str, Any]], float],
        n_trials: int,
        *,
        show_progress_bar: bool = True,
    ):
        """
        Run the Optuna optimization process.

        Parameters
        ----------
        objective_func : Callable[[Trial, Dict[str, Any]], float]
            The objective function to optimize. It should accept the Optuna trial
            and the sampled hyperparameters, and return the objective value.
        n_trials : int
            Number of trials to run.
        show_progress_bar : bool, optional
            Whether to display a progress bar. Defaults to True.

        Raises
        ------
        RuntimeError
            If the study is not initialized.
        Exception
            If the Optuna optimization process fails.
        """
        if not self.study:
            raise RuntimeError("Study not initialized before running optimization.")

        def _wrapped_objective(trial: Trial) -> float:
            """
            Internal wrapper for the objective function.

            Parameters
            ----------
            trial : Trial
                The Optuna trial object.

            Returns
            -------
            float
                The objective value.
            """
            params = self._sample_hyperparameters(trial)
            try:
                value = objective_func(trial, params)
                return value
            except TrialPruned:
                logger.info(f"Trial {trial.number} was pruned.")
                raise

        self.study.optimize(_wrapped_objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    def get_study_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the study results.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the search summary.

        Raises
        ------
        RuntimeError
            If the study is not available.
        """
        if not self.study:
            raise RuntimeError("Study not available for summarization.")

        best_params = {}
        best_value = None
        best_trial_num = None

        if self.study.best_trial:
            best_params = self.study.best_trial.params
            best_value = self.study.best_trial.value
            best_trial_num = self.study.best_trial.number
        else:
            logger.info("No best trial found in Optuna study.")

        all_trials = [
            {"number": trial.number, "value": trial.value, "params": trial.params}
            for trial in self.study.trials
            if trial.state == TrialState.COMPLETE
        ]

        summary = {
            "num_completed_trials": len(all_trials),
            "best_trial_number": best_trial_num,
            "best_value": best_value,
            "best_hyperparameters": best_params,
            "all_completed_trials": all_trials,
            "study_name": self.study.study_name,
            "direction": str(self.study.direction),
        }
        return summary
