# -*- coding: utf-8 -*-
"""
Hyperparameter search for reinforcement learning experiments using Optuna.

This module provides a class for running hyperparameter optimization studies using the Optuna framework.
It supports parallel execution, pruning of underperforming trials, and visualization of results.
"""

import json
from pathlib import Path
from traceback import format_exc
from typing import Any, Dict, Optional, Union

from loguru import logger
from optuna import Trial, TrialPruned

from reinforce.configs.manager.reader import YamlReader
from reinforce.configs.models import ExperimentConfig
from reinforce.experiments.experiment_runner import ExperimentRunner
from reinforce.utils.logger import AimTracker, OptunaManager, setup_logger

setup_logger()


class HyperparameterSearch:
    """
    Search over hyperparameters for reinforcement learning experiments using Optuna.

    This class implements an efficient hyperparameter optimization using Optuna framework with
    support for parallel execution, pruning, and visualization.
    """

    experiment_runner = ExperimentRunner()
    results_dir = Path("outputs/hyperparameter_search")
    search_config: Optional[Dict[str, Any]] = None
    base_config: Optional[Dict[str, Any]] = None
    search_name: Optional[str] = None
    optuna_manager: Optional[OptunaManager] = None

    def _setup_search_logging(self, search_name: str, n_trials: int) -> AimTracker:
        """
        Initialize AIM logger for the search and log setup parameters.

        Sets up an AimLogger instance to track the hyperparameter search process, logging search configuration
        details and setup parameters. The logger is configured with an experiment name derived from the search
        configuration file's stem.

        Parameters
        ----------
        search_name : str
            Name of the hyperparameter search.
        n_trials : int
            Number of trials to run in the hyperparameter search.

        Returns
        -------
        AimTracker
            An AimLogger instance if the initialization was successful.

        Raises
        ------
        Exception
            If there are issues with initializing the AimLogger.
        """
        search_aim_logger = AimTracker(
            experiment_name=f"hyperparameter_{search_name}", tags=["hyperparameter-search", "summary", search_name]
        )
        self.search_name = search_name
        search_aim_logger.log_params({"n_trials": n_trials}, prefix="search_setup")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        return search_aim_logger

    def _load_configs(self, search_config_path: Union[str, Path], search_aim_logger: AimTracker):
        """
        Loads the hyperparameter search configuration.

        Parameters
        ----------
        search_config_path : str | Path
            Path to the hyperparameter search configuration file.
        search_aim_logger : AimTracker
            AimLogger instance for logging.

        Raises
        ------
        FileNotFoundError
            If the search or base configuration file is not found.
        """
        try:
            self.search_config = YamlReader().read(Path(search_config_path))
            self.base_config = self.search_config.get("base_config", {})

            if self.search_config:
                search_aim_logger.log_params(self.search_config.get("hyperparameters", {}), prefix="search_space")
        except FileNotFoundError as exc:
            logger.error(f"Search config file not found: {exc}")
            raise

    @classmethod
    def _summarize_and_log_results(cls, summary: Dict[str, Any], search_aim_logger: AimTracker) -> None:
        """
        Log the search summary and print results.

        Parameters
        ----------
        summary : Dict[str, Any]
            The summary dictionary obtained from OptunaManager.
        search_aim_logger : AimTracker
            AimLogger instance for logging.
        """
        best_value = summary.get("best_value")

        # ##: Log summary to AIM.
        search_aim_logger.log_params(summary, prefix="search_summary")
        if best_value is not None:
            search_aim_logger.log_metric("best_mean_reward", best_value)

        # ##: Print results.
        logger.info("Hyperparameter search complete!")
        logger.info(f"Number of completed trials: {summary['num_completed_trials']}")
        logger.info(f"Best trial number: {summary['best_trial_number']}")
        if best_value is not None:
            logger.info(f"Best objective value (mean reward): {best_value:.4f}")
        else:
            logger.info("Best objective value (mean reward): N/A")
        logger.info(f"Best hyperparameters: {json.dumps(summary['best_hyperparameters'], indent=2)}")

    def _prepare_trial_config(self, params: Dict[str, Any], trial_number: int) -> tuple[Dict[str, Any], list[str]]:
        """
        Prepare the configuration for a specific trial using sampled parameters.

        Creates a trial-specific configuration by merging the sampled hyperparameters
        into the base configuration. Adds metadata about the trial.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of sampled hyperparameters for the current trial.
        trial_number : int
            The number of the current Optuna trial.

        Returns
        -------
        tuple[Dict[str, Any], list[str]]
            A tuple containing the configuration dictionary and a list of AIM tags.
        """
        if self.base_config is None:
            raise RuntimeError("Base config is not set before preparing trial config.")

        experiment_config = self._create_experiment_config(self.base_config, params)
        aim_tags = [
            "hyperparameter-search",
            "trial",
            str(self.search_name),
            f"trial_{trial_number}",
            self.base_config.get("agent", {}).get("agent_type", "unknown_agent"),
        ]

        # ##: Add trial info and sampled params to config.
        experiment_config["_trial_info"] = {"number": trial_number}
        experiment_config["sampled_hyperparameters"] = params
        experiment_config["aim_experiment_name"] = f"hyperparameter_{self.search_name}_trial_{trial_number}"

        logger.info(f"\n--- Running Trial {trial_number} ---")
        logger.info(f"Parameters: {params}")
        logger.info(f"AIM Run Name: {experiment_config['aim_experiment_name']}")

        return experiment_config, list(set(aim_tags))

    def _execute_trial(self, trial: Trial, experiment_config: Dict[str, Any], aim_tags: list[str]) -> float:
        """
        Execute the experiment for the trial and handle pruning.

        Executes a single trial using the trial-specific configuration.
        Uses the passed Optuna trial object for reporting values and checking for pruning signals.

        Parameters
        ----------
        trial : optuna.Trial
            The current Optuna trial object.
        experiment_config : Dict[str, Any]
            Dictionary containing the experiment configuration.
        aim_tags : list[str]
            List of AIM tags for the experiment.

        Returns
        -------
        float
            The objective value (e.g., mean reward) for the trial.

        Raises
        ------
        TrialPruned
            If Optuna signals that the trial should be pruned based on reported values.
        """

        # ##: Define the pruning callback using the passed trial object.
        def pruning_callback(step: int, value: float):
            """
            Callback passed to ExperimentRunner for intermediate reporting.

            Parameters
            ----------
            step : int
                The current step number.
            value : float
                The current value (e.g., mean reward) to report.
            """
            trial.report(value, step)
            if trial.should_prune():
                raise TrialPruned()

        try:
            # ##: Run the actual experiment.
            experiment_results = self.experiment_runner.run_experiment(
                experiment_config=ExperimentConfig(**experiment_config),
                pruning_callback=pruning_callback,
                aim_tags=aim_tags,
            )

            # ##: Determine the final objective value to return to Optuna.
            objective_value = experiment_results.get("final_mean_reward", experiment_results.get("mean_reward", 0.0))

            logger.info(f"--- Trial {trial.number} Completed ---")
            logger.info(f"Final Objective Value: {objective_value:.4f}")
            return objective_value  # Return the final value for this trial

        except TrialPruned:
            raise

    def _objective(self, trial: Trial, params: Dict[str, Any]) -> float:
        """
        Objective function called by OptunaManager for each trial.

        Prepares configuration and executes the experiment trial.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object (passed by OptunaManager's wrapper).
        params : Dict[str, Any]
            Sampled hyperparameters for this trial (passed by OptunaManager's wrapper).

        Returns
        -------
        float
            Mean reward (objective value) for the trial.

        Raises
        ------
        TrialPruned
            If the trial is pruned during execution.
        Exception
            If any other error occurs during trial execution.
        """
        experiment_config, aim_tags = self._prepare_trial_config(params, trial.number)
        return self._execute_trial(trial, experiment_config, aim_tags)

    @staticmethod
    def _create_experiment_config(base_config: Dict[str, Any], hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an experiment configuration from a base configuration and hyperparameters.

        Parameters
        ----------
        base_config : Dict[str, Any]
            Base configuration dictionary.
        hyperparameters : Dict[str, Any]
            Dictionary of hyperparameters to merge, with dot notation paths.

        Returns
        -------
        Dict[str, Any]
            Combined experiment configuration with hyperparameters merged into the base config.

        Examples
        --------
        >>> base = {"agent": {"lr": 0.01}, "env": {}}
        >>> params = {"agent.lr": 0.001, "env.name": "CartPole-v1"}
        >>> HyperparameterSearch._create_experiment_config(base, params)
        {'agent': {'lr': 0.001}, 'env': {'name': 'CartPole-v1'}}
        """
        experiment_config = base_config.copy()

        for param_path, value in hyperparameters.items():
            components = param_path.split(".")

            current_level = experiment_config
            for i, component in enumerate(components[:-1]):
                if component not in current_level:
                    current_level[component] = {}
                elif not isinstance(current_level[component], dict):
                    raise ValueError(
                        f"Configuration conflict: Expected dict at '{'.'.join(components[:i+1])}', found {type(current_level[component])}"
                    )
                current_level = current_level[component]

            final_key = components[-1]
            current_level[final_key] = value

        return experiment_config

    def run_search(self, search_config_path: Union[str, Path], n_trials: int = 20) -> Optional[Dict[str, Any]]:
        """
        Run a hyperparameter search using Optuna.

        Orchestrates the search process by calling helper methods for logging, config loading,
        study creation, optimization, and result summarization.

        Parameters
        ----------
        search_config_path : str | Path
            Path to the search configuration file.
        n_trials : int, default=20
            Number of trials to run.

        Returns
        -------
        Dict[str, Any] | None
            Dictionary containing the search summary.

        Raises
        ------
        FileNotFoundError
            If configuration files are not found.
        ValueError
            If configurations are invalid.
        RuntimeError
            If Optuna study or optimization encounters critical errors.
        """
        search_aim_logger = None
        try:
            search_name = Path(search_config_path).stem
            search_aim_logger = self._setup_search_logging(search_name, n_trials)
            self._load_configs(search_config_path, search_aim_logger)

            # ##: Initialize OptunaManager.
            self.optuna_manager = OptunaManager(search_name=search_name, results_dir=self.results_dir)
            self.optuna_manager.search_config = self.search_config

            # ##: Run optimization using OptunaManager.
            self.optuna_manager.run_optimization(self._objective, n_trials=n_trials)

            # ##: Get summary and log results.
            summary = self.optuna_manager.get_study_summary()
            self._summarize_and_log_results(summary, search_aim_logger)
            return summary

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            logger.error(f"Hyperparameter search failed: {exc}\n{format_exc()}")
            if search_aim_logger:
                search_aim_logger.log_text(f"Search failed: {exc}\n{format_exc()}", name="error_log")
            raise
        finally:
            if search_aim_logger:
                search_aim_logger.close()
            self.optuna_manager = None
