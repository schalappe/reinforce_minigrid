# -*- coding: utf-8 -*-
"""
Hyperparameter search for reinforcement learning experiments using Optuna.

This module provides a class for running hyperparameter optimization studies using the Optuna framework.
It supports parallel execution, pruning of underperforming trials, and visualization of results.
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from traceback import format_exc
from typing import Any, Dict, Optional, Union

from loguru import logger
from optuna import Study, create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.trial import Trial, TrialState

from reinforce.configs.manager.reader import YamlReader
from reinforce.configs.models import ExperimentConfig
from reinforce.experiments.experiment_runner import ExperimentRunner
from reinforce.utils.logger import AimLogger, setup_logger

setup_logger()


class HyperparameterSearch:
    """
    Search over hyperparameters for reinforcement learning experiments using Optuna.

    This class implements an efficient hyperparameter optimization using Optuna framework
    with support for parallel execution, pruning, and visualization.
    """

    experiment_runner = ExperimentRunner()
    results_dir = Path("outputs/hyperparameter_search")
    results_dir.mkdir(parents=True, exist_ok=True)
    study: Optional[Study] = None
    search_config: Optional[Dict[str, Any]] = None
    base_config: Optional[Dict[str, Any]] = None
    search_name: Optional[str] = None

    def _setup_search_logging(self, search_name: str, n_trials: int) -> AimLogger:
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
        AimLogger
            An AimLogger instance if the initialization was successful.

        Raises
        ------
        Exception
            If there are issues with initializing the AimLogger.
        """
        search_aim_logger = AimLogger(
            experiment_name=f"hyperparameter_{search_name}", tags=["hyperparameter-search", "summary", search_name]
        )
        search_aim_logger.log_params({"n_trials": n_trials}, prefix="search_setup")
        self.search_name = search_name

        return search_aim_logger

    def _load_configs(self, search_config_path: Union[str, Path], search_aim_logger: AimLogger) -> None:
        """
        Loads the hyperparameter search configuration.

        Parameters
        ----------
        search_config_path : Union[str, Path]
            Path to the hyperparameter search configuration file.
        search_aim_logger : AimLogger
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
        except Exception as exc:
            logger.error(f"Error loading search config: {exc}")
            raise

    def _create_or_load_study(self) -> None:
        """
        Create or load the Optuna study with storage and pruner.

        Creates an Optuna study for hyperparameter optimization, using persistent storage if available and
        a MedianPruner for early stopping of unpromising trials. The study is loaded from a SQLite database
        if it exists, allowing the optimization process to be resumed.

        Raises
        ------
        RuntimeError
            If the search name is not set before creating the study.
        ImportError
            If the database backend for Optuna storage is not installed.
        Exception
            If there are issues creating the study.
        """
        if not self.search_name:
            raise RuntimeError("Search name not set before creating study.")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        storage_name = f"sqlite:///{self.results_dir}/{self.search_name}.db"
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)

        try:
            self.study = create_study(
                study_name=self.search_name,
                storage=storage_name,
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
            )
            logger.info(f"Using persistent storage for study: {storage_name}")
        except ImportError as exc:
            logger.error(f"Database backend for Optuna storage not installed: {exc}")
            logger.warning("Using in-memory storage instead.")
            self.study = create_study(study_name=self.search_name, direction="maximize", pruner=pruner)

    def _run_optimization(self, n_trials: int, search_aim_logger: AimLogger) -> None:
        """
        Run the Optuna optimization process.

        Executes the Optuna optimization process, sampling hyperparameters and evaluating the objective function
        for each trial. The optimization process is monitored and any exceptions are logged to AIM, if available.

        Parameters
        ----------
        n_trials : int
            Number of trials to run in the hyperparameter search.
        search_aim_logger : AimLogger
            AimLogger instance for logging.

        Raises
        ------
        RuntimeError
            If the study is not initialized before running optimization.
        Exception
            If the Optuna optimization process fails.
        """
        if not self.study:
            raise RuntimeError("Study not initialized before running optimization.")

        try:
            self.study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        except Exception as exc:
            logger.error(f"Optuna optimization failed: {exc}")
            search_aim_logger.log_text(f"Optimization failed: {exc}\n{format_exc()}", name="error_log")
            raise

    def _summarize_and_save_results(self, search_aim_logger: AimLogger) -> Dict[str, Any]:
        """
        Prepare, save, and log the search summary and visualizations.

        Collects the results of the hyperparameter search, including the best trial and all completed trials.
        The results are saved to a JSON file and logged to AIM, if available. Additionally, visualizations
        of the optimization process are generated and saved.

        Parameters
        ----------
        search_aim_logger : AimLogger
            AimLogger instance for logging.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the search summary.

        Raises
        ------
        RuntimeError
            If the study is not available for summarization.
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
            logger.info("No best trial found (optimization might have failed or yielded no results).")

        all_trials = [
            {"number": trial.number, "value": trial.value, "params": trial.params}
            for trial in self.study.trials
            if trial.state == TrialState.COMPLETE
        ]

        summary = {
            "num_completed_trials": len(all_trials),
            "best_trial_number": best_trial_num,
            "best_mean_reward": best_value,
            "best_hyperparameters": best_params,
            "all_completed_trials": all_trials,
            "study_name": self.study.study_name,
            "direction": str(self.study.direction),
        }

        # ##: Log summary to AIM.
        search_aim_logger.log_params(summary, prefix="search_summary")
        if best_value is not None:
            search_aim_logger.log_metric("best_mean_reward", best_value)

        # ##: Print results.
        logger.info("Hyperparameter search complete!")
        logger.info(f"Number of trials: {summary['num_completed_trials']}")
        logger.info(f"Best trial: {summary['best_trial_number']}")
        if summary["best_mean_reward"] is not None:
            logger.info(f"Best mean reward: {summary['best_mean_reward']:.4f}")
        else:
            logger.info("Best mean reward: N/A")
        logger.info(f"Best hyperparameters: {json.dumps(summary['best_hyperparameters'], indent=2)}")
        logger.info(f"Visualization plots saved to: {self.results_dir / 'visualizations'}")

        return summary

    def run_search(self, search_config_path: Union[str, Path], n_trials: int = 20) -> Optional[Dict[str, Any]]:
        """
        Run a hyperparameter search using Optuna.

        Orchestrates the search process by calling helper methods for logging, config loading,
        study creation, optimization, and result summarization.

        Parameters
        ----------
        search_config_path : Union[str, Path]
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
            search_aim_logger = self._setup_search_logging(Path(search_config_path).stem, n_trials)
            self._load_configs(search_config_path, search_aim_logger)

            # ##: Create or load study.
            self._create_or_load_study()
            self._run_optimization(n_trials, search_aim_logger)

            return self._summarize_and_save_results(search_aim_logger)

        except Exception as exc:
            if not isinstance(exc, RuntimeError) or "Optuna optimization failed" not in str(exc):
                logger.error(f"Hyperparameter search failed: {exc}\n{format_exc()}")

            if search_aim_logger:
                search_aim_logger.log_text(f"Search failed: {exc}\n{format_exc()}", name="error_log")
            raise
        finally:
            if search_aim_logger:
                search_aim_logger.close()

    def _sample_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters for the current trial based on search config.

        Samples hyperparameters from the search space defined in the search configuration ile. The hyperparameters
        can be categorical, float, or integer, and can be sampled on a log scale.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial object.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the sampled hyperparameters.

        Raises
        ------
        ValueError
            If the configuration for a parameter is invalid.
        Exception
            If there are errors during the sampling process.
        """
        params = {}
        if not self.search_config or "hyperparameters" not in self.search_config:
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
            except Exception as exc:
                logger.error(f"Error suggesting parameter {param_path}: {exc}")
                raise

        return params

    def _prepare_trial_config(self, params: Dict[str, Any], trial: Trial) -> tuple[Dict[str, Any], list[str]]:
        """
        Prepare and save the configuration for a specific trial.

        Creates a trial-specific configuration by merging the sampled hyperparameters into the base configuration.
        The resulting configuration is saved to a YAML file. Additionally, metadata about the trial is added
        to the configuration.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of sampled hyperparameters for the current trial.
        trial : optuna.Trial
            Current trial object.

        Returns
        -------
        tuple[Dict[str, Any], list[str]]
            A tuple containing the configuration dictionary and a list of AIM tags.
        """
        experiment_config = self._create_experiment_config(self.base_config, params)
        aim_tags = [
            "hyperparameter-search",
            "trial",
            str(self.search_name),
            f"trial_{trial.number}",
            self.base_config.get("agent", {}).get("agent_type", "unknown_agent"),
        ]

        # ##: Add trial info and sampled params to config.
        experiment_config["_trial_info"] = {"number": trial.number, "optuna_params": trial.params}
        experiment_config["sampled_hyperparameters"] = params
        experiment_config["aim_experiment_name"] = f"hyperparameter_{self.search_name}_trial_{trial.number}"

        logger.info(f"\n--- Running Trial {trial.number} ---")
        logger.info(f"Parameters: {params}")
        logger.info(f"AIM Run Name: {experiment_config['aim_experiment_name']}")

        return experiment_config, list(set(aim_tags))

    def _execute_trial(self, trial: Trial, experiment_config: Dict[str, Any], aim_tags: list[str]) -> float:
        """
        Define pruning callback and execute the experiment for the trial.

        Executes a single trial of the reinforcement learning experiment, using the trial-specific configuration.
        A pruning callback is defined to allow Optuna to prune unpromising trials early.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial object.
        experiment_config : Dict[str, Any]
            Dictionary containing the experiment configuration.
        aim_tags : list[str]
            List of AIM tags for the experiment.

        Returns
        -------
        float
            The objective value (mean reward) for the trial.

        Raises
        ------
        TrialPruned
            If the trial is pruned by the pruning callback.
        """

        def pruning_callback(step: int, value: float):
            """Inner function for pruning callback."""
            try:
                trial.report(value, step)
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at step {step} with value {value:.4f}.")
                    raise TrialPruned()
            except TrialPruned:
                raise
            except Exception as error:
                logger.error(f"Error in pruning callback for trial {trial.number} (step {step}): {error}")

        try:
            experiment_results = self.experiment_runner.run_experiment(
                experiment_config=ExperimentConfig(**experiment_config),
                pruning_callback=pruning_callback,
                aim_tags=aim_tags,
            )

            objective_value = experiment_results.get("final_mean_reward", experiment_results.get("mean_reward", 0.0))
            logger.info(f"--- Trial {trial.number} Completed ---")
            logger.info(f"Final Objective Value: {objective_value:.4f}")
            return objective_value
        except TrialPruned:
            logger.info(f"Trial {trial.number} was pruned.")
            raise

    def _objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        Orchestrates sampling, config preparation, and execution for a single trial.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial object.

        Returns
        -------
        float
            Mean reward (objective value) for the trial.
        """
        params = self._sample_hyperparameters(trial)
        experiment_config, aim_tags = self._prepare_trial_config(params, trial)

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
        >>> base = {"agent": {"lr": 0.01}}
        >>> params = {"agent.lr": 0.001, "env.name": "CartPole"}
        >>> HyperparameterSearch._create_experiment_config(base, params)
        {'agent': {'lr': 0.001}, 'env': {'name': 'CartPole'}}
        """
        experiment_config = base_config.copy()

        for param_path, value in hyperparameters.items():
            components = param_path.split(".")

            current = experiment_config
            for component in components[:-1]:
                if component not in current:
                    current[component] = {}
                current = current[component]

            current[components[-1]] = value

        return experiment_config


def main():
    """
    Entry point for running hyperparameter search from the command line.

    Parses arguments, initializes the HyperparameterSearch class, and runs the search.
    """
    # ##: Parse command line arguments.
    parser = ArgumentParser(description="Run a hyperparameter search for reinforcement learning experiments")
    parser.add_argument("config", help="Path to the search configuration file")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    args = parser.parse_args()

    # ##: Create hyperparameter search.
    search = HyperparameterSearch()

    # ##: Run the search.
    search.run_search(args.config, n_trials=args.trials)


if __name__ == "__main__":
    main()
